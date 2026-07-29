"""Transactional, content-addressed Parquet storage for daily market bars."""

from __future__ import annotations

import hashlib
import json
import msvcrt
import re
import struct
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


STORE_SCHEMA = "market_daily_store_v1"
MANIFEST_SCHEMA = "market_daily_manifest_v1"
DAILY_FIELDS = (
    "trade_date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "money",
    "factor",
    "source",
)
NUMERIC_FIELDS = ("open", "high", "low", "close", "volume", "money", "factor")
EQUITY_CODE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")

ARROW_SCHEMA = pa.schema(
    [
        pa.field("trade_date", pa.date32(), nullable=False),
        pa.field("code", pa.string(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("money", pa.float64(), nullable=False),
        pa.field("factor", pa.float64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ],
    metadata={b"schema": STORE_SCHEMA.encode("ascii")},
)


@dataclass(frozen=True)
class CommitResult:
    status: str
    manifest_path: str
    partition_path: str
    logical_sha256: str
    physical_sha256: str
    row_count: int


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_hash(frame: pd.DataFrame, instrument_type: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"{STORE_SCHEMA}|{instrument_type}|".encode("ascii"))
    for row in frame.itertuples(index=False, name=None):
        trade_date, code, *values, source = row
        digest.update(pd.Timestamp(trade_date).strftime("%Y%m%d").encode("ascii"))
        digest.update(b"\0")
        digest.update(str(code).encode("utf-8"))
        digest.update(b"\0")
        for value in values:
            digest.update(struct.pack(">d", float(value)))
        digest.update(str(source).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _coverage(partitions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for partition in partitions.values():
        grouped.setdefault(partition["instrument_type"], []).append(partition)
    result = {}
    for instrument_type, values in grouped.items():
        dates = sorted(item["trade_date"] for item in values)
        result[instrument_type] = {
            "date_start": dates[0],
            "date_end": dates[-1],
            "partitions": len(values),
            "rows": sum(int(item["row_count"]) for item in values),
        }
    return result


def validate_market_daily_frame(
    frame: pd.DataFrame,
    *,
    instrument_type: str,
    source: str,
) -> pd.DataFrame:
    if instrument_type not in {"equity", "index"}:
        raise ValueError("instrument_type must be equity or index")
    if not source or not str(source).strip():
        raise ValueError("source is required")
    required = set(DAILY_FIELDS) - {"source"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing market-daily columns: {missing}")

    result = frame.loc[:, [column for column in DAILY_FIELDS if column != "source"]].copy()
    result["trade_date"] = pd.to_datetime(
        result["trade_date"], format="mixed", errors="coerce"
    ).dt.normalize()
    if result["trade_date"].isna().any():
        raise ValueError("trade_date contains invalid values")
    dates = result["trade_date"].drop_duplicates()
    if len(dates) != 1:
        raise ValueError("one partition must contain exactly one trade_date")

    result["code"] = result["code"].astype(str).str.strip().str.upper()
    if (result["code"] == "").any():
        raise ValueError("code contains empty values")
    if instrument_type == "equity" and not result["code"].map(EQUITY_CODE.fullmatch).all():
        raise ValueError("equity code must match NNNNNN.SH/SZ/BJ")
    if result.duplicated(["trade_date", "code"]).any():
        raise ValueError("duplicate (trade_date, code) keys")

    for column in NUMERIC_FIELDS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"{column} contains missing or invalid values")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC values must be positive")
    if (result["high"] < result[["open", "close"]].max(axis=1)).any():
        raise ValueError("high is below open or close")
    if (result["low"] > result[["open", "close"]].min(axis=1)).any():
        raise ValueError("low is above open or close")
    if (result[["volume", "money"]] < 0).any().any():
        raise ValueError("volume and money must be non-negative")
    if (result["factor"] <= 0).any():
        raise ValueError("factor must be positive")

    result["source"] = str(source).strip()
    return result.loc[:, DAILY_FIELDS].sort_values("code").reset_index(drop=True)


@contextmanager
def _writer_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"market-daily writer is already active: {path}") from exc
    try:
        yield
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


class MarketDailyStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifest_dir = self.root / "manifests"
        self.staging_dir = self.root / "staging"
        self.current_path = self.root / "CURRENT"
        self.event_path = self.root / "update_events.jsonl"
        self.lock_path = self.root / "locks" / "writer.lock"

    def _read_current_pointer(self) -> dict[str, Any] | None:
        if not self.current_path.is_file():
            return None
        return json.loads(self.current_path.read_text(encoding="utf-8"))

    def load_manifest(self) -> tuple[dict[str, Any] | None, Path | None]:
        pointer = self._read_current_pointer()
        if pointer is None:
            return None, None
        path = (self.root / pointer["manifest"]).resolve()
        if path.parent != self.manifest_dir.resolve():
            raise ValueError("CURRENT points outside the manifest directory")
        if not path.is_file():
            raise FileNotFoundError(path)
        physical = _sha256_file(path)
        if physical != pointer["sha256"]:
            raise ValueError("CURRENT manifest hash mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("unsupported market-daily manifest schema")
        return payload, path

    def _write_current(self, manifest_path: Path, manifest_sha256: str) -> None:
        pointer = {
            "manifest": manifest_path.relative_to(self.root).as_posix(),
            "sha256": manifest_sha256,
        }
        temp = self.current_path.with_suffix(f".tmp-{uuid.uuid4().hex}")
        try:
            temp.write_bytes(_canonical_json(pointer))
            temp.replace(self.current_path)
        finally:
            temp.unlink(missing_ok=True)

    def _write_manifest(self, payload: dict[str, Any]) -> tuple[Path, str]:
        content = _canonical_json(payload)
        sha256 = hashlib.sha256(content).hexdigest()
        path = self.manifest_dir / f"manifest-{sha256}.json"
        if path.exists():
            if _sha256_file(path) != sha256:
                raise ValueError(f"immutable manifest hash mismatch: {path}")
            return path, sha256
        temp = self.manifest_dir / f".manifest-{uuid.uuid4().hex}.tmp"
        try:
            temp.write_bytes(content)
            if _sha256_file(temp) != sha256:
                raise ValueError("staged manifest hash mismatch")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        return path, sha256

    def _append_event(self, payload: dict[str, Any]) -> None:
        with self.event_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(_canonical_json(payload).decode("utf-8"))

    def commit_partition(
        self,
        frame: pd.DataFrame,
        *,
        instrument_type: str,
        source: str,
        allow_revision: bool = False,
    ) -> CommitResult:
        canonical = validate_market_daily_frame(
            frame,
            instrument_type=instrument_type,
            source=source,
        )
        trade_date = pd.Timestamp(canonical["trade_date"].iloc[0]).normalize()
        date_key = trade_date.strftime("%Y%m%d")
        partition_key = f"{instrument_type}:{date_key}"
        logical_sha256 = _logical_hash(canonical, instrument_type)

        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        with _writer_lock(self.lock_path):
            current, current_path = self.load_manifest()
            partitions = dict((current or {}).get("partitions", {}))
            previous = partitions.get(partition_key)
            if previous and previous["logical_sha256"] == logical_sha256:
                return CommitResult(
                    "already_present",
                    str(current_path),
                    str(self.root / previous["path"]),
                    logical_sha256,
                    previous["physical_sha256"],
                    int(previous["row_count"]),
                )
            if previous and not allow_revision:
                raise ValueError(
                    f"partition {partition_key} already exists with different content"
                )

            table = pa.Table.from_pandas(
                canonical,
                schema=ARROW_SCHEMA,
                preserve_index=False,
                safe=True,
            )
            staging = self.staging_dir / f"{partition_key.replace(':', '-')}-{uuid.uuid4().hex}.parquet"
            pq.write_table(
                table,
                staging,
                compression="zstd",
                use_dictionary=["code", "source"],
                write_statistics=True,
            )
            physical_sha256 = _sha256_file(staging)
            relative = (
                Path(instrument_type)
                / f"year={trade_date.year:04d}"
                / f"month={trade_date.month:02d}"
                / f"day={trade_date.day:02d}"
                / f"part-{logical_sha256}.parquet"
            )
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                if destination.exists():
                    existing = pq.read_table(destination, columns=list(DAILY_FIELDS)).to_pandas()
                    if _logical_hash(existing, instrument_type) != logical_sha256:
                        raise ValueError(f"content-address collision at {destination}")
                    staging.unlink()
                    physical_sha256 = _sha256_file(destination)
                else:
                    staging.replace(destination)

                partition = {
                    "instrument_type": instrument_type,
                    "trade_date": str(trade_date.date()),
                    "path": relative.as_posix(),
                    "logical_sha256": logical_sha256,
                    "physical_sha256": physical_sha256,
                    "row_count": int(len(canonical)),
                    "source": str(source).strip(),
                    "schema": STORE_SCHEMA,
                }
                partitions[partition_key] = partition
                manifest = {
                    "schema": MANIFEST_SCHEMA,
                    "generation": int((current or {}).get("generation", 0)) + 1,
                    "parent_manifest_sha256": (
                        _sha256_file(current_path) if current_path is not None else None
                    ),
                    "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "partitions": dict(sorted(partitions.items())),
                    "coverage": _coverage(partitions),
                }
                manifest_path, manifest_sha256 = self._write_manifest(manifest)
                self._write_current(manifest_path, manifest_sha256)
                self._append_event(
                    {
                        "event": "partition_committed",
                        "at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        "partition_key": partition_key,
                        "logical_sha256": logical_sha256,
                        "physical_sha256": physical_sha256,
                        "manifest_sha256": manifest_sha256,
                        "previous_logical_sha256": (
                            previous.get("logical_sha256") if previous else None
                        ),
                        "revision": bool(previous),
                    }
                )
            finally:
                staging.unlink(missing_ok=True)

        return CommitResult(
            "revised" if previous else "committed",
            str(manifest_path),
            str(destination),
            logical_sha256,
            physical_sha256,
            len(canonical),
        )

    def load(
        self,
        *,
        instrument_type: str,
        start_date: Any,
        end_date: Any,
        codes: Sequence[str] | None = None,
        fields: Sequence[str] = NUMERIC_FIELDS,
    ) -> pd.DataFrame:
        manifest, _ = self.load_manifest()
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if end < start:
            raise ValueError("end_date precedes start_date")
        requested = ["trade_date", "code"]
        requested.extend(
            field for field in fields if field not in {"trade_date", "code"}
        )
        unknown = sorted(set(requested) - set(DAILY_FIELDS))
        if unknown:
            raise ValueError(f"unknown market-daily fields: {unknown}")
        paths = []
        for partition in (manifest or {}).get("partitions", {}).values():
            date = pd.Timestamp(partition["trade_date"]).normalize()
            if partition["instrument_type"] == instrument_type and start <= date <= end:
                paths.append(self.root / partition["path"])
        if not paths:
            return pd.DataFrame(columns=requested)
        tables = [pq.read_table(path, columns=requested) for path in sorted(paths)]
        frame = pa.concat_tables(tables).to_pandas()
        if codes is not None:
            selected = {str(code).strip().upper() for code in codes}
            frame = frame.loc[frame["code"].isin(selected)]
        return frame.sort_values(["trade_date", "code"]).reset_index(drop=True)
