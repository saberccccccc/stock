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
MANIFEST_SCHEMA = "market_daily_manifest_v2"
MONTH_INDEX_SCHEMA = "market_daily_month_index_v1"
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


def _coverage(monthly_indexes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index in monthly_indexes.values():
        grouped.setdefault(index["instrument_type"], []).append(index)
    result = {}
    for instrument_type, values in grouped.items():
        starts = sorted(item["date_start"] for item in values)
        ends = sorted(item["date_end"] for item in values)
        result[instrument_type] = {
            "date_start": starts[0],
            "date_end": ends[-1],
            "months": len(values),
            "partitions": sum(int(item["partitions"]) for item in values),
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
        self.index_dir = self.root / "indexes"
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

    def _write_immutable_json(
        self,
        payload: dict[str, Any],
        *,
        directory: Path,
        prefix: str,
    ) -> tuple[Path, str]:
        directory.mkdir(parents=True, exist_ok=True)
        content = _canonical_json(payload)
        sha256 = hashlib.sha256(content).hexdigest()
        path = directory / f"{prefix}-{sha256}.json"
        if path.exists():
            if _sha256_file(path) != sha256:
                raise ValueError(f"immutable JSON hash mismatch: {path}")
            return path, sha256
        temp = directory / f".{prefix}-{uuid.uuid4().hex}.tmp"
        try:
            temp.write_bytes(content)
            if _sha256_file(temp) != sha256:
                raise ValueError("staged immutable JSON hash mismatch")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        return path, sha256

    def _write_manifest(self, payload: dict[str, Any]) -> tuple[Path, str]:
        return self._write_immutable_json(
            payload,
            directory=self.manifest_dir,
            prefix="manifest",
        )

    def _write_month_index(
        self,
        payload: dict[str, Any],
        *,
        instrument_type: str,
        month_key: str,
    ) -> tuple[Path, str]:
        directory = (
            self.index_dir
            / instrument_type
            / f"year={month_key[:4]}"
            / f"month={month_key[4:]}"
        )
        return self._write_immutable_json(
            payload,
            directory=directory,
            prefix="index",
        )

    def _load_month_index(self, record: dict[str, Any]) -> dict[str, Any]:
        path = (self.root / record["path"]).resolve()
        if self.index_dir.resolve() not in path.parents:
            raise ValueError("manifest points outside the index directory")
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"month index hash mismatch: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != MONTH_INDEX_SCHEMA:
            raise ValueError("unsupported market-daily month-index schema")
        return payload

    def get_partition(
        self,
        instrument_type: str,
        trade_date: Any,
    ) -> dict[str, Any] | None:
        manifest, _ = self.load_manifest()
        date = pd.Timestamp(trade_date).normalize()
        month_key = f"{instrument_type}:{date.strftime('%Y%m')}"
        index_record = (manifest or {}).get("monthly_indexes", {}).get(month_key)
        if index_record is None:
            return None
        index = self._load_month_index(index_record)
        return index.get("partitions", {}).get(date.strftime("%Y%m%d"))

    def active_state(self) -> dict[str, Any]:
        """Return the hash-verified active root state without scanning partitions."""
        manifest, manifest_path = self.load_manifest()
        if manifest is None or manifest_path is None:
            raise ValueError("market-daily store has no active manifest")
        return {
            "manifest": manifest_path.relative_to(self.root).as_posix(),
            "manifest_sha256": _sha256_file(manifest_path),
            "generation": int(manifest["generation"]),
            "coverage": manifest.get("coverage", {}),
            "active_months": len(manifest.get("monthly_indexes", {})),
        }

    def audit(self, *, verify_physical_hashes: bool = True) -> dict[str, Any]:
        """Validate the complete active manifest graph without loading bar values."""
        manifest, manifest_path = self.load_manifest()
        if manifest is None or manifest_path is None:
            raise ValueError("market-daily store has no active manifest")

        monthly_indexes = manifest.get("monthly_indexes", {})
        audited_months: dict[str, dict[str, Any]] = {}
        seen_partitions: set[tuple[str, str]] = set()
        active_bytes = 0
        active_rows = 0
        active_partitions = 0

        for month_key, record in sorted(monthly_indexes.items()):
            instrument_type = record.get("instrument_type")
            month = str(record.get("month", ""))
            expected_key = f"{instrument_type}:{month.replace('-', '')}"
            if month_key != expected_key:
                raise ValueError(f"month-index key mismatch: {month_key}")

            month_index = self._load_month_index(record)
            if month_index.get("instrument_type") != instrument_type:
                raise ValueError(f"month-index instrument mismatch: {month_key}")
            if month_index.get("month") != month:
                raise ValueError(f"month-index month mismatch: {month_key}")

            partitions = month_index.get("partitions", {})
            if not partitions:
                raise ValueError(f"month index contains no partitions: {month_key}")
            month_rows = 0
            month_bytes = 0
            month_dates = []
            for date_key, partition in sorted(partitions.items()):
                trade_date = pd.Timestamp(partition.get("trade_date")).normalize()
                expected_date_key = trade_date.strftime("%Y%m%d")
                if date_key != expected_date_key:
                    raise ValueError(f"partition date-key mismatch: {month_key}:{date_key}")
                if trade_date.strftime("%Y-%m") != month:
                    raise ValueError(f"partition lies outside month: {month_key}:{date_key}")
                if partition.get("instrument_type") != instrument_type:
                    raise ValueError(f"partition instrument mismatch: {month_key}:{date_key}")
                identity = (str(instrument_type), date_key)
                if identity in seen_partitions:
                    raise ValueError(f"duplicate active partition: {instrument_type}:{date_key}")
                seen_partitions.add(identity)

                path = (self.root / partition["path"]).resolve()
                expected_parent = (
                    self.root
                    / str(instrument_type)
                    / f"year={trade_date.year:04d}"
                    / f"month={trade_date.month:02d}"
                    / f"day={trade_date.day:02d}"
                ).resolve()
                if path.parent != expected_parent:
                    raise ValueError(f"partition path mismatch: {month_key}:{date_key}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                if verify_physical_hashes:
                    physical_sha256 = _sha256_file(path)
                    if physical_sha256 != partition.get("physical_sha256"):
                        raise ValueError(f"partition hash mismatch: {path}")

                metadata = pq.read_metadata(path)
                if metadata.num_rows != int(partition.get("row_count", -1)):
                    raise ValueError(f"partition row-count mismatch: {path}")
                field_names = tuple(metadata.schema.to_arrow_schema().names)
                if field_names != DAILY_FIELDS:
                    raise ValueError(f"partition schema mismatch: {path}")

                size = path.stat().st_size
                rows = int(partition["row_count"])
                month_rows += rows
                month_bytes += size
                month_dates.append(str(trade_date.date()))

            summary = {
                "instrument_type": instrument_type,
                "month": month,
                "date_start": min(month_dates),
                "date_end": max(month_dates),
                "partitions": len(partitions),
                "row_count": month_rows,
                "active_bytes": month_bytes,
            }
            for field in ("instrument_type", "month", "date_start", "date_end", "partitions", "row_count"):
                if record.get(field) != summary[field]:
                    raise ValueError(f"root/month summary mismatch for {month_key}: {field}")
            audited_months[month_key] = summary
            active_rows += month_rows
            active_partitions += len(partitions)
            active_bytes += month_bytes

        expected_coverage = _coverage(monthly_indexes)
        if manifest.get("coverage") != expected_coverage:
            raise ValueError("root manifest coverage mismatch")
        if active_rows != sum(int(item["rows"]) for item in expected_coverage.values()):
            raise ValueError("active row total does not match coverage")
        if active_partitions != sum(int(item["partitions"]) for item in expected_coverage.values()):
            raise ValueError("active partition total does not match coverage")

        return {
            "status": "passed",
            **self.active_state(),
            "coverage": expected_coverage,
            "active_partitions": active_partitions,
            "active_rows": active_rows,
            "active_bytes": active_bytes,
            "physical_hashes_verified": bool(verify_physical_hashes),
        }

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
        month_key = f"{instrument_type}:{trade_date.strftime('%Y%m')}"
        logical_sha256 = _logical_hash(canonical, instrument_type)

        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        with _writer_lock(self.lock_path):
            current, current_path = self.load_manifest()
            monthly_indexes = dict((current or {}).get("monthly_indexes", {}))
            current_month_record = monthly_indexes.get(month_key)
            current_month = (
                self._load_month_index(current_month_record)
                if current_month_record is not None
                else None
            )
            partitions = dict((current_month or {}).get("partitions", {}))
            previous = partitions.get(date_key)
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
                partitions[date_key] = partition
                month_index = {
                    "schema": MONTH_INDEX_SCHEMA,
                    "instrument_type": instrument_type,
                    "month": trade_date.strftime("%Y-%m"),
                    "parent_index_sha256": (
                        current_month_record.get("sha256")
                        if current_month_record is not None
                        else None
                    ),
                    "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "partitions": dict(sorted(partitions.items())),
                }
                index_path, index_sha256 = self._write_month_index(
                    month_index,
                    instrument_type=instrument_type,
                    month_key=trade_date.strftime("%Y%m"),
                )
                partition_values = list(partitions.values())
                monthly_indexes[month_key] = {
                    "instrument_type": instrument_type,
                    "month": trade_date.strftime("%Y-%m"),
                    "path": index_path.relative_to(self.root).as_posix(),
                    "sha256": index_sha256,
                    "date_start": min(item["trade_date"] for item in partition_values),
                    "date_end": max(item["trade_date"] for item in partition_values),
                    "partitions": len(partition_values),
                    "row_count": sum(int(item["row_count"]) for item in partition_values),
                }
                manifest = {
                    "schema": MANIFEST_SCHEMA,
                    "generation": int((current or {}).get("generation", 0)) + 1,
                    "parent_manifest_sha256": (
                        _sha256_file(current_path) if current_path is not None else None
                    ),
                    "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "monthly_indexes": dict(sorted(monthly_indexes.items())),
                    "coverage": _coverage(monthly_indexes),
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
                        "month_index_sha256": index_sha256,
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
        for index_record in (manifest or {}).get("monthly_indexes", {}).values():
            if index_record["instrument_type"] != instrument_type:
                continue
            if pd.Timestamp(index_record["date_end"]) < start:
                continue
            if pd.Timestamp(index_record["date_start"]) > end:
                continue
            month_index = self._load_month_index(index_record)
            for partition in month_index["partitions"].values():
                date = pd.Timestamp(partition["trade_date"]).normalize()
                if start <= date <= end:
                    paths.append(self.root / partition["path"])
        if not paths:
            return pd.DataFrame(columns=requested)
        tables = [pq.read_table(path, columns=requested) for path in sorted(paths)]
        frame = pa.concat_tables(tables).to_pandas()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        if codes is not None:
            selected = {str(code).strip().upper() for code in codes}
            frame = frame.loc[frame["code"].isin(selected)]
        return frame.sort_values(["trade_date", "code"]).reset_index(drop=True)
