"""Month-sharded dense OHLCV cache backed by MarketDailyStore partitions."""

from __future__ import annotations

import hashlib
import json
import msvcrt
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from backtest.ohlc_matrix_cache import DERIVED_FIELDS, MATRIX_DTYPE, RAW_FIELDS
from data.market_daily_store import MarketDailyStore


CACHE_VERSION = 3
CACHE_MASK_FIELDS = (
    "valid_ohlc_mask",
    "zero_volume_mask",
    "basic_open_tradable_mask",
)


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


@contextmanager
def _cache_lock(path: Path):
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
        raise RuntimeError(f"monthly OHLCV cache writer is already active: {path}") from exc
    try:
        yield
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


class MonthlyOhlcvCache:
    def __init__(self, *, store_root: str | Path, cache_root: str | Path):
        self.store = MarketDailyStore(store_root)
        self.cache_root = Path(cache_root).resolve()

    def _month_root(self, month: Any) -> Path:
        period = pd.Period(month, freq="M")
        return (
            self.cache_root
            / f"year={period.year:04d}"
            / f"month={period.month:02d}"
        )

    def _read_current(self, month_root: Path) -> tuple[dict[str, Any], Path] | None:
        pointer_path = month_root / "CURRENT"
        if not pointer_path.is_file():
            return None
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        generation = (month_root / pointer["generation"]).resolve()
        if generation.parent != month_root.resolve() or not generation.is_dir():
            raise ValueError(f"invalid monthly cache CURRENT: {pointer_path}")
        meta_path = generation / "meta.json"
        if not meta_path.is_file() or _sha256_file(meta_path) != pointer["meta_sha256"]:
            raise ValueError(f"monthly cache metadata hash mismatch: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8")), generation

    def _write_current(self, month_root: Path, generation: Path, meta_sha256: str) -> None:
        pointer = {
            "generation": generation.name,
            "meta_sha256": meta_sha256,
        }
        temp = month_root / f".CURRENT-{uuid.uuid4().hex}.tmp"
        try:
            temp.write_bytes(_canonical_json(pointer))
            temp.replace(month_root / "CURRENT")
        finally:
            temp.unlink(missing_ok=True)

    def _is_current(self, meta: dict[str, Any], source_sha256: str) -> bool:
        return (
            meta.get("version") == CACHE_VERSION
            and meta.get("source_month_index_sha256") == source_sha256
            and tuple(meta.get("fields", ())) == tuple(RAW_FIELDS)
            and meta.get("dtype") == np.dtype(MATRIX_DTYPE).name
        )

    def _validate_files(
        self,
        meta: dict[str, Any],
        generation: Path,
        *,
        verify_hashes: bool,
    ) -> int:
        expected_bytes = int(np.prod(meta["shape"])) * np.dtype(meta["dtype"]).itemsize
        total_bytes = 0
        for field in meta["fields"]:
            record = meta["files"][field]
            path = generation / record["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            if size != expected_bytes or size != int(record["bytes"]):
                raise ValueError(f"monthly cache field size mismatch: {path}")
            if verify_hashes and _sha256_file(path) != record["sha256"]:
                raise ValueError(f"monthly cache field hash mismatch: {path}")
            total_bytes += size
        return total_bytes

    def audit_month(self, month: Any, *, verify_file_hashes: bool = True) -> dict[str, Any]:
        period = pd.Period(month, freq="M")
        source = self.store.month_snapshot(instrument_type="equity", month=period)
        current = self._read_current(self._month_root(period))
        if current is None:
            raise FileNotFoundError(self._month_root(period) / "CURRENT")
        meta, generation = current
        if not self._is_current(meta, source["record"]["sha256"]):
            raise ValueError(f"monthly cache is stale: {period}")
        active_bytes = self._validate_files(
            meta,
            generation,
            verify_hashes=verify_file_hashes,
        )
        return {
            "status": "passed",
            "month": str(period),
            "generation": generation.name,
            "source_month_index_sha256": meta["source_month_index_sha256"],
            "shape": meta["shape"],
            "fields": meta["fields"],
            "active_bytes": active_bytes,
            "file_hashes_verified": bool(verify_file_hashes),
        }

    def active_identity(
        self,
        *,
        start_date: Any,
        end_date: Any,
        ensure_current: bool = False,
    ) -> dict[str, Any]:
        """Return immutable active-generation identity for a requested range."""
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if end < start:
            raise ValueError("end_date precedes start_date")
        months = []
        for period in pd.period_range(start, end, freq="M"):
            if ensure_current:
                self.ensure_month(period)
            current = self._read_current(self._month_root(period))
            if current is None:
                raise FileNotFoundError(self._month_root(period) / "CURRENT")
            meta, generation = current
            source = self.store.month_snapshot(
                instrument_type="equity",
                month=period,
            )
            if not self._is_current(meta, source["record"]["sha256"]):
                raise ValueError(f"monthly cache is stale: {period}")
            months.append(
                {
                    "month": str(period),
                    "generation": generation.name,
                    "source_month_index_sha256": meta[
                        "source_month_index_sha256"
                    ],
                }
            )
        return {
            "schema": "monthly_ohlcv_active_identity_v1",
            "store_root": str(self.store.root),
            "cache_root": str(self.cache_root),
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "months": months,
        }

    def ensure_month(self, month: Any) -> dict[str, Any]:
        period = pd.Period(month, freq="M")
        source = self.store.month_snapshot(instrument_type="equity", month=period)
        source_sha256 = source["record"]["sha256"]
        month_root = self._month_root(period)
        month_root.mkdir(parents=True, exist_ok=True)
        with _cache_lock(month_root / "writer.lock"):
            current = self._read_current(month_root)
            if current is not None and self._is_current(current[0], source_sha256):
                return {"status": "already_current", **current[0]}

            start = period.start_time.normalize()
            end = period.end_time.normalize()
            long = self.store.load(
                instrument_type="equity",
                start_date=start,
                end_date=end,
                fields=RAW_FIELDS,
            )
            if long.empty:
                raise ValueError(f"market-daily month contains no equity rows: {period}")
            dates = pd.DatetimeIndex(sorted(long["trade_date"].unique()), name="trade_date")
            codes = sorted(long["code"].astype(str).unique())
            wide = long.pivot(
                index="trade_date",
                columns="code",
                values=list(RAW_FIELDS),
            ).reindex(index=dates)

            staging = month_root / f".staging-{uuid.uuid4().hex}"
            staging.mkdir()
            files = {}
            try:
                for field in RAW_FIELDS:
                    values = wide[field].reindex(columns=codes).to_numpy(
                        dtype=MATRIX_DTYPE,
                        copy=True,
                    )
                    path = staging / f"{field}.dat"
                    matrix = np.memmap(
                        path,
                        dtype=MATRIX_DTYPE,
                        mode="w+",
                        shape=values.shape,
                    )
                    matrix[:] = values
                    matrix.flush()
                    del matrix
                    files[field] = {
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                meta = {
                    "schema": "monthly_ohlcv_cache_v3",
                    "version": CACHE_VERSION,
                    "month": str(period),
                    "dtype": np.dtype(MATRIX_DTYPE).name,
                    "shape": [len(dates), len(codes)],
                    "fields": list(RAW_FIELDS),
                    "dates": [str(date.date()) for date in dates],
                    "codes": codes,
                    "source_month_index_sha256": source_sha256,
                    "source_month_index_path": source["record"]["path"],
                    "source_partitions": [
                        {
                            "trade_date": item["trade_date"],
                            "logical_sha256": item["logical_sha256"],
                            "physical_sha256": item["physical_sha256"],
                        }
                        for item in source["index"]["partitions"].values()
                    ],
                    "files": files,
                }
                meta_path = staging / "meta.json"
                meta_path.write_bytes(_canonical_json(meta))
                meta_sha256 = _sha256_file(meta_path)
                generation = month_root / f"generation-{meta_sha256}"
                if generation.exists():
                    existing_meta = generation / "meta.json"
                    if _sha256_file(existing_meta) != meta_sha256:
                        raise ValueError(f"monthly cache generation collision: {generation}")
                    self._validate_files(meta, generation, verify_hashes=True)
                    shutil.rmtree(staging)
                else:
                    staging.replace(generation)
                self._write_current(month_root, generation, meta_sha256)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            return {"status": "rebuilt", **meta}

    def _load_current_month(self, month: Any) -> tuple[dict[str, Any], Path]:
        period = pd.Period(month, freq="M")
        source = self.store.month_snapshot(instrument_type="equity", month=period)
        current = self._read_current(self._month_root(period))
        if current is None or not self._is_current(current[0], source["record"]["sha256"]):
            self.ensure_month(period)
            current = self._read_current(self._month_root(period))
        if current is None:
            raise RuntimeError(f"failed to materialize monthly cache: {period}")
        self._validate_files(current[0], current[1], verify_hashes=False)
        return current

    def load(
        self,
        *,
        codes: Sequence[str],
        fields: Sequence[str],
        start_date: Any,
        end_date: Any,
        money_scale: float = 1.0,
    ) -> dict[str, pd.DataFrame]:
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if end < start:
            raise ValueError("end_date precedes start_date")
        requested = tuple(dict.fromkeys(str(field).lower() for field in fields))
        unknown = sorted(
            set(requested) - set(RAW_FIELDS) - set(DERIVED_FIELDS) - set(CACHE_MASK_FIELDS)
        )
        if unknown:
            raise ValueError(f"unknown monthly OHLCV cache fields: {unknown}")
        raw_needed = set(requested) & set(RAW_FIELDS)
        if set(requested) & set(DERIVED_FIELDS):
            raw_needed.add("close")
        if "zero_volume_mask" in requested or "basic_open_tradable_mask" in requested:
            raw_needed.add("volume")
        if "valid_ohlc_mask" in requested or "basic_open_tradable_mask" in requested:
            raw_needed.update(("open", "high", "low", "close"))
        normalized_codes = list(
            dict.fromkeys(str(code).strip().upper() for code in codes if str(code).strip())
        )
        month_frames: dict[str, list[pd.DataFrame]] = {
            field: [] for field in raw_needed
        }
        present_codes: set[str] = set()
        for period in pd.period_range(start, end, freq="M"):
            meta, generation = self._load_current_month(period)
            month_codes = meta["codes"]
            code2idx = {code: index for index, code in enumerate(month_codes)}
            selected = [code for code in normalized_codes if code in code2idx]
            present_codes.update(selected)
            dates = pd.DatetimeIndex(pd.to_datetime(meta["dates"]), name="trade_date")
            date_mask = (dates >= start) & (dates <= end)
            row_idx = np.flatnonzero(date_mask)
            col_idx = np.asarray([code2idx[code] for code in selected], dtype=np.int64)
            for field in raw_needed:
                matrix = np.memmap(
                    generation / meta["files"][field]["path"],
                    dtype=meta["dtype"],
                    mode="r",
                    shape=tuple(meta["shape"]),
                )
                values = (
                    np.asarray(matrix[np.ix_(row_idx, col_idx)], dtype=np.float64)
                    if len(row_idx) and len(col_idx)
                    else np.empty((len(row_idx), len(col_idx)), dtype=np.float64)
                )
                frame = pd.DataFrame(values, index=dates[date_mask], columns=selected)
                frame.columns.name = "code"
                month_frames[field].append(frame)
        ordered_codes = [code for code in normalized_codes if code in present_codes]
        frames = {}
        for field, pieces in month_frames.items():
            frame = pd.concat(pieces, axis=0).sort_index().reindex(columns=ordered_codes)
            if field == "money":
                frame = frame * float(money_scale)
            frames[field] = frame
        if "pre_close" in requested or "pct_chg" in requested:
            frames["pre_close"] = frames["close"].shift(1)
        if "pct_chg" in requested:
            frames["pct_chg"] = (
                frames["close"] / frames["pre_close"] - 1.0
            ).replace([np.inf, -np.inf], np.nan)
        if "valid_ohlc_mask" in requested or "basic_open_tradable_mask" in requested:
            valid_ohlc = (
                frames["open"].notna()
                & frames["high"].notna()
                & frames["low"].notna()
                & frames["close"].notna()
                & (frames["open"] > 0)
                & (frames["high"] > 0)
                & (frames["low"] > 0)
                & (frames["close"] > 0)
            )
            frames["valid_ohlc_mask"] = valid_ohlc
        if "zero_volume_mask" in requested or "basic_open_tradable_mask" in requested:
            zero_volume = frames["volume"].isna() | (frames["volume"] <= 0)
            frames["zero_volume_mask"] = zero_volume
        if "basic_open_tradable_mask" in requested:
            frames["basic_open_tradable_mask"] = valid_ohlc & ~zero_volume
        return {field: frames[field] for field in requested}
