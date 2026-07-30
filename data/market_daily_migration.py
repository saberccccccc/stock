"""Resumable one-month migration from per-stock CSVs to MarketDailyStore."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.ohlc_matrix_cache import RAW_FIELDS, discover_stock_csvs, source_signature
from data.market_daily_store import (
    DAILY_FIELDS,
    MarketDailyStore,
    validate_market_daily_frame,
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    )
    return digest.hexdigest()


def _period_bounds(period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    value = str(period).strip()
    if len(value) == 4 and value.isdigit():
        start = pd.Timestamp(f"{value}-01-01")
        end = pd.Timestamp(f"{value}-12-31")
        return start, end
    start = pd.Period(value, freq="M").start_time.normalize()
    end = pd.Period(value, freq="M").end_time.normalize()
    return start, end


def _load_progress(
    path: Path,
    *,
    source_root: Path,
    store_root: Path,
    month: str,
    signature: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "source_root": str(source_root),
        "store_root": str(store_root),
        "month": month,
        "source_signature": signature,
    }
    if not path.is_file():
        return {
            "schema": "csv_market_daily_migration_v1",
            **expected,
            "status": "running",
            "completed_dates": [],
            "dates": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"migration progress {key} does not match current request")
    return payload


def extract_csv_month(
    source_root: str | Path,
    month: str,
    *,
    source: str,
    progress_every: int = 1000,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    source_root = Path(source_root).resolve()
    start, end = _period_bounds(month)
    paths = discover_stock_csvs(source_root)
    month_pieces: list[pd.DataFrame] = []
    failures = []
    started = time.perf_counter()
    peak_rss = 0
    try:
        import psutil

        process = psutil.Process()
    except ImportError:
        process = None

    columns = ["trade_date", "code", *RAW_FIELDS, "factor"]
    for index, path in enumerate(paths, start=1):
        try:
            frame = pd.read_csv(path, usecols=columns)
            dates = pd.to_datetime(frame["trade_date"], format="mixed", errors="coerce")
            selected = frame.loc[(dates >= start) & (dates <= end)].copy()
            if selected.empty:
                continue
            selected["trade_date"] = dates.loc[selected.index].dt.normalize()
            month_pieces.append(selected)
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
        if process is not None and (index % max(progress_every, 1) == 0 or index == len(paths)):
            peak_rss = max(peak_rss, int(process.memory_info().rss))
        if progress_every > 0 and index % progress_every == 0:
            print(f"scanned CSV files {index}/{len(paths)}", flush=True)

    if failures:
        raise ValueError(f"failed to read {len(failures)} source CSV files: {failures[:3]}")
    canonical = {}
    month_frame = (
        pd.concat(month_pieces, ignore_index=True)
        if month_pieces
        else pd.DataFrame(columns=columns)
    )
    for trade_date, group in month_frame.groupby("trade_date", sort=True):
        date_key = pd.Timestamp(trade_date).strftime("%Y%m%d")
        canonical[date_key] = validate_market_daily_frame(
            group,
            instrument_type="equity",
            source=source,
        )
    return canonical, {
        "source_files": len(paths),
        "source_bytes": int(sum(path.stat().st_size for path in paths)),
        "dates": sorted(canonical),
        "rows": int(sum(len(frame) for frame in canonical.values())),
        "scan_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss or None,
    }


def _audit_loaded(
    expected_by_date: dict[str, pd.DataFrame],
    store: MarketDailyStore,
    month: str,
) -> dict[str, Any]:
    start, end = _period_bounds(month)
    expected = pd.concat(expected_by_date.values(), ignore_index=True)
    expected = expected.loc[:, DAILY_FIELDS].sort_values(
        ["trade_date", "code"]
    ).reset_index(drop=True)
    actual = store.load(
        instrument_type="equity",
        start_date=start,
        end_date=end,
        fields=[*RAW_FIELDS, "factor", "source"],
    )
    actual = actual.loc[:, DAILY_FIELDS].sort_values(
        ["trade_date", "code"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=True,
        check_dtype=True,
        check_like=False,
    )
    return {
        "status": "passed",
        "rows": len(actual),
        "dates": int(actual["trade_date"].nunique()),
        "codes": int(actual["code"].nunique()),
        "frame_sha256": _frame_hash(actual),
        "comparison": "exact_keys_values_dtypes",
    }


def migrate_csv_month(
    *,
    source_root: str | Path,
    store_root: str | Path,
    month: str,
    source: str = "legacy_csv",
    progress_path: str | Path | None = None,
    progress_every: int = 1000,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    store_root = Path(store_root).resolve()
    paths = discover_stock_csvs(source_root)
    signature_before = source_signature(paths)
    progress_path = (
        Path(progress_path).resolve()
        if progress_path
        else store_root / "migration_manifests" / f"csv_to_parquet_{month.replace('-', '')}.json"
    )
    progress = _load_progress(
        progress_path,
        source_root=source_root,
        store_root=store_root,
        month=month,
        signature=signature_before,
    )
    expected_by_date, scan = extract_csv_month(
        source_root,
        month,
        source=source,
        progress_every=progress_every,
    )
    signature_after = source_signature(discover_stock_csvs(source_root))
    if signature_after != signature_before:
        raise RuntimeError("source CSV signature changed during migration")

    store = MarketDailyStore(store_root)
    commit_started = time.perf_counter()
    completed = set(progress["completed_dates"])
    for date_key, frame in expected_by_date.items():
        if date_key in completed:
            if store.get_partition("equity", date_key) is None:
                raise ValueError(f"progress claims missing partition equity:{date_key}")
            continue
        result = store.commit_partition(
            frame,
            instrument_type="equity",
            source=source,
        )
        progress["dates"][date_key] = {
            "status": result.status,
            "rows": result.row_count,
            "logical_sha256": result.logical_sha256,
            "physical_sha256": result.physical_sha256,
            "path": result.partition_path,
        }
        progress["completed_dates"].append(date_key)
        progress["completed_dates"].sort()
        _write_json_atomic(progress_path, progress)
        completed.add(date_key)
        print(f"committed {date_key}: {result.status} rows={result.row_count}", flush=True)

    audit = _audit_loaded(expected_by_date, store, month)
    progress.update(
        {
            "status": "completed",
            "scan": scan,
            "commit_seconds": time.perf_counter() - commit_started,
            "audit": audit,
            "source_signature_after": signature_after,
        }
    )
    _write_json_atomic(progress_path, progress)
    return progress
