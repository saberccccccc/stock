"""Build and audit one monthly dense cache against the Parquet provider."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.ohlc_matrix_cache import DERIVED_FIELDS, RAW_FIELDS
from backtest.monthly_ohlcv_cache import CACHE_MASK_FIELDS, MonthlyOhlcvCache
from data.providers import (
    DataView,
    MarketDailyProvider,
    MonthlyCachedOhlcvProvider,
    ParquetMarketDailyBackend,
)


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(frame.index.view("i8"), dtype="<i8").tobytes())
    digest.update(("\n".join(map(str, frame.columns)) + "\n").encode("utf-8"))
    digest.update(np.ascontiguousarray(frame.to_numpy(dtype="<f8")).tobytes())
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def audit_month(*, store_root: Path, cache_root: Path, month: str) -> dict:
    cache = MonthlyOhlcvCache(store_root=store_root, cache_root=cache_root)
    started = time.perf_counter()
    build = cache.ensure_month(month)
    build_seconds = time.perf_counter() - started
    integrity = cache.audit_month(month)
    meta = build
    start = meta["dates"][0]
    end = meta["dates"][-1]
    view = DataView.create(
        name=f"monthly_cache_{month}",
        physical_root=store_root,
        feature_warmup_start=start,
        feature_warmup_end=start,
        task_start=start,
        task_end=end,
        evaluation_start=start,
        evaluation_end=end,
        max_data_date=end,
    )
    fields = [*RAW_FIELDS, *DERIVED_FIELDS, *CACHE_MASK_FIELDS]
    parquet = MarketDailyProvider(
        data_view=view,
        backend=ParquetMarketDailyBackend(store_root),
    )
    cached = MonthlyCachedOhlcvProvider(
        data_view=view,
        store_root=store_root,
        cache_root=cache_root,
    )
    started = time.perf_counter()
    expected = parquet.load(
        codes=meta["codes"], fields=fields, start_date=start, end_date=end
    )
    parquet_seconds = time.perf_counter() - started
    started = time.perf_counter()
    actual = cached.load(
        codes=meta["codes"], fields=fields, start_date=start, end_date=end
    )
    cache_seconds = time.perf_counter() - started
    field_records = {}
    for field in fields:
        pd.testing.assert_frame_equal(
            expected[field], actual[field], check_exact=True, check_dtype=True
        )
        field_records[field] = {
            "sha256": _frame_hash(actual[field]),
            "missing": int(actual[field].isna().sum().sum()),
        }
    return {
        "schema": "monthly_ohlcv_cache_audit_v1",
        "status": "passed",
        "month": month,
        "start_date": start,
        "end_date": end,
        "codes": len(meta["codes"]),
        "dates": len(meta["dates"]),
        "shape": meta["shape"],
        "build_status": build["status"],
        "build_seconds": build_seconds,
        "parquet_seconds": parquet_seconds,
        "cache_seconds": cache_seconds,
        "integrity": integrity,
        "comparison": "exact_index_columns_values_dtypes_missing",
        "fields": field_records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    resolve = lambda value: Path(value).resolve() if Path(value).is_absolute() else (ROOT / value).resolve()
    report = audit_month(
        store_root=resolve(args.store_root),
        cache_root=resolve(args.cache_root),
        month=args.month,
    )
    output = resolve(args.output)
    _write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
