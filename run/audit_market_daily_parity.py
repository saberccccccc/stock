"""Audit exact CSV/Parquet provider parity over bounded date splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.ohlc_matrix_cache import discover_stock_csvs
from data.providers import (
    CsvMarketDailyBackend,
    DataView,
    MarketDailyProvider,
    ParquetMarketDailyBackend,
)


DEFAULT_SPLITS = (
    ("val_2024", "2024-01-01", "2024-12-31"),
    ("test_2025", "2025-01-01", "2025-12-31"),
    ("forward_2026", "2026-01-01", "2026-12-31"),
)
DEFAULT_FIELDS = ("open", "high", "low", "close", "volume", "money", "factor")
ANCHOR_CODES = ("000001.SZ", "600000.SH", "300750.SZ", "688981.SH")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def select_audit_codes(csv_root: str | Path, count: int) -> list[str]:
    codes = [path.stem for path in discover_stock_csvs(csv_root)]
    if not codes:
        raise ValueError(f"no stock CSVs found under {csv_root}")
    selected = [code for code in ANCHOR_CODES if code in set(codes)]
    remaining = max(int(count) - len(selected), 0)
    if remaining:
        positions = {
            round(index * (len(codes) - 1) / max(remaining - 1, 1))
            for index in range(remaining)
        }
        selected.extend(codes[position] for position in sorted(positions))
    selected = list(dict.fromkeys(selected))
    if len(selected) < int(count):
        seen = set(selected)
        for code in codes:
            if code not in seen:
                selected.append(code)
                seen.add(code)
            if len(selected) >= int(count):
                break
    return selected[: int(count)]


def _view(root: Path, name: str, start: str, end: str) -> DataView:
    return DataView.create(
        name=name,
        physical_root=root,
        feature_warmup_start=start,
        feature_warmup_end=start,
        task_start=start,
        task_end=end,
        evaluation_start=start,
        evaluation_end=end,
        max_data_date=end,
    )


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(str(frame.index.dtype).encode("ascii"))
    digest.update(np.asarray(frame.index.view("i8"), dtype="<i8").tobytes())
    for column in frame.columns:
        digest.update(str(column).encode("utf-8"))
        digest.update(b"\0")
    digest.update(str(frame.dtypes.astype(str).tolist()).encode("ascii"))
    values = np.asarray(frame.to_numpy(dtype=np.float64), dtype="<f8")
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def build_parity_report(
    *,
    csv_root: str | Path,
    parquet_root: str | Path,
    codes: Sequence[str],
    fields: Sequence[str] = DEFAULT_FIELDS,
    splits: Sequence[tuple[str, str, str]] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    csv_root = Path(csv_root).resolve()
    parquet_root = Path(parquet_root).resolve()
    parquet_backend = ParquetMarketDailyBackend(parquet_root)
    code_list = list(codes)
    code_sha256 = hashlib.sha256(
        ("\n".join(code_list) + "\n").encode("utf-8")
    ).hexdigest()
    physical_end = pd.Timestamp(
        parquet_backend.store.active_state()["coverage"]["equity"]["date_end"]
    )
    covered = []
    rows = []
    for name, requested_start, requested_end in splits:
        start = pd.Timestamp(requested_start).normalize()
        end = min(pd.Timestamp(requested_end).normalize(), physical_end)
        if end < start:
            rows.append(
                {
                    "split": name,
                    "status": "not_covered",
                    "requested_start": str(start.date()),
                    "requested_end": str(pd.Timestamp(requested_end).date()),
                    "physical_end": str(physical_end.date()),
                }
            )
            continue
        covered.append((name, start, end))
    if not covered:
        return {
            "schema": "market_daily_provider_parity_v1",
            "status": "incomplete",
            "csv_root": str(csv_root),
            "parquet_root": str(parquet_root),
            "physical_end": str(physical_end.date()),
            "code_count": len(code_list),
            "codes_sha256": code_sha256,
            "codes": code_list if len(code_list) <= 256 else None,
            "fields": list(fields),
            "splits": rows,
        }

    scan_start = min(item[1] for item in covered)
    scan_end = max(item[2] for item in covered)
    csv_provider = MarketDailyProvider(
        data_view=_view(csv_root, "parity_scan", str(scan_start.date()), str(scan_end.date())),
        backend=CsvMarketDailyBackend(csv_root),
    )
    parquet_provider = MarketDailyProvider(
        data_view=_view(
            parquet_root,
            "parity_scan",
            str(scan_start.date()),
            str(scan_end.date()),
        ),
        backend=parquet_backend,
    )
    started = time.perf_counter()
    csv_all = csv_provider.load(
        codes=codes,
        fields=fields,
        start_date=scan_start,
        end_date=scan_end,
    )
    csv_seconds = time.perf_counter() - started
    started = time.perf_counter()
    parquet_all = parquet_provider.load(
        codes=codes,
        fields=fields,
        start_date=scan_start,
        end_date=scan_end,
    )
    parquet_seconds = time.perf_counter() - started

    for name, start, end in covered:
        started = time.perf_counter()
        field_records = {}
        for field in fields:
            csv_frame = csv_all[field].loc[start:end]
            parquet_frame = parquet_all[field].loc[start:end]
            pd.testing.assert_frame_equal(
                csv_frame,
                parquet_frame,
                check_exact=True,
                check_dtype=True,
                check_names=True,
            )
            frame = parquet_frame
            field_records[field] = {
                "rows": len(frame),
                "codes": len(frame.columns),
                "missing": int(frame.isna().sum().sum()),
                "sha256": _frame_sha256(frame),
            }
        rows.append(
            {
                "split": name,
                "status": "passed",
                "start_date": str(start.date()),
                "end_date": str(end.date()),
                "comparison": "exact_index_columns_values_dtypes_missing",
                "fields": field_records,
            }
        )
    return {
        "schema": "market_daily_provider_parity_v1",
        "status": "passed" if all(row["status"] == "passed" for row in rows) else "incomplete",
        "csv_root": str(csv_root),
        "parquet_root": str(parquet_root),
        "physical_end": str(physical_end.date()),
        "code_count": len(code_list),
        "codes_sha256": code_sha256,
        "codes": code_list if len(code_list) <= 256 else None,
        "fields": list(fields),
        "scan_start": str(scan_start.date()),
        "scan_end": str(scan_end.date()),
        "csv_seconds": csv_seconds,
        "parquet_seconds": parquet_seconds,
        "scan_strategy": "single_provider_read_then_split_slice",
        "splits": rows,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", default="data/forward_raw")
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--sample-codes", type=int, default=48)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    csv_root = (ROOT / args.csv_root).resolve() if not Path(args.csv_root).is_absolute() else Path(args.csv_root).resolve()
    parquet_root = (ROOT / args.parquet_root).resolve() if not Path(args.parquet_root).is_absolute() else Path(args.parquet_root).resolve()
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    report = build_parity_report(
        csv_root=csv_root,
        parquet_root=parquet_root,
        codes=select_audit_codes(csv_root, args.sample_codes),
    )
    _write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
