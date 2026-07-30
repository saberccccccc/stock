"""Run resumable full-universe CSV/monthly dual-read observation by split."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import load_alpha_rows
from backtest.market_data_contract import (
    ExecutionMarketDataContract,
    configured_candidate_paths,
)
from backtest.open_ledger import infer_ohlc_load_window, load_execution_market_frames
from core.research_protocol import get_split_spec
from run.run_open_ledger_backend_parity_matrix import ALPHA_ROOT, SPLITS


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _report_passed(path):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")).get("status") == "passed"
    except (OSError, ValueError):
        return False


def parse_args(argv=None):
    configured_store, configured_cache = configured_candidate_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-data-dir", default="data/raw")
    parser.add_argument("--forward-data-dir", default="data/forward_raw")
    parser.add_argument(
        "--market-daily-store-root", default=configured_store
    )
    parser.add_argument(
        "--ohlc-monthly-cache-dir", default=configured_cache
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "reports/non_training_closure_20260719/"
            "nt6_market_data_pilot_20260730/md9_dual_read_observation"
        ),
    )
    parser.add_argument("--min-free-memory-gib", type=float, default=3.0)
    parser.add_argument("--estimated-peak-memory-gib", type=float, default=0.75)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    status_path = output / "dual_read_matrix_status.json"
    status = {
        "schema": "execution_market_dual_read_matrix_v1",
        "primary_backend": "monthly",
        "shadow_backend": "csv",
        "runs": [],
        "updated_at": _utc_now(),
    }
    for split in SPLITS:
        report_path = output / f"dual_read.{split}.json"
        record = {"split": split, "report": str(report_path)}
        if _report_passed(report_path):
            record["status"] = "already_complete"
        else:
            free_gib = psutil.virtual_memory().available / (1024**3)
            record["available_memory_gib_before"] = round(free_gib, 3)
            required_before = args.min_free_memory_gib + args.estimated_peak_memory_gib
            if free_gib < required_before:
                record["status"] = "blocked_low_memory"
                record["required_memory_gib_before"] = required_before
                record["reserved_memory_gib"] = args.min_free_memory_gib
                record["estimated_peak_memory_gib"] = args.estimated_peak_memory_gib
                status["runs"].append(record)
                status["status"] = "blocked_low_memory"
                status["updated_at"] = _utc_now()
                _atomic_json(status_path, status)
                print(
                    f"blocked: free_memory={free_gib:.2f} GiB "
                    f"required_before={required_before:.2f} GiB "
                    f"reserve={args.min_free_memory_gib:.2f} GiB next={split}",
                    flush=True,
                )
                return 3
            spec = get_split_spec(split)
            start, end, max_date = spec.command_dates()
            alpha_path = ALPHA_ROOT / split / "alpha_policy.jsonl"
            rows = [
                row
                for row in load_alpha_rows(alpha_path)
                if start <= str(row["date"].date()) <= end
            ]
            codes = sorted({code for row in rows for code in row["codes"]})
            load_start, load_end = infer_ohlc_load_window(
                rows, max_data_date=max_date, execution_lag=0, lookback_days=160
            )
            data_dir = args.forward_data_dir if spec.is_forward else args.research_data_dir
            market_data = ExecutionMarketDataContract(
                backend="monthly",
                market_daily_store_root=args.market_daily_store_root,
                monthly_cache_root=args.ohlc_monthly_cache_dir,
                shadow_backend="csv",
                shadow_report=report_path,
            )
            load_args = SimpleNamespace(
                data_dir=data_dir,
                money_scale=1000.0,
                progress_every=0,
                ohlc_backend=market_data.backend,
                market_daily_store_root=market_data.market_daily_store_root,
                ohlc_monthly_cache_dir=market_data.monthly_cache_root,
                ohlc_shadow_backend=market_data.shadow_backend,
                ohlc_shadow_report=str(market_data.shadow_report),
            )
            started = time.perf_counter()
            frames = load_execution_market_frames(
                load_args,
                codes,
                start_date=load_start,
                end_date=load_end,
            )
            record.update(
                {
                    "status": "completed",
                    "codes": len(codes),
                    "dates": len(frames["open"].index),
                    "start_date": str(load_start.date()),
                    "end_date": str(load_end.date()),
                    "seconds": round(time.perf_counter() - started, 3),
                    "rss_mb": round(psutil.Process().memory_info().rss / (1024**2), 1),
                    "available_memory_gib_after": round(
                        psutil.virtual_memory().available / (1024**3), 3
                    ),
                }
            )
            del frames
            gc.collect()
        status["runs"].append(record)
        status["updated_at"] = _utc_now()
        _atomic_json(status_path, status)
    status["status"] = "passed"
    status["updated_at"] = _utc_now()
    _atomic_json(status_path, status)
    print(status_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
