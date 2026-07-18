"""Build the temporal-tower dataset.

Example:
    python run/build_temporal_dataset.py --train-start 2018-01-01 --val-start 2024-01-01 --test-start 2025-01-01
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.config import DataConfig
from core.research_protocol import research_end_date_str
from data.temporal_pipeline import build_temporal_cross_section_dataset


def _parse_args():
    parser = argparse.ArgumentParser(description="Build temporal cross-section memmap dataset")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--val-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--end-date", default=research_end_date_str())
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--history-calendar-days", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--target-horizon", type=int, default=5)
    parser.add_argument("--max-horizon", type=int, default=10)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--min-stocks", type=int, default=30)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-tech", action="store_true")
    parser.add_argument("--no-market", action="store_true")
    parser.add_argument("--macro", action="store_true")
    parser.add_argument("--fundamental", action="store_true")
    parser.add_argument(
        "--fundamental-quality-features",
        action="store_true",
        help="Add PIT fundamental quality/staleness flags when --fundamental is enabled.",
    )
    parser.add_argument("--shareholder", action="store_true")
    parser.add_argument("--restricted", action="store_true")
    parser.add_argument("--report-path", default="reports/temporal_dataset_plan.json")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = DataConfig()
    cfg.data_dir = args.data_dir
    cfg.seq_len = args.seq_len
    cfg.target_horizon = args.target_horizon
    cfg.max_horizon = args.max_horizon
    cfg.max_stocks = args.max_stocks
    cfg.min_stocks_per_time = args.min_stocks
    cfg.force_rebuild = args.force_rebuild

    cfg.use_technical_features = not args.no_tech
    cfg.use_market_features = not args.no_market
    cfg.use_macro_features = args.macro
    cfg.use_fundamental_features = args.fundamental
    cfg.use_fundamental_quality_features = args.fundamental_quality_features
    cfg.use_shareholder_features = args.shareholder
    cfg.use_restricted_features = args.restricted

    cfg.temporal_cache_dir = args.cache_dir
    cfg.temporal_lookback = args.lookback
    if args.history_calendar_days is not None:
        cfg.temporal_history_calendar_days = args.history_calendar_days
    cfg.temporal_train_start = args.train_start
    cfg.temporal_val_start = args.val_start
    cfg.temporal_test_start = args.test_start
    cfg.temporal_end_date = args.end_date

    meta = build_temporal_cross_section_dataset(cfg, use_cache=True)

    report = {
        "meta_path": meta["meta_path"],
        "cache_key": meta["cache_key"],
        "n_stocks": len(meta["all_codes"]),
        "n_dates": len(meta["all_dates"]),
        "x_dim": meta["x_dim"],
        "seq_dim": meta["seq_dim"],
        "seq_lookback": meta["seq_lookback"],
        "risk_full_dim": meta["risk_full_dim"],
        "max_horizon": meta["max_horizon"],
        "train_samples": len(meta["train_indices"]),
        "val_samples": len(meta["val_indices"]),
        "test_samples": len(meta.get("test_indices", [])),
        "train_date_range": [
            str(meta["all_dates"][meta["train_indices"][0]]) if meta["train_indices"] else None,
            str(meta["all_dates"][meta["train_indices"][-1]]) if meta["train_indices"] else None,
        ],
        "val_date_range": [
            str(meta["all_dates"][meta["val_indices"][0]]) if meta["val_indices"] else None,
            str(meta["all_dates"][meta["val_indices"][-1]]) if meta["val_indices"] else None,
        ],
        "test_date_range": [
            str(meta["all_dates"][meta["test_indices"][0]]) if meta.get("test_indices") else None,
            str(meta["all_dates"][meta["test_indices"][-1]]) if meta.get("test_indices") else None,
        ],
        "feature_cols": meta["feature_cols"],
        "seq_feature_cols": meta["seq_feature_cols"],
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
