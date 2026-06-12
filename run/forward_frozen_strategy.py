"""Generate frozen V9 average-w3 alpha rows from forward-only observations."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backtest.engine import detect_regime
from backtest.predictors import PersistentPredictor
from backtest.runtime import build_v9_backtest_config, load_dl_predictor
from core.research_protocol import FORWARD_START_DATE
from data.pipeline import (
    build_cross_section_dataset,
    build_inference_samples,
    samples_from_precomputed_metadata,
)
from run.v9_long_only_optimization import V9RankPredictor


REQUIRED_BROAD_INDICES = (
    "hs300_index.csv",
    "sz50_index.csv",
    "zz500_index.csv",
    "cyb_index.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen forward alpha generator")
    parser.add_argument("--data-dir", default="data/forward_raw")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt",
    )
    parser.add_argument("--start-date", default="2026-05-19")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def latest_csv_date(path, date_column):
    df = pd.read_csv(path, usecols=[date_column])
    return pd.to_datetime(df[date_column]).max()


def assert_market_data_current(data_dir, end_date):
    data_dir = Path(data_dir)
    stale = []
    for filename in REQUIRED_BROAD_INDICES:
        path = data_dir / filename
        if not path.exists():
            stale.append(f"{filename}: missing")
            continue
        latest = latest_csv_date(path, "date")
        if latest < end_date:
            stale.append(f"{filename}: {latest.date()}")

    industry_dir = data_dir / "sw_industry"
    industry_files = sorted(industry_dir.glob("*.csv"))
    if len(industry_files) != 31:
        stale.append(f"sw_industry: expected 31 files, found {len(industry_files)}")
    for path in industry_files:
        latest = latest_csv_date(path, "date")
        if latest < end_date:
            stale.append(f"{path.name}: {latest.date()}")
    if stale:
        details = "; ".join(stale)
        raise ValueError(
            f"Forward market data is older than requested end date {end_date.date()}: "
            f"{details}. Run scripts/update_forward_market_data.py first."
        )


def load_frozen_predictor(checkpoint, device):
    cfg = build_v9_backtest_config()
    result = build_cross_section_dataset(cfg, use_cache=True)
    if not isinstance(result, dict):
        train_samples, _ = result
        schema_samples = train_samples[:1]
    else:
        cfg.low_feat_dim = result.get("low_agg_dim", getattr(cfg, "low_feat_dim", 14))
        schema_meta = dict(result)
        schema_meta["train_indices"] = [result["train_indices"][0]]
        schema_samples = samples_from_precomputed_metadata(schema_meta, "train")
    base = load_dl_predictor(checkpoint, schema_samples, cfg, device)
    raw = V9RankPredictor(base, "v9_raw", cache={})
    return PersistentPredictor(raw, window=3, mode="average")


def main():
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if start < FORWARD_START_DATE:
        raise ValueError(f"Forward start must be on or after {FORWARD_START_DATE.date()}")
    if end < start:
        raise ValueError("end-date precedes start-date")
    assert_market_data_current(args.data_dir, end)

    predictor = load_frozen_predictor(args.checkpoint, args.device)
    cfg = build_v9_backtest_config()
    cfg.data_dir = args.data_dir

    # Two prior trading observations warm the frozen three-day average.
    query_start = start - pd.Timedelta(days=7)
    query_dates = [
        d.strftime("%Y-%m-%d")
        for d in pd.bdate_range(query_start, end)
    ]
    samples = build_inference_samples(cfg, query_dates)

    rows = []
    seen_dates = set()
    for sample in samples:
        date = pd.Timestamp(sample["date"])
        if date in seen_dates:
            continue
        seen_dates.add(date)
        valid = np.ones(len(sample["codes"]), dtype=bool)
        regime = detect_regime(sample)
        alpha = predictor.predict_alpha(sample, valid, regime)
        order = np.argsort(alpha)[::-1]
        if date >= start:
            codes = np.asarray(sample["codes"], dtype=object)
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "codes": codes[order].tolist(),
                    "alpha": np.asarray(alpha)[order].astype(float).tolist(),
                    "n_stocks": int(len(codes)),
                    "regime": regime,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"Saved {len(rows)} forward alpha dates "
        f"({rows[0]['date'] if rows else 'none'} ~ {rows[-1]['date'] if rows else 'none'}): "
        f"{output}"
    )


if __name__ == "__main__":
    main()
