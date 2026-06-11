#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Layered return diagnostics for a V9 checkpoint."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from backtest.engine import detect_regime
from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_dl_predictor
from run.v9_long_only_optimization import V9RankPredictor


BUCKETS = [
    ("top01", 0.00, 0.01),
    ("top02", 0.00, 0.02),
    ("top05", 0.00, 0.05),
    ("top10", 0.00, 0.10),
    ("top20", 0.00, 0.20),
    ("mid40_60", 0.40, 0.60),
    ("bot20", 0.80, 1.00),
    ("bot10", 0.90, 1.00),
    ("bot05", 0.95, 1.00),
]


def corr(x, y):
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rank01(x):
    order = np.argsort(x)
    out = np.empty_like(order, dtype=np.float64)
    out[order] = np.arange(len(x), dtype=np.float64)
    return out / max(len(x) - 1, 1)


def evaluate_layers(predictor, val_samples, horizon_indices):
    rows = []
    daily_rows = []
    for sample in val_samples:
        y_seq = sample["y_seq"]
        valid = np.isfinite(y_seq).all(axis=1)
        if valid.sum() < 50:
            continue
        regime = detect_regime(sample)
        alpha = predictor.predict_alpha(sample, valid, regime)
        n = len(alpha)
        if n < 50:
            continue
        order_desc = np.argsort(alpha)[::-1]
        date = str(sample.get("date", ""))
        year = date[:4]
        y_valid = y_seq[valid]

        for h_idx in horizon_indices:
            if h_idx >= y_valid.shape[1]:
                continue
            ret = y_valid[:, h_idx]
            finite = np.isfinite(alpha) & np.isfinite(ret)
            if finite.sum() < 50:
                continue
            a = alpha[finite]
            r = ret[finite]
            o = np.argsort(a)[::-1]
            ic = corr(a, r)
            ric = corr(rank01(a), rank01(r))
            daily_rows.append({
                "date": date,
                "year": year,
                "horizon": f"h{h_idx + 1}",
                "ic": ic,
                "rank_ic": ric,
                "n": int(len(a)),
            })
            for name, lo, hi in BUCKETS:
                start = int(len(a) * lo)
                end = max(start + 1, int(len(a) * hi))
                idx = o[start:end]
                rows.append({
                    "date": date,
                    "year": year,
                    "horizon": f"h{h_idx + 1}",
                    "bucket": name,
                    "mean_label": float(np.mean(r[idx])),
                    "count": int(len(idx)),
                    "ic": ic,
                    "rank_ic": ric,
                })
    return pd.DataFrame(rows), pd.DataFrame(daily_rows)


def summarize(layer_df, daily_df):
    layer_summary = (
        layer_df
        .groupby(["horizon", "bucket"], as_index=False)
        .agg(mean_label=("mean_label", "mean"), count=("count", "mean"))
    )
    year_summary = (
        layer_df
        .groupby(["year", "horizon", "bucket"], as_index=False)
        .agg(mean_label=("mean_label", "mean"), count=("count", "mean"))
    )
    ic_summary = (
        daily_df
        .groupby(["horizon"], as_index=False)
        .agg(ic=("ic", "mean"), rank_ic=("rank_ic", "mean"), n_days=("date", "count"))
    )
    return layer_summary, year_summary, ic_summary


def main():
    parser = argparse.ArgumentParser(description="V9 layered return diagnostics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="backtest_results_exp_v9_layer_diagnostics")
    parser.add_argument("--horizons", default="1,3,5,7", help="Human horizons, e.g. 1,3,5,7")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    h_indices = [int(x.strip()) - 1 for x in args.horizons.split(",") if x.strip()]

    cfg = build_v9_backtest_config()
    runtime = load_backtest_runtime(cfg, use_cache=True)
    base = load_dl_predictor(args.checkpoint, runtime.train, runtime.cfg)
    predictor = V9RankPredictor(base, "v9_diag")

    layer_df, daily_df = evaluate_layers(predictor, runtime.val, h_indices)
    layer_summary, year_summary, ic_summary = summarize(layer_df, daily_df)

    layer_df.to_csv(out_dir / "layer_daily.csv", index=False)
    daily_df.to_csv(out_dir / "ic_daily.csv", index=False)
    layer_summary.to_csv(out_dir / "layer_summary.csv", index=False)
    year_summary.to_csv(out_dir / "layer_year_summary.csv", index=False)
    ic_summary.to_csv(out_dir / "ic_summary.csv", index=False)

    print("\n=== IC summary ===")
    print(ic_summary.to_string(index=False))
    print("\n=== Layer summary h5 ===")
    print(layer_summary[layer_summary["horizon"] == "h5"].to_string(index=False))
    print(f"\nSaved diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
