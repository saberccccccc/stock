#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multi-day recommendation: top stocks by average single-day alpha across recent dates."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.runtime import build_v9_backtest_config
from data.pipeline import build_cross_section_dataset, build_inference_samples
from run.recommend_utils import build_recommendation_predictor, generate_query_dates, predict_alpha_with_regime


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-date recommendation by average alpha")
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--ndates", type=int, default=5)
    parser.add_argument("--predictor", default="avg_score", choices=["v9", "gat", "avg_score"])
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--output", default=None)
    parser.add_argument("--v9-checkpoint", default="checkpoints/ultimate_v7_best.pt")
    parser.add_argument("--gat-checkpoint", default="checkpoints/ultimate_v7_gat_best.pt")
    parser.add_argument("--test-stocks", type=int, default=None)
    parser.add_argument("--main-board-only", action="store_true", default=True)
    parser.add_argument("--all-boards", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    cfg = build_v9_backtest_config()
    dim_cfg = build_v9_backtest_config()
    dim_cfg.test_mode = True
    dim_cfg.test_stocks = 50
    dim_cfg.max_stocks = 50
    train_samples, _ = build_cross_section_dataset(dim_cfg, use_cache=False)
    pred = build_recommendation_predictor(
        args.predictor,
        train_samples,
        dim_cfg,
        v9_checkpoint=args.v9_checkpoint,
        gat_checkpoint=args.gat_checkpoint,
        device=args.device,
    )

    if args.test_stocks is not None:
        cfg.test_mode = True
        cfg.test_stocks = args.test_stocks
        cfg.max_stocks = args.test_stocks

    as_of_dates = generate_query_dates(args.from_date, args.to_date, args.ndates)
    print(f"Query dates: {as_of_dates}")
    samples = build_inference_samples(cfg, as_of_dates)

    stock_scores = {}
    date_info = []
    for sample in samples:
        alpha, regime = predict_alpha_with_regime(pred, sample)
        date_info.append({"date": pd.Timestamp(sample["date"]).strftime('%Y-%m-%d'), "n": len(alpha), "regime": regime})
        for i, code in enumerate(sample["codes"]):
            stock_scores.setdefault(code, []).append(float(alpha[i]))

    rows = []
    for code, alphas in stock_scores.items():
        if len(alphas) < 2:
            continue
        rows.append({"code": code, "appearances": len(alphas), "avg_alpha": float(np.mean(alphas))})
    df = pd.DataFrame(rows).sort_values("avg_alpha", ascending=False).reset_index(drop=True)

    if not args.all_boards:
        df = df[~df["code"].str[:3].isin(["688", "300", "301", "689"])].reset_index(drop=True)

    print(f"\nDates: {date_info[0]['date']} ~ {date_info[-1]['date']}")
    print(f"Stocks ranked: {len(df)}  |  Predictor: {getattr(pred, 'name', pred.__class__.__name__)}")
    print(f"\nTop {args.top_n} by average alpha across {len(date_info)} dates:")
    top = df.head(args.top_n)
    for i, row in top.iterrows():
        print(f"  {i+1:3d}. {row['code']}  avg_alpha={row['avg_alpha']:.4f}  days={int(row['appearances'])}")

    if args.output:
        out = PROJECT_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f"{out}.csv", index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out}.csv")


if __name__ == "__main__":
    main()
