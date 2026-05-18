#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from data.pipeline import _normalize_ts_code, build_cross_section_dataset, build_inference_samples
from run.recommend_utils import build_recommendation_predictor, generate_query_dates, predict_alpha_with_regime


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-date watchlist ranking and summary")
    parser.add_argument("--watchlist", default="watchlist.txt", help="File with one stock code per line")
    parser.add_argument("--from-date", default=None, help="Start date YYYY-MM-DD. Defaults to 10 dates before latest.")
    parser.add_argument("--to-date", default=None, help="End date YYYY-MM-DD. Defaults to latest available.")
    parser.add_argument("--ndates", type=int, default=10, help="Max number of dates to query (default 10)")
    parser.add_argument("--predictor", default="avg_score", choices=["v9", "gat", "avg_score", "intersection", "top_union_bottom_intersection"])
    parser.add_argument("--top-n", type=int, default=20, help="Top-N threshold for 'selected' count (default 20)")
    parser.add_argument("--signal-top-pct", type=float, default=0.10)
    parser.add_argument("--output", default=None, help="Output CSV prefix. Saves <prefix>_detail.csv and <prefix>_summary.csv")
    parser.add_argument("--v9-checkpoint", default="checkpoints/ultimate_v7_best.pt")
    parser.add_argument("--gat-checkpoint", default="checkpoints/ultimate_v7_gat_best.pt")
    parser.add_argument("--test-stocks", type=int, default=None, help="Limit stocks loaded for fast smoke test.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_watchlist(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    codes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                codes.append(_normalize_ts_code(line))
    return sorted(set(codes))


def build_predictor(args, train_samples, cfg):
    return build_recommendation_predictor(
        args.predictor,
        train_samples,
        cfg,
        v9_checkpoint=args.v9_checkpoint,
        gat_checkpoint=args.gat_checkpoint,
        device=args.device,
        signal_top_pct=args.signal_top_pct,
    )


def score_sample(predictor, sample):
    df, _ = predict_alpha_with_regime(predictor, sample)
    return df[["date", "rank", "code", "alpha", "regime"]].sort_values("rank").reset_index(drop=True)


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    watchlist_codes = load_watchlist(args.watchlist)
    print(f"自选股 ({len(watchlist_codes)} 只): {watchlist_codes}")

    cfg = build_v9_backtest_config()
    dim_cfg = build_v9_backtest_config()
    dim_cfg.test_mode = True
    dim_cfg.test_stocks = 50
    dim_cfg.max_stocks = 50
    print("加载训练样本缓存用于模型维度推断...")
    train_samples, _ = build_cross_section_dataset(dim_cfg, use_cache=False)
    predictor = build_predictor(args, train_samples, dim_cfg)

    if args.test_stocks is not None:
        cfg.test_mode = True
        cfg.test_stocks = args.test_stocks
        cfg.max_stocks = args.test_stocks

    as_of_dates = generate_query_dates(args.from_date, args.to_date, args.ndates)
    print(f"Query dates: {as_of_dates}")
    samples = build_inference_samples(cfg, as_of_dates)

    if not samples:
        print("没有有效推理样本。")
        return

    all_rows = []
    for sample in samples:
        ranked = score_sample(predictor, sample)
        total = len(ranked)
        for code in watchlist_codes:
            row = ranked[ranked["code"] == code]
            if row.empty:
                continue
            item = row.iloc[0]
            pct = 1.0 - (int(item["rank"]) - 1) / (total - 1) if total > 1 else 1.0
            all_rows.append({
                "date": item["date"],
                "code": code,
                "alpha": item["alpha"],
                "rank": int(item["rank"]),
                "total": total,
                "percentile": pct,
                "regime": item.get("regime", ""),
                "selected": int(item["rank"]) <= args.top_n,
            })

    if not all_rows:
        print("自选股在所选日期范围内没有有效预测结果。")
        return

    detail = pd.DataFrame(all_rows).sort_values(["code", "date"]).reset_index(drop=True)
    print(f"\n========== 自选股排名详情 ==========")
    display = detail[["date", "code", "rank", "total", "percentile", "selected"]].copy()
    display["percentile"] = (display["percentile"] * 100).map(lambda x: f"{x:.2f}%")
    display["selected"] = display["selected"].map({True: "★", False: ""})
    print(display.to_string(index=False))

    summary_rows = []
    for code in watchlist_codes:
        cd = detail[detail["code"] == code]
        if cd.empty:
            summary_rows.append({"code": code, "appearances": 0, "avg_rank": 99999, "best_rank": 99999,
                                  "worst_rank": 99999, "avg_percentile": 0.0, "top_n_rate": "0%", "regimes": "未在截面中"})
            continue
        regimes = cd["regime"].value_counts().to_dict()
        regimes_str = ", ".join(f"{k}:{v}" for k, v in sorted(regimes.items(), key=lambda x: -x[1]))
        summary_rows.append({
            "code": code,
            "appearances": len(cd),
            "avg_rank": round(cd["rank"].mean(), 1),
            "best_rank": int(cd["rank"].min()),
            "worst_rank": int(cd["rank"].max()),
            "avg_percentile": round(cd["percentile"].mean() * 100, 1),
            "top_n_rate": f"{cd['selected'].mean() * 100:.0f}%",
            "regimes": regimes_str,
        })

    summary = pd.DataFrame(summary_rows).sort_values("avg_rank").reset_index(drop=True)
    print(f"\n========== 自选股排名统计 (Top N={args.top_n}) ==========")
    print(summary.to_string(index=False))

    if args.output:
        prefix = PROJECT_ROOT / args.output
        prefix.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
        print(f"\n已保存: {prefix}_detail.csv, {prefix}_summary.csv")


if __name__ == "__main__":
    main()
