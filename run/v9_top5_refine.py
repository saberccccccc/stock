#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Focused refinements around the best V9 long-only top5% result."""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from core.config import V9_CKPT
from backtest.reports import save_summary_csv
from backtest.runners import ProductionBacktestParams, run_production_backtest_once
from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_dl_predictor
from run.v9_long_only_optimization import V9RankPredictor


def run_case(runtime, predictor, rows, label, **kwargs):
    output_dir = kwargs.pop("output_dir", "backtest_results_exp_v9_top5_refine")
    base = dict(
        future_len=runtime.cfg.target_horizon,
        portfolio_mode="simple_long",
        top_frac=0.05,
        optimizer_base_mode="simple_long",
        optimizer_dollar_neutral=False,
        optimizer_beta_limit=0.30,
    )
    base.update(kwargs)
    params = ProductionBacktestParams(**base)
    row, _ = run_production_backtest_once(
        predictor,
        runtime.val,
        runtime.price_dict,
        runtime.vol_dict,
        runtime.cfg,
        params,
        label=label,
        output_dir=output_dir,
        save_full_data=False,
        extra_fields={
            "top_frac": params.top_frac,
            "portfolio": params.portfolio_mode,
            "beta_limit": params.optimizer_beta_limit,
            "alpha_vol_power": params.alpha_vol_power,
            "long_hold_frac": params.long_hold_frac,
            "market_timing": params.market_timing_mode,
        },
    )
    rows.append(row)


def main():
    parser = argparse.ArgumentParser(description="Refine V9 long-only top4%-6% candidates.")
    parser.add_argument("--checkpoint", default=V9_CKPT)
    parser.add_argument("--output-dir", default="backtest_results_exp_v9_top5_refine")
    args = parser.parse_args()

    cfg = build_v9_backtest_config()
    runtime = load_backtest_runtime(cfg, use_cache=True)
    v9 = load_dl_predictor(args.checkpoint, runtime.train, runtime.cfg)
    predictor = V9RankPredictor(v9, "v9_raw")

    rows = []
    for top_frac in [0.04, 0.05, 0.06]:
        run_case(runtime, predictor, rows, f"simple_long_top{int(top_frac * 1000):03d}", top_frac=top_frac, output_dir=args.output_dir)
        run_case(
            runtime,
            predictor,
            rows,
            f"simple_long_top{int(top_frac * 1000):03d}_alpha_vol",
            top_frac=top_frac,
            alpha_vol_power=0.5,
            output_dir=args.output_dir,
        )
        run_case(
            runtime,
            predictor,
            rows,
            f"projected_top{int(top_frac * 1000):03d}_beta015",
            portfolio_mode="optimizer_projected",
            top_frac=top_frac,
            optimizer_beta_limit=0.15,
            output_dir=args.output_dir,
        )

    run_case(runtime, predictor, rows, "top050_hold075", portfolio_mode="optimizer_projected", long_hold_frac=0.075, output_dir=args.output_dir)
    run_case(runtime, predictor, rows, "top050_hold100", portfolio_mode="optimizer_projected", long_hold_frac=0.10, output_dir=args.output_dir)
    run_case(
        runtime,
        predictor,
        rows,
        "top050_weak_timing",
        portfolio_mode="optimizer_projected",
        market_timing_mode="dynamic",
        market_min_mult=0.50,
        output_dir=args.output_dir,
    )
    run_case(
        runtime,
        predictor,
        rows,
        "top050_weak_timing_alpha_vol",
        portfolio_mode="optimizer_projected",
        market_timing_mode="dynamic",
        market_min_mult=0.50,
        alpha_vol_power=0.5,
        output_dir=args.output_dir,
    )

    save_summary_csv(
        rows,
        PROJECT_ROOT / args.output_dir / "v9_top5_refine_summary.csv",
        display_columns=[
            "mode",
            "top_frac",
            "portfolio",
            "beta_limit",
            "alpha_vol_power",
            "long_hold_frac",
            "market_timing",
            "ann_raw",
            "sharpe_raw",
            "mdd_raw",
            "ann_neu",
            "sharpe_neu",
            "mdd_neu",
        ],
        title="V9 top5 long-only refinements",
    )


if __name__ == "__main__":
    main()
