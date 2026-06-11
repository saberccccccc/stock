#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Micro sweep for V9 long-only portfolio parameters."""
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


def run_case(runtime, predictor, rows, label, output_dir, **kwargs):
    base = dict(
        future_len=runtime.cfg.target_horizon,
        portfolio_mode="simple_long",
        top_frac=0.05,
        max_weight=0.05,
        optimizer_base_mode="simple_long",
        optimizer_dollar_neutral=False,
        optimizer_beta_limit=0.30,
        market_timing_mode="legacy",
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
            "max_weight": params.max_weight,
            "beta_limit": params.optimizer_beta_limit,
            "long_hold_frac": params.long_hold_frac,
            "market_timing": params.market_timing_mode,
        },
    )
    rows.append(row)


def main():
    parser = argparse.ArgumentParser(description="Micro sweep around the best V9 long-only portfolio.")
    parser.add_argument("--checkpoint", default=V9_CKPT)
    parser.add_argument("--output-dir", default="backtest_results_exp_v9_portfolio_micro_sweep")
    args = parser.parse_args()

    cfg = build_v9_backtest_config()
    runtime = load_backtest_runtime(cfg, use_cache=True)
    v9 = load_dl_predictor(args.checkpoint, runtime.train, runtime.cfg)
    predictor = V9RankPredictor(v9, "v9_raw")

    rows = []
    for top_frac in [0.045, 0.050, 0.055]:
        tag = int(round(top_frac * 1000))
        run_case(runtime, predictor, rows, f"simple_top{tag:03d}", args.output_dir, top_frac=top_frac)
        for beta_limit in [0.20, 0.25, 0.30]:
            run_case(
                runtime,
                predictor,
                rows,
                f"projected_top{tag:03d}_beta{int(beta_limit * 100):02d}",
                args.output_dir,
                portfolio_mode="optimizer_projected",
                top_frac=top_frac,
                optimizer_beta_limit=beta_limit,
            )

    for hold_frac in [0.065, 0.075, 0.085, 0.100]:
        run_case(
            runtime,
            predictor,
            rows,
            f"projected_top050_hold{int(round(hold_frac * 1000)):03d}",
            args.output_dir,
            portfolio_mode="optimizer_projected",
            top_frac=0.05,
            optimizer_beta_limit=0.30,
            long_hold_frac=hold_frac,
        )

    save_summary_csv(
        rows,
        PROJECT_ROOT / args.output_dir / "v9_portfolio_micro_sweep_summary.csv",
        display_columns=[
            "mode",
            "top_frac",
            "portfolio",
            "max_weight",
            "beta_limit",
            "long_hold_frac",
            "market_timing",
            "ann_raw",
            "sharpe_raw",
            "mdd_raw",
            "ann_neu",
            "sharpe_neu",
            "mdd_neu",
        ],
        title="V9 long-only portfolio micro sweep",
    )


if __name__ == "__main__":
    main()
