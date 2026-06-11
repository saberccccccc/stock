#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persistence sweep for the best single V9 predictor."""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from core.config import V9_CKPT
from backtest.predictors import PersistentPredictor
from backtest.reports import save_summary_csv
from backtest.runners import ProductionBacktestParams, run_production_backtest_once
from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_dl_predictor
from run.v9_long_only_optimization import V9RankPredictor


def run_case(runtime, predictor, rows, label, output_dir, top_frac):
    params = ProductionBacktestParams(
        future_len=runtime.cfg.target_horizon,
        portfolio_mode="simple_long",
        top_frac=top_frac,
        max_weight=0.05,
        optimizer_base_mode="simple_long",
        optimizer_dollar_neutral=False,
        optimizer_beta_limit=0.30,
        market_timing_mode="legacy",
    )
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
            "predictor_variant": getattr(predictor, "name", label),
            "top_frac": top_frac,
            "window": getattr(predictor, "window", 1),
            "persistent_mode": getattr(predictor, "mode", "none"),
        },
    )
    rows.append(row)


def main():
    parser = argparse.ArgumentParser(description="Sweep persistent V9 alpha variants.")
    parser.add_argument("--checkpoint", default=V9_CKPT)
    parser.add_argument("--output-dir", default="backtest_results_exp_v9_persistent_sweep")
    parser.add_argument("--top-fracs", default="0.045,0.05,0.055")
    parser.add_argument("--windows", default="3,5,10")
    parser.add_argument("--modes", default="average,composite")
    args = parser.parse_args()

    cfg = build_v9_backtest_config()
    runtime = load_backtest_runtime(cfg, use_cache=True)
    base_model = load_dl_predictor(args.checkpoint, runtime.train, runtime.cfg)
    base_predictor = V9RankPredictor(base_model, "v9_raw")

    top_fracs = [float(x.strip()) for x in args.top_fracs.split(",") if x.strip()]
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    rows = []
    for top_frac in top_fracs:
        run_case(
            runtime,
            base_predictor,
            rows,
            f"v9_raw_top{int(round(top_frac * 1000)):03d}",
            args.output_dir,
            top_frac,
        )

    for mode in modes:
        for window in windows:
            predictor = PersistentPredictor(base_predictor, window=window, mode=mode)
            for top_frac in top_fracs:
                run_case(
                    runtime,
                    predictor,
                    rows,
                    f"v9_persistent_{mode}_w{window}_top{int(round(top_frac * 1000)):03d}",
                    args.output_dir,
                    top_frac,
                )

    save_summary_csv(
        rows,
        PROJECT_ROOT / args.output_dir / "v9_persistent_sweep_summary.csv",
        display_columns=[
            "predictor_variant",
            "persistent_mode",
            "window",
            "top_frac",
            "ann_raw",
            "sharpe_raw",
            "mdd_raw",
            "ann_neu",
            "sharpe_neu",
            "mdd_neu",
        ],
        title="V9 persistent alpha sweep",
    )


if __name__ == "__main__":
    main()
