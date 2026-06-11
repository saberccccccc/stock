#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Focused refinement for the V9 3-day persistent signal."""
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


def make_predictor(base_predictor, window, mode):
    if mode == "none":
        return base_predictor
    return PersistentPredictor(base_predictor, window=window, mode=mode)


def run_case(runtime, base_predictor, rows, label, output_dir, **kwargs):
    predictor_mode = kwargs.pop("predictor_mode", "average")
    window = kwargs.pop("window", 3)
    predictor = make_predictor(base_predictor, window, predictor_mode)

    params = ProductionBacktestParams(
        future_len=runtime.cfg.target_horizon,
        portfolio_mode=kwargs.pop("portfolio_mode", "simple_long"),
        top_frac=kwargs.pop("top_frac", 0.045),
        max_weight=kwargs.pop("max_weight", 0.05),
        optimizer_base_mode="simple_long",
        optimizer_dollar_neutral=False,
        optimizer_beta_limit=kwargs.pop("optimizer_beta_limit", 0.30),
        market_timing_mode=kwargs.pop("market_timing_mode", "legacy"),
        market_min_mult=kwargs.pop("market_min_mult", 0.20),
        long_risk_filter=kwargs.pop("long_risk_filter", "none"),
        risk_filter_vol_quantile=kwargs.pop("risk_filter_vol_quantile", 1.0),
        risk_filter_beta_abs_max=kwargs.pop("risk_filter_beta_abs_max", None),
        alpha_vol_power=kwargs.pop("alpha_vol_power", 0.0),
        long_hold_frac=kwargs.pop("long_hold_frac", None),
    )
    if kwargs:
        raise ValueError(f"unused kwargs: {kwargs}")

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
            "predictor_mode": predictor_mode,
            "window": window,
            "top_frac": params.top_frac,
            "portfolio": params.portfolio_mode,
            "alpha_vol_power": params.alpha_vol_power,
            "long_risk_filter": params.long_risk_filter,
            "risk_filter_vol_quantile": params.risk_filter_vol_quantile,
            "risk_filter_beta_abs_max": params.risk_filter_beta_abs_max,
            "long_hold_frac": params.long_hold_frac,
            "market_timing": params.market_timing_mode,
        },
    )
    rows.append(row)


def main():
    parser = argparse.ArgumentParser(description="Refine the V9 3-day persistent long-only candidate.")
    parser.add_argument("--checkpoint", default=V9_CKPT)
    parser.add_argument("--output-dir", default="backtest_results_exp_v9_persistent_refine")
    args = parser.parse_args()

    cfg = build_v9_backtest_config()
    runtime = load_backtest_runtime(cfg, use_cache=True)
    base_model = load_dl_predictor(args.checkpoint, runtime.train, runtime.cfg)
    base_predictor = V9RankPredictor(base_model, "v9_raw")

    rows = []
    for top_frac in [0.04, 0.045, 0.05, 0.055, 0.06]:
        run_case(
            runtime,
            base_predictor,
            rows,
            f"avg_w3_top{int(round(top_frac * 1000)):03d}",
            args.output_dir,
            top_frac=top_frac,
        )

    for alpha_vol_power in [0.25, 0.50]:
        run_case(
            runtime,
            base_predictor,
            rows,
            f"avg_w3_top045_alpha_vol{int(alpha_vol_power * 100):02d}",
            args.output_dir,
            top_frac=0.045,
            alpha_vol_power=alpha_vol_power,
        )
        run_case(
            runtime,
            base_predictor,
            rows,
            f"avg_w3_top050_alpha_vol{int(alpha_vol_power * 100):02d}",
            args.output_dir,
            top_frac=0.050,
            alpha_vol_power=alpha_vol_power,
        )

    for vol_q in [0.80, 0.90]:
        run_case(
            runtime,
            base_predictor,
            rows,
            f"avg_w3_top045_volq{int(vol_q * 100):02d}",
            args.output_dir,
            top_frac=0.045,
            long_risk_filter="vol",
            risk_filter_vol_quantile=vol_q,
        )
        run_case(
            runtime,
            base_predictor,
            rows,
            f"avg_w3_top050_volq{int(vol_q * 100):02d}",
            args.output_dir,
            top_frac=0.050,
            long_risk_filter="vol",
            risk_filter_vol_quantile=vol_q,
        )

    for hold_frac in [0.065, 0.075]:
        run_case(
            runtime,
            base_predictor,
            rows,
            f"avg_w3_top045_hold{int(round(hold_frac * 1000)):03d}",
            args.output_dir,
            top_frac=0.045,
            long_hold_frac=hold_frac,
        )

    save_summary_csv(
        rows,
        PROJECT_ROOT / args.output_dir / "v9_persistent_refine_summary.csv",
        display_columns=[
            "mode",
            "predictor_mode",
            "window",
            "top_frac",
            "portfolio",
            "alpha_vol_power",
            "long_risk_filter",
            "risk_filter_vol_quantile",
            "long_hold_frac",
            "ann_raw",
            "sharpe_raw",
            "mdd_raw",
            "ann_neu",
            "sharpe_neu",
            "mdd_neu",
        ],
        title="V9 3-day persistent refinement",
    )


if __name__ == "__main__":
    main()
