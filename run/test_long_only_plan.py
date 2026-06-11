#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Focused long-only strategy tests.

Runs the main long-only candidate only:
top_union_bottom_intersection + top10% + optimizer_projected_long.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from backtest.predictors import V9GATEnsemblePredictor
from backtest.reports import save_summary_csv
from backtest.runners import ProductionBacktestParams, run_production_backtest_once
from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_v9_gat_predictors


def main():
    cfg = build_v9_backtest_config()
    print("加载数据...")
    runtime = load_backtest_runtime(cfg, use_cache=True)

    print("加载 V9 Transformer 和 GAT...")
    predictors = load_v9_gat_predictors(runtime.train, runtime.cfg)
    shared_cache = {}
    predictor = V9GATEnsemblePredictor(
        predictors.v9_predictor,
        predictors.gat_predictor,
        strategy="top_union_bottom_intersection",
        top_pct=0.10,
        cache=shared_cache,
    )

    tests = [
        {
            "label": "tubi_top10_projected_long_hold15",
            "long_hold_frac": 0.15,
            "alpha_vol_power": 0.0,
        },
        {
            "label": "tubi_top10_projected_long_hold20",
            "long_hold_frac": 0.20,
            "alpha_vol_power": 0.0,
        },
        {
            "label": "tubi_top10_projected_long_hold15_alpha_vol_p0p5",
            "long_hold_frac": 0.15,
            "alpha_vol_power": 0.5,
        },
    ]

    results = []
    output_dir = "backtest_results_exp_long_only_hysteresis"
    for test in tests:
        params = ProductionBacktestParams(
            future_len=runtime.cfg.target_horizon,
            portfolio_mode="optimizer_projected",
            top_frac=0.10,
            optimizer_base_mode="simple_long",
            optimizer_dollar_neutral=False,
            optimizer_beta_limit=0.30,
            long_hold_frac=test["long_hold_frac"],
            alpha_vol_power=test["alpha_vol_power"],
        )
        row, _ = run_production_backtest_once(
            predictor,
            runtime.val,
            runtime.price_dict,
            runtime.vol_dict,
            runtime.cfg,
            params,
            label=test["label"],
            output_dir=output_dir,
            save_full_data=False,
            extra_fields={
                "strategy": "top_union_bottom_intersection",
                "variant": "long_top_10pct",
                "run_top_frac": 0.10,
                "long_hold_frac": test["long_hold_frac"],
                "alpha_vol_power": test["alpha_vol_power"],
            },
        )
        results.append(row)

    save_summary_csv(
        results,
        PROJECT_ROOT / output_dir / "v9_gat_tubi_top10_hysteresis_summary.csv",
        display_columns=[
            "strategy",
            "variant",
            "run_top_frac",
            "long_hold_frac",
            "alpha_vol_power",
            "mode",
            "ann_raw",
            "sharpe_raw",
            "mdd_raw",
            "ann_neu",
            "sharpe_neu",
            "mdd_neu",
            "avg_long_count",
            "avg_short_count",
        ],
        title="Long-only hysteresis focused tests",
    )


if __name__ == "__main__":
    main()
