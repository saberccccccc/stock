#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified backtest entry: python run/backtest.py --experiment {ensemble,intersection,concentrated,persistent}"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from backtest.predictors import PersistentPredictor, V9GATEnsemblePredictor, V9GATIntersectionPredictor
from backtest.reports import save_summary_csv
from backtest.runners import ProductionBacktestParams, run_production_backtest_once
from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_v9_gat_predictors

# ── Experiment definitions ────────────────────────────────
EXPERIMENTS = {
    "ensemble": {
        "predictor_class": "ensemble",
        "use_cache": True,
        "strategies": [
            {"name": "union", "signal_top_pct": 0.10, "run_top_frac": 0.25},
            {"name": "avg_score", "signal_top_pct": 0.10, "run_top_frac": 0.10},
            {"name": "top_union_bottom_intersection", "signal_top_pct": 0.10, "run_top_frac": 0.25},
        ],
        "modes": [
            {"label": "simple_ls", "portfolio_mode": "simple_ls"},
            {"label": "optimizer_projected", "portfolio_mode": "optimizer_projected"},
            {"label": "optimizer_mvo_ra0p1", "portfolio_mode": "optimizer_mvo", "mvo_risk_aversion": 0.1},
        ],
        "output_dir": "backtest_results_ensemble",
        "csv_name": "v9_gat_ensemble_modes_summary.csv",
        "display_columns": [
            "strategy", "mode",
            "ann_raw", "sharpe_raw", "mdd_raw",
            "ann_neu", "sharpe_neu", "mdd_neu",
            "avg_long_count", "avg_short_count",
        ],
    },
    "intersection": {
        "predictor_class": "intersection",
        "use_cache": False,
        "strategies": None,
        "modes": [
            {"label": "simple_ls", "portfolio_mode": "simple_ls"},
            {"label": "simple_long", "portfolio_mode": "simple_long"},
            {"label": "optimizer_projected", "portfolio_mode": "optimizer_projected"},
            {"label": "optimizer_mvo_ra1", "portfolio_mode": "optimizer_mvo", "mvo_risk_aversion": 1.0},
            {"label": "optimizer_mvo_ra0p1", "portfolio_mode": "optimizer_mvo", "mvo_risk_aversion": 0.1},
            {"label": "optimizer", "portfolio_mode": "optimizer"},
        ],
        "output_dir": "backtest_results_intersection",
        "csv_name": "v9_gat_intersection_modes_summary.csv",
        "display_columns": [
            "mode",
            "ann_raw", "sharpe_raw", "mdd_raw",
            "ann_neu", "sharpe_neu", "mdd_neu",
            "avg_top_intersection", "avg_bottom_intersection",
        ],
    },
    "concentrated": {
        "predictor_class": "ensemble",
        "use_cache": True,
        "strategies": [
            {"name": "avg_score", "signal_top_pct": 0.10, "run_top_frac": 0.05, "variant": "run_top_5pct"},
            {"name": "avg_score", "signal_top_pct": 0.10, "run_top_frac": 0.03, "variant": "run_top_3pct"},
            {"name": "top_union_bottom_intersection", "signal_top_pct": 0.10, "run_top_frac": 0.05, "variant": "run_top_5pct"},
            {"name": "top_union_bottom_intersection", "signal_top_pct": 0.10, "run_top_frac": 0.03, "variant": "run_top_3pct"},
        ],
        "modes": [
            {"label": "simple_ls", "portfolio_mode": "simple_ls"},
            {"label": "optimizer_projected", "portfolio_mode": "optimizer_projected"},
        ],
        "output_dir": "backtest_results_concentrated",
        "csv_name": "v9_gat_concentrated_modes_summary.csv",
        "display_columns": [
            "strategy", "variant", "mode", "signal_top_pct", "run_top_frac",
            "ann_raw", "sharpe_raw", "mdd_raw",
            "ann_neu", "sharpe_neu", "mdd_neu",
            "avg_long_count", "avg_short_count",
        ],
    },
    "persistent": {
        "predictor_class": "persistent",
        "use_cache": True,
        "strategies": [
            {"name": "avg_score", "signal_top_pct": 0.10, "run_top_frac": 0.10},
            {"name": "top_union_bottom_intersection", "signal_top_pct": 0.10, "run_top_frac": 0.25},
        ],
        "configs": [
            {"persistent": "none", "window": 0, "label_tag": "baseline"},
            {"persistent": "average", "window": 5, "label_tag": "average_w5"},
            {"persistent": "composite", "window": 5, "label_tag": "composite_w5"},
        ],
        "output_dir": "backtest_results_persistent",
        "csv_name": "v9_gat_persistent_summary.csv",
        "display_columns": [
            "strategy", "persistent", "window", "mode",
            "ann_raw", "sharpe_raw", "mdd_raw",
            "ann_neu", "sharpe_neu", "mdd_neu",
            "avg_long_count", "avg_short_count",
        ],
    },
}


# ── CLI ───────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Unified backtest entry for V9+GAT experiments")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()), required=True)
    return parser.parse_args()


# ── Runner ────────────────────────────────────────────────
def run_experiment(exp, runtime, predictors):
    shared_cache = {} if exp.get("use_cache") else None
    results = []

    # ── predictor dispatch ──
    pcls = exp["predictor_class"]

    if pcls == "intersection":
        pred = V9GATIntersectionPredictor(predictors.v9_predictor, predictors.gat_predictor, top_frac=0.10)
        strategy_loop = [{"name": "intersection", "run_top_frac": 0.10}]
    else:
        strategy_loop = exp["strategies"]

    for strategy_cfg in strategy_loop:
        if pcls == "intersection":
            pass  # pred already built above
        elif pcls == "persistent":
            base = V9GATEnsemblePredictor(
                predictors.v9_predictor, predictors.gat_predictor,
                strategy=strategy_cfg["name"], top_pct=strategy_cfg["signal_top_pct"],
                cache=shared_cache,
            )
        else:  # ensemble / concentrated
            pred = V9GATEnsemblePredictor(
                predictors.v9_predictor, predictors.gat_predictor,
                strategy=strategy_cfg["name"], top_pct=strategy_cfg["signal_top_pct"],
                cache=shared_cache,
            )

        # ── inner loop ──
        inner = exp.get("configs") or exp["modes"]
        for inner_cfg in inner:
            # persistent wraps the predictor per config
            if pcls == "persistent":
                if inner_cfg["persistent"] == "none":
                    pred = base
                else:
                    pred = PersistentPredictor(base, window=inner_cfg["window"], mode=inner_cfg["persistent"])
                extra = {"strategy": strategy_cfg["name"], "persistent": inner_cfg["persistent"], "window": inner_cfg["window"]}
                label = f"{strategy_cfg['name']}_{inner_cfg['label_tag']}"
                top_frac = strategy_cfg["run_top_frac"]
                mode = {"portfolio_mode": "simple_ls"}

            elif pcls == "intersection":
                extra = None
                label = inner_cfg["label"]
                top_frac = 0.10
                mode = inner_cfg

            elif exp.get("variant") is not None or "variant" in strategy_cfg:
                # concentrated: variant comes from strategy_cfg
                extra = {
                    "strategy": getattr(pred, "strategy", strategy_cfg["name"]),
                    "signal_top_pct": strategy_cfg["signal_top_pct"],
                    "run_top_frac": strategy_cfg["run_top_frac"],
                    "variant": strategy_cfg["variant"],
                }
                label = f"{inner_cfg['label']}_{strategy_cfg['variant']}"
                top_frac = strategy_cfg["run_top_frac"]
                mode = inner_cfg

            else:
                # ensemble
                extra = {"strategy": getattr(pred, "strategy", strategy_cfg["name"]), "run_top_frac": strategy_cfg["run_top_frac"]}
                label = inner_cfg["label"]
                top_frac = strategy_cfg["run_top_frac"]
                mode = inner_cfg

            params = ProductionBacktestParams(
                future_len=runtime.cfg.target_horizon,
                portfolio_mode=mode["portfolio_mode"],
                top_frac=top_frac,
                optimizer_base_mode=mode.get("optimizer_base_mode", "simple_ls"),
                mvo_risk_aversion=mode.get("mvo_risk_aversion", 1.0),
            )

            row, _ = run_production_backtest_once(
                pred, runtime.val, runtime.price_dict, runtime.vol_dict,
                runtime.cfg, params, label=label,
                output_dir=exp["output_dir"],
                extra_fields=extra,
            )
            results.append(row)
            if pcls == "persistent":
                print(f"  -> raw {row['ann_raw']:.1f}% sharpe {row['sharpe_raw']:.2f} mdd {row['mdd_raw']*100:.1f}%")

    # ── save ──
    out_path = PROJECT_ROOT / exp["output_dir"] / exp["csv_name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_summary_csv(results, out_path, display_columns=exp["display_columns"])


def main():
    args = parse_args()
    exp = EXPERIMENTS[args.experiment]

    cfg = build_v9_backtest_config()
    print("加载数据...")
    runtime = load_backtest_runtime(cfg, use_cache=True)

    print("加载 V9 Transformer 和 GAT...")
    predictors = load_v9_gat_predictors(runtime.train, runtime.cfg)

    run_experiment(exp, runtime, predictors)


if __name__ == "__main__":
    main()
