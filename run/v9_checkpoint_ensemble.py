#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backtest rank ensembles across existing V9 checkpoints."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from backtest.reports import save_summary_csv
from backtest.runners import ProductionBacktestParams, run_production_backtest_once
from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_dl_predictor


DEFAULT_CHECKPOINTS = [
    "checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt",
    "checkpoints_exp/ultimate_v7_best.pt",
    "checkpoints_exp_topfocus_w005_topret/ultimate_v7_best.pt",
    "checkpoints_exp_topfocus_w005_pairwise_w001_20260530_053300/ultimate_v7_best.pt",
]


def rank01(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks / max(len(x) - 1, 1)


class MultiCheckpointRankEnsemble:
    def __init__(self, predictors, name, weights=None):
        self.predictors = list(predictors)
        self.name = name
        self.weights = np.ones(len(self.predictors), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / (self.weights.sum() + 1e-12)
        self.cache = {}
        self.reset_stats()

    def reset_stats(self):
        self.n_counts = []
        self.model_count = len(self.predictors)

    def predict_alpha(self, sample, valid, regime):
        key = sample["date"]
        if key not in self.cache:
            ranks = []
            for pred in self.predictors:
                alpha = pred.predict_alpha(sample, valid, regime)
                ranks.append(rank01(alpha))
            self.cache[key] = np.average(np.vstack(ranks), axis=0, weights=self.weights).astype(np.float32)
        out = self.cache[key]
        self.n_counts.append(len(out))
        return out

    def stats(self):
        if not self.n_counts:
            return {}
        return {
            "avg_universe": float(np.mean(self.n_counts)),
            "ensemble_models": self.model_count,
        }


def parse_weights(raw, n):
    if not raw:
        return None
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"weights length {len(vals)} != checkpoints length {n}")
    return vals


def run_case(runtime, predictor, rows, label, output_dir, top_frac):
    params = ProductionBacktestParams(
        future_len=runtime.cfg.target_horizon,
        portfolio_mode="simple_long",
        top_frac=top_frac,
        optimizer_base_mode="simple_long",
        optimizer_dollar_neutral=False,
        optimizer_beta_limit=0.30,
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
            "predictor_variant": predictor.name,
            "top_frac": params.top_frac,
            "portfolio": params.portfolio_mode,
        },
    )
    rows.append(row)


def main():
    parser = argparse.ArgumentParser(description="V9 checkpoint rank ensemble backtest.")
    parser.add_argument("--checkpoints", nargs="*", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--weights", default=None, help="Optional comma-separated checkpoint weights.")
    parser.add_argument("--name", default="v9_rank_ensemble")
    parser.add_argument("--output-dir", default="backtest_results_exp_v9_checkpoint_ensemble")
    parser.add_argument("--top-fracs", default="0.045,0.05,0.055")
    args = parser.parse_args()

    checkpoints = [Path(p) for p in args.checkpoints if Path(p).exists()]
    if len(checkpoints) < 2:
        raise RuntimeError(f"Need at least two existing checkpoints, got: {checkpoints}")
    weights = parse_weights(args.weights, len(checkpoints))

    cfg = build_v9_backtest_config()
    runtime = load_backtest_runtime(cfg, use_cache=True)
    predictors = [load_dl_predictor(str(p), runtime.train, runtime.cfg) for p in checkpoints]
    ensemble = MultiCheckpointRankEnsemble(predictors, args.name, weights=weights)

    rows = []
    for top_frac in [float(x.strip()) for x in args.top_fracs.split(",") if x.strip()]:
        run_case(
            runtime,
            ensemble,
            rows,
            f"{args.name}_top{int(round(top_frac * 1000)):03d}",
            args.output_dir,
            top_frac,
        )

    save_summary_csv(
        rows,
        PROJECT_ROOT / args.output_dir / "v9_checkpoint_ensemble_summary.csv",
        display_columns=[
            "predictor_variant",
            "top_frac",
            "portfolio",
            "ann_raw",
            "sharpe_raw",
            "mdd_raw",
            "ann_neu",
            "sharpe_neu",
            "mdd_neu",
            "avg_universe",
            "ensemble_models",
        ],
        title="V9 checkpoint rank ensemble",
    )


if __name__ == "__main__":
    main()
