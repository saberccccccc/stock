#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Focused V9 long-only optimization sweep.

This script intentionally avoids the GAT checkpoint so long-only strategy
changes can be tested quickly against the stronger/faster V9 model.
"""
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from core.config import V9_CKPT
from backtest.reports import save_summary_csv
from backtest.runners import ProductionBacktestParams, run_production_backtest_once
from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_dl_predictor


OUTPUT_DIR = "backtest_results_exp_v9_long_only_opt"


class V9RankPredictor:
    """Thin wrapper that exposes stats and a stable name for plain V9 alpha."""

    def __init__(self, base, name="v9", cache=None):
        self.base = base
        self.name = name
        self.cache = {} if cache is None else cache
        self.reset_stats()

    def reset_stats(self):
        self.n_counts = []
        self.long_counts = []

    def predict_alpha(self, sample, valid, regime):
        key = sample["date"]
        if key not in self.cache:
            self.cache[key] = self.base.predict_alpha(sample, valid, regime)
        alpha = self.cache[key]
        n = len(alpha)
        self.n_counts.append(n)
        self.long_counts.append(max(1, int(n * 0.10)) if n else 0)
        return alpha

    def stats(self):
        if not self.n_counts:
            return {}
        n = np.asarray(self.n_counts, dtype=float)
        long = np.asarray(self.long_counts, dtype=float)
        return {
            "avg_universe": float(n.mean()),
            "avg_signal_top10": float(long.mean()),
            "avg_signal_top10_pct": float((long / np.maximum(n, 1)).mean()),
        }


class V9LayeredLongPredictor(V9RankPredictor):
    """Map V9 ranks into top buckets so top names get extra weight, not fewer names."""

    def __init__(self, base, top3_weight=1.5, next7_weight=1.0, cache=None):
        super().__init__(base, name=f"v9_layered_t3x{top3_weight:g}", cache=cache)
        self.top3_weight = float(top3_weight)
        self.next7_weight = float(next7_weight)

    def predict_alpha(self, sample, valid, regime):
        key = sample["date"]
        if key not in self.cache:
            self.cache[key] = self.base.predict_alpha(sample, valid, regime)
        alpha = self.cache[key]
        n = len(alpha)
        self.n_counts.append(n)
        self.long_counts.append(max(1, int(n * 0.10)) if n else 0)
        if n == 0:
            return alpha

        order = np.argsort(alpha)
        score = np.full(n, -1.0, dtype=np.float32)
        k10 = max(1, int(n * 0.10))
        k3 = max(1, int(n * 0.03))
        top10 = order[-k10:]
        top3 = order[-k3:]
        score[top10] = self.next7_weight
        score[top3] = self.top3_weight
        # Tiny within-bucket tiebreaker preserves V9 ordering without changing buckets.
        rank = np.argsort(np.argsort(alpha)).astype(np.float32) / max(n - 1, 1)
        score += 1e-3 * rank
        return score


def make_params(label, **overrides):
    base = dict(
        future_len=None,
        portfolio_mode="optimizer_projected",
        top_frac=0.10,
        optimizer_base_mode="simple_long",
        optimizer_dollar_neutral=False,
        optimizer_beta_limit=0.30,
        lambda_t=0.05,
        max_weight=0.05,
    )
    base.update(overrides)
    return label, base


def main():
    cfg = build_v9_backtest_config()
    print("加载数据...")
    runtime = load_backtest_runtime(cfg, use_cache=True)

    print("加载 V9 checkpoint...")
    v9 = load_dl_predictor(V9_CKPT, runtime.train, runtime.cfg)

    shared_alpha_cache = {}
    predictors = [
        ("v9_raw", V9RankPredictor(v9, "v9_raw", cache=shared_alpha_cache)),
        ("v9_layered_3_10", V9LayeredLongPredictor(v9, top3_weight=1.5, next7_weight=1.0, cache=shared_alpha_cache)),
        ("v9_layered_5_10", V9LayeredLongPredictor(v9, top3_weight=1.25, next7_weight=1.0, cache=shared_alpha_cache)),
    ]

    tests = [
        make_params("projected_beta030"),
        make_params("projected_beta050", optimizer_beta_limit=0.50),
        make_params("projected_beta015", optimizer_beta_limit=0.15),
        make_params("projected_alpha_vol_p05", alpha_vol_power=0.5),
        make_params("projected_hold15", long_hold_frac=0.15),
        make_params("projected_hold20", long_hold_frac=0.20),
        make_params(
            "projected_weak_timing",
            market_timing_mode="dynamic",
            market_min_mult=0.50,
            market_max_mult=1.00,
        ),
        make_params(
            "projected_weak_timing_alpha_vol",
            market_timing_mode="dynamic",
            market_min_mult=0.50,
            market_max_mult=1.00,
            alpha_vol_power=0.5,
        ),
        make_params("simple_long", portfolio_mode="simple_long"),
    ]

    rows = []
    for predictor_label, predictor in predictors:
        for test_label, params_dict in tests:
            label = f"{predictor_label}_{test_label}"
            if list((PROJECT_ROOT / OUTPUT_DIR).glob(f"*_{label}_*_summary.txt")):
                print(f"跳过已完成: {label}")
                continue
            params_dict = dict(params_dict)
            params_dict["future_len"] = runtime.cfg.target_horizon
            params = ProductionBacktestParams(**params_dict)
            row, _ = run_production_backtest_once(
                predictor,
                runtime.val,
                runtime.price_dict,
                runtime.vol_dict,
                runtime.cfg,
                params,
                label=label,
                output_dir=OUTPUT_DIR,
                save_full_data=False,
                extra_fields={
                    "predictor_variant": predictor_label,
                    "strategy_variant": test_label,
                    "top_frac": params.top_frac,
                    "beta_limit": params.optimizer_beta_limit,
                    "alpha_vol_power": params.alpha_vol_power,
                    "long_hold_frac": params.long_hold_frac,
                    "market_timing": params.market_timing_mode,
                },
            )
            rows.append(row)

    save_summary_csv(
        rows,
        PROJECT_ROOT / OUTPUT_DIR / "v9_long_only_optimization_summary.csv",
        display_columns=[
            "predictor_variant",
            "strategy_variant",
            "mode",
            "ann_raw",
            "sharpe_raw",
            "mdd_raw",
            "ann_neu",
            "sharpe_neu",
            "mdd_neu",
            "avg_universe",
            "avg_signal_top10",
        ],
        title="V9 long-only optimization sweep",
    )


if __name__ == "__main__":
    main()
