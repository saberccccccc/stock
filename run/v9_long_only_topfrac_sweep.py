#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V9 long-only top-fraction sweep."""
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


def main():
    cfg = build_v9_backtest_config()
    runtime = load_backtest_runtime(cfg, use_cache=True)
    v9 = load_dl_predictor(V9_CKPT, runtime.train, runtime.cfg)
    predictor = V9RankPredictor(v9, "v9_raw")

    rows = []
    output_dir = "backtest_results_exp_v9_long_only_topfrac"
    for top_frac in [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]:
        for mode in ["simple_long", "optimizer_projected"]:
            params = ProductionBacktestParams(
                future_len=runtime.cfg.target_horizon,
                portfolio_mode=mode,
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
                label=f"v9_top{int(top_frac * 1000):03d}_{mode}",
                output_dir=output_dir,
                save_full_data=False,
                extra_fields={"top_frac": top_frac, "portfolio": mode},
            )
            rows.append(row)

    save_summary_csv(
        rows,
        PROJECT_ROOT / output_dir / "v9_long_only_topfrac_summary.csv",
        display_columns=[
            "top_frac",
            "portfolio",
            "ann_raw",
            "sharpe_raw",
            "mdd_raw",
            "ann_neu",
            "sharpe_neu",
            "mdd_neu",
        ],
        title="V9 long-only top-fraction sweep",
    )


if __name__ == "__main__":
    main()
