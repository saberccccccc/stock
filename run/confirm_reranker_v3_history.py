"""Run the frozen V3 and M0 on the untouched post-2024 confirmation period."""

import os
import pickle
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from run.validate_reranker_2024 import (
    load_alpha_rows,
    load_backtest_rows,
    load_close_money,
    load_index_returns,
    make_run_args,
    recompute_adv,
    run_constrained,
    write_alpha,
)
from run.validate_reranker_v3_2024 import build_rows


DATASET = ROOT / (
    "reranker_v3_data_20260615/m0_confirmation_2025_20260518/"
    "reranker_v3_dataset.parquet"
)
MODEL = ROOT / "reranker_models_20260615/regression_v3/reranker_model.pkl"
M0_ALPHA = ROOT / (
    "reranker_data_20260614/m0_confirmation_2025_20260518/"
    "alpha_maxret095.jsonl"
)
OUTPUT = ROOT / "reranker_confirmation_20260615/regression_v3"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(DATASET)
    with MODEL.open("rb") as handle:
        payload = pickle.load(handle)
    m0_rows = load_alpha_rows(M0_ALPHA)
    v3_rows, audit = build_rows(dataset, payload, m0_rows)
    v3_path = OUTPUT / "alpha_v3.jsonl"
    write_alpha(v3_path, v3_rows)
    audit.to_csv(OUTPUT / "fill_audit.csv", index=False)

    row_sets = {
        "m0": load_backtest_rows(M0_ALPHA),
        "v3": load_backtest_rows(v3_path),
    }
    all_codes = sorted(
        {
            code
            for rows in row_sets.values()
            for row in rows
            for code in row["codes"]
        }
    )
    close, money = load_close_money("data/raw", all_codes, 1000.0, 1000)
    adv = recompute_adv(money, 20)
    idx_close, idx_daily = load_index_returns("data/raw", "hs300_index.csv", close.index)
    scenarios = {
        "base": {"adv_cap": 0.05, "cost_mult": 1.0, "lag": 0},
        "cap3": {"adv_cap": 0.03, "cost_mult": 1.0, "lag": 0},
        "cost2x": {"adv_cap": 0.05, "cost_mult": 2.0, "lag": 0},
        "lag1": {"adv_cap": 0.05, "cost_mult": 1.0, "lag": 1},
    }
    summary = []
    for model_name, rows in row_sets.items():
        for scenario_name, scenario in scenarios.items():
            for capital in (500_000.0, 1_000_000.0):
                result, returns, diagnostics = run_constrained(
                    rows,
                    close,
                    adv,
                    0.006,
                    0.10,
                    make_run_args(capital, scenario),
                    idx_close,
                    idx_daily,
                )
                result.update(
                    {
                        "model": model_name,
                        "scenario": scenario_name,
                        "portfolio_value": capital,
                    }
                )
                summary.append(result)
                tag = f"{model_name}_{scenario_name}_{int(capital / 10000)}w"
                returns.to_csv(OUTPUT / f"returns_{tag}.csv", index=False)
                diagnostics.to_csv(OUTPUT / f"diagnostics_{tag}.csv", index=False)
                print(
                    f"{tag}: ann={result['ann']:.2f}% "
                    f"sharpe={result['sharpe']:.3f} "
                    f"mdd={result['mdd'] * 100:.2f}%",
                    flush=True,
                )
    pd.DataFrame(summary).to_csv(OUTPUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
