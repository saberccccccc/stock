"""Apply the frozen V4 gate to the existing M0 forward shadow data."""

import os
import pickle
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from run.evaluate_reranker_v3_forward import realized_metrics
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
from run.validate_reranker_v4_2024 import build_rows


FORWARD_DIR = ROOT / "forward_results/m0_v3_20260615"
DATASET = FORWARD_DIR / "reranker_v3_forward.parquet"
MODEL = ROOT / "reranker_models_20260615/gated_v4/reranker_model.pkl"
M0_ALPHA = FORWARD_DIR / "alpha_m0_maxret095.jsonl"
V4_ALPHA = FORWARD_DIR / "alpha_v4_maxret095.jsonl"
EVALUATION_START = pd.Timestamp("2026-05-20")


def main():
    dataset = pd.read_parquet(DATASET)
    with MODEL.open("rb") as handle:
        payload = pickle.load(handle)
    m0_rows = load_alpha_rows(M0_ALPHA)
    v4_rows, audit = build_rows(dataset, payload, m0_rows)
    write_alpha(V4_ALPHA, v4_rows)
    audit.to_csv(FORWARD_DIR / "v4_gate_audit.csv", index=False)
    print(
        f"Active dates={int(audit['active'].sum())}/{len(audit)}",
        flush=True,
    )

    row_sets = {
        "m0": load_backtest_rows(M0_ALPHA),
        "v4": load_backtest_rows(V4_ALPHA),
    }
    all_codes = sorted(
        {
            code
            for rows in row_sets.values()
            for row in rows
            for code in row["codes"]
        }
    )
    close, money = load_close_money("data/forward_raw", all_codes, 1000.0, 1000)
    adv = recompute_adv(money, 20)
    idx_close, idx_daily = load_index_returns(
        "data/forward_raw", "hs300_index.csv", close.index
    )
    summary = []
    for model_name, rows in row_sets.items():
        for capital in (500_000.0, 1_000_000.0):
            result, returns, _ = run_constrained(
                rows,
                close,
                adv,
                0.006,
                0.10,
                make_run_args(
                    capital,
                    {"adv_cap": 0.05, "cost_mult": 1.0, "lag": 0},
                ),
                idx_close,
                idx_daily,
            )
            realized = returns[pd.to_datetime(returns["date"]) >= EVALUATION_START]
            compounded, ann, sharpe, mdd = realized_metrics(realized)
            result.update(
                {
                    "model": model_name,
                    "portfolio_value": capital,
                    "forward_start": str(EVALUATION_START.date()),
                    "forward_days": len(realized),
                    "forward_realized": compounded,
                    "forward_ann": ann * 100.0,
                    "forward_sharpe": sharpe,
                    "forward_mdd": mdd,
                }
            )
            summary.append(result)
            print(
                f"{model_name}_{int(capital / 10000)}w: "
                f"realized={compounded * 100:.2f}% "
                f"sharpe={sharpe:.3f} mdd={mdd * 100:.2f}%",
                flush=True,
            )
    pd.DataFrame(summary).to_csv(
        FORWARD_DIR / "v4_forward_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
