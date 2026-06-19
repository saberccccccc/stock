"""Run the frozen V2 boundary reranker on the strict 2024 portfolio protocol."""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.research_protocol import assert_alpha_rows_within_research
from run.validate_reranker_2024 import (
    load_alpha_rows,
    load_backtest_rows,
    load_close_money,
    load_index_returns,
    load_signal_returns,
    make_run_args,
    recompute_adv,
    run_constrained,
    transform_rows,
    write_alpha,
)


DATASET = ROOT / "reranker_v2_data_20260615/m0_validation_2024/reranker_v2_dataset.parquet"
MODEL = ROOT / "reranker_models_20260615/regression_v2/reranker_model.pkl"
M0_ALPHA = ROOT / "multi_loss_validation_20260614/portfolio/m0_nomulti_e6/alpha_raw.jsonl"
OUTPUT = ROOT / "reranker_validation_20260615/regression_v2"


def build_rows(dataset, payload, m0_rows):
    model = payload["model"]
    features = payload["feature_columns"]
    boundary_start = int(payload["boundary_start"])
    boundary_end = int(payload["boundary_end"])
    replacements = int(payload["max_replacements"])
    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    frame["v2_score"] = model.predict(frame[features])
    by_date = {
        date: group.set_index("code")
        for date, group in frame.groupby("date", sort=False)
    }
    rows = []
    audits = []
    for row in m0_rows:
        date = pd.Timestamp(row["date"])
        original = [str(code) for code in row["codes"]]
        group = by_date[date]
        protected = original[:boundary_start]
        boundary = [
            code
            for code in original[boundary_start:boundary_end]
            if code in group.index
        ]
        selected = sorted(
            boundary,
            key=lambda code: (
                -float(group.loc[code, "v2_score"]),
                original.index(code),
            ),
        )[:replacements]
        protected_set = set(protected)
        selected_set = set(selected)
        final_codes = protected + selected + [
            code
            for code in original
            if code not in protected_set and code not in selected_set
        ]
        old_top = set(original[:30])
        new_top = set(final_codes[:30])
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "codes": final_codes,
                "alpha": (
                    1.0
                    - np.arange(len(final_codes), dtype=np.float64)
                    / max(len(final_codes) - 1, 1)
                ).tolist(),
                "n_stocks": len(final_codes),
                "reranker_v2": {
                    "boundary_start": boundary_start,
                    "boundary_end": boundary_end,
                    "max_replacements": replacements,
                },
            }
        )
        audits.append(
            {
                "date": date,
                "replacements": len(new_top - old_top),
                "overlap": len(new_top & old_top),
            }
        )
    return rows, pd.DataFrame(audits)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(DATASET)
    with MODEL.open("rb") as handle:
        payload = pickle.load(handle)
    m0_rows = load_alpha_rows(M0_ALPHA)
    assert_alpha_rows_within_research(m0_rows, context="V2 reranker validation")
    rows, audit = build_rows(dataset, payload, m0_rows)
    write_alpha(OUTPUT / "alpha_raw.jsonl", rows)
    codes = {code for row in rows for code in row["codes"]}
    signal_returns = load_signal_returns(
        "data/raw", codes, pd.Timestamp(rows[0]["date"]), pd.Timestamp(rows[-1]["date"])
    )
    filtered = transform_rows(rows, signal_returns, max_signal_return=0.095)
    filtered_path = OUTPUT / "alpha_maxret095.jsonl"
    write_alpha(filtered_path, filtered)
    audit.to_csv(OUTPUT / "boundary_audit.csv", index=False)
    print(
        f"Mean replacements={audit['replacements'].mean():.2f}; "
        f"mean overlap={audit['overlap'].mean():.2f}",
        flush=True,
    )

    backtest_rows = load_backtest_rows(filtered_path)
    all_codes = sorted({code for row in backtest_rows for code in row["codes"]})
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
    for scenario_name, scenario in scenarios.items():
        for capital in (500_000.0, 1_000_000.0):
            result, returns, diagnostics = run_constrained(
                backtest_rows,
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
                    "model": "regression_v2",
                    "scenario": scenario_name,
                    "portfolio_value": capital,
                }
            )
            summary.append(result)
            tag = f"{scenario_name}_{int(capital / 10000)}w"
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
