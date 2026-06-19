"""Validate conservative Top30 boundary corrections using the v1 reranker."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from run.validate_reranker_2024 import (
    ROOT,
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
from core.research_protocol import assert_alpha_rows_within_research


DATASET = ROOT / "reranker_data_20260614/m0_validation_2024/reranker_dataset.parquet"
MODEL = ROOT / "reranker_models_20260615/lambdarank_v1/reranker_model.pkl"
M0_ALPHA = ROOT / "multi_loss_validation_20260614/portfolio/m0_nomulti_e6/alpha_raw.jsonl"
OUTPUT = ROOT / "reranker_validation_20260615/conservative_v1"
CONFIGS = (
    ("k3_m0w075", 3, 0.75),
    ("k3_m0w050", 3, 0.50),
    ("k6_m0w075", 6, 0.75),
    ("k6_m0w050", 6, 0.50),
)
BOUNDARY_END = 80


def descending_percentile(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = 1.0 - np.arange(len(values)) / max(len(values) - 1, 1)
    return result


def build_rows(dataset, model_payload, m0_rows, max_replacements, m0_weight):
    model = model_payload["model"]
    features = model_payload["feature_columns"]
    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    frame["reranker_score"] = model.predict(frame[features])
    frame["reranker_pct"] = frame.groupby("date")["reranker_score"].transform(
        descending_percentile
    )
    frame["m0_pct"] = 1.0 - frame["m0_rank_pct"].astype(float)
    frame["blend_score"] = (
        m0_weight * frame["m0_pct"] + (1.0 - m0_weight) * frame["reranker_pct"]
    )
    by_date = {
        pd.Timestamp(date): group.set_index("code")
        for date, group in frame.groupby("date", sort=False)
    }

    output = []
    audit = []
    protected_count = 30 - max_replacements
    for row in m0_rows:
        date = pd.Timestamp(row["date"])
        original = [str(code) for code in row["codes"]]
        group = by_date[date]
        protected = original[:protected_count]
        boundary = [
            code
            for code in original[protected_count:BOUNDARY_END]
            if code in group.index
        ]
        ranked_boundary = sorted(
            boundary,
            key=lambda code: (
                -float(group.loc[code, "blend_score"]),
                original.index(code),
            ),
        )
        selected = ranked_boundary[:max_replacements]
        selected_set = set(selected)
        protected_set = set(protected)
        final_codes = (
            protected
            + selected
            + [
                code
                for code in original
                if code not in protected_set and code not in selected_set
            ]
        )
        old_top = set(original[:30])
        new_top = set(final_codes[:30])
        output.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "codes": final_codes,
                "alpha": (
                    1.0
                    - np.arange(len(final_codes), dtype=np.float64)
                    / max(len(final_codes) - 1, 1)
                ).tolist(),
                "n_stocks": len(final_codes),
                "conservative_reranker": {
                    "max_replacements": max_replacements,
                    "m0_weight": m0_weight,
                    "boundary_end": BOUNDARY_END,
                },
            }
        )
        audit.append(
            {
                "date": date,
                "replacements": len(new_top - old_top),
                "overlap": len(new_top & old_top),
            }
        )
    return output, pd.DataFrame(audit)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(DATASET)
    with MODEL.open("rb") as handle:
        model_payload = pickle.load(handle)
    m0_rows = load_alpha_rows(M0_ALPHA)
    assert_alpha_rows_within_research(m0_rows, context="conservative reranker validation")

    alpha_sets = {}
    audits = []
    for name, max_replacements, m0_weight in CONFIGS:
        rows, audit = build_rows(
            dataset, model_payload, m0_rows, max_replacements, m0_weight
        )
        raw_path = OUTPUT / name / "alpha_raw.jsonl"
        filtered_path = OUTPUT / name / "alpha_maxret095.jsonl"
        write_alpha(raw_path, rows)
        codes = {code for row in rows for code in row["codes"]}
        signal_returns = load_signal_returns(
            "data/raw",
            codes,
            pd.Timestamp(rows[0]["date"]),
            pd.Timestamp(rows[-1]["date"]),
        )
        filtered = transform_rows(rows, signal_returns, max_signal_return=0.095)
        write_alpha(filtered_path, filtered)
        alpha_sets[name] = load_backtest_rows(filtered_path)
        audit["model"] = name
        audits.append(audit)
        print(
            f"{name}: mean replacements={audit['replacements'].mean():.2f}",
            flush=True,
        )

    pd.concat(audits, ignore_index=True).to_csv(
        OUTPUT / "boundary_audit.csv", index=False
    )
    all_codes = sorted(
        {
            code
            for rows in alpha_sets.values()
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
    for name, rows in alpha_sets.items():
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
                        "model": name,
                        "scenario": scenario_name,
                        "portfolio_value": capital,
                    }
                )
                summary.append(result)
                tag = f"{name}_{scenario_name}_{int(capital / 10000)}w"
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
