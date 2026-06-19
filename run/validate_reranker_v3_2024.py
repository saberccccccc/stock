"""Validate the frozen state-aware V3 fill reranker on 2024."""

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
    make_run_args,
    recompute_adv,
    run_constrained,
    write_alpha,
)


DATASET = ROOT / "reranker_v3_data_20260615/m0_validation_2024/reranker_v3_dataset.parquet"
MODEL = ROOT / "reranker_models_20260615/regression_v3/reranker_model.pkl"
M0_ALPHA = ROOT / "reranker_validation_20260615/m0w100/alpha_maxret095.jsonl"
OUTPUT = ROOT / "reranker_validation_20260615/regression_v3"


def build_rows(dataset, payload, m0_rows):
    model = payload["model"]
    features = payload["feature_columns"]
    candidate_end = int(payload["candidate_end"])
    max_slots = int(payload["max_reranked_fills"])
    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    by_date = {
        date: group.set_index("code")
        for date, group in frame.groupby("date", sort=False)
    }
    current_selected = []
    holding_ages = {}
    rows = []
    audits = []
    for row in m0_rows:
        date = pd.Timestamp(row["date"])
        original = [str(code) for code in row["codes"]]
        n = len(original)
        target_n = max(1, int(n * 0.006))
        hold_n = max(target_n, int(n * 0.10))
        rank_map = {code: index for index, code in enumerate(original)}
        kept = [
            code
            for code in current_selected
            if rank_map.get(code, n + 1) < hold_n
        ]
        if len(kept) > target_n:
            kept = sorted(kept, key=lambda code: rank_map[code])[:target_n]
        vacancies = max(target_n - len(kept), 0)
        fill_candidates = [code for code in original if code not in set(kept)]
        baseline_fills = fill_candidates[:vacancies]
        protected_count = max(vacancies - max_slots, 0)
        protected = baseline_fills[:protected_count]
        slots = min(vacancies, max_slots)
        eligible = [
            code
            for code in fill_candidates[protected_count:]
            if rank_map[code] < candidate_end and code in by_date[date].index
        ]

        selected_by_model = []
        if slots > 0 and eligible:
            candidates = by_date[date].loc[eligible].copy()
            candidates["v3_vacancies"] = vacancies
            candidates["v3_rerank_slots"] = slots
            candidates["v3_was_held"] = candidates.index.isin(current_selected).astype(np.int8)
            candidates["v3_holding_age"] = [
                holding_ages.get(code, 0) for code in candidates.index
            ]
            scores = model.predict(candidates[features])
            selected_by_model = [
                code
                for code, _ in sorted(
                    zip(candidates.index.astype(str), scores),
                    key=lambda item: (-item[1], rank_map[item[0]]),
                )[:slots]
            ]
        selected_fills = protected + selected_by_model
        if len(selected_fills) < vacancies:
            selected_set = set(selected_fills)
            selected_fills.extend(
                code
                for code in fill_candidates
                if code not in selected_set
            )
            selected_fills = selected_fills[:vacancies]

        priority_codes = kept + selected_fills
        priority_set = set(priority_codes)
        priority = priority_codes + [
            code for code in original if code not in priority_set
        ]
        alpha = (
            1.0
            - np.arange(n, dtype=np.float64) / max(n - 1, 1)
        ).tolist()
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "codes": priority,
                "alpha": alpha,
                "n_stocks": n,
                "reranker_v3": {
                    "vacancies": vacancies,
                    "rerank_slots": slots,
                },
            }
        )
        baseline_set = set(baseline_fills)
        selected_set = set(selected_fills)
        labelled = by_date[date]
        has_targets = "exec_target_raw" in labelled.columns
        selected_targets = (
            [
                float(labelled.loc[code, "exec_target_raw"])
                for code in selected_fills
                if code in labelled.index
                and pd.notna(labelled.loc[code, "exec_target_raw"])
            ]
            if has_targets
            else []
        )
        baseline_targets = (
            [
                float(labelled.loc[code, "exec_target_raw"])
                for code in baseline_fills
                if code in labelled.index
                and pd.notna(labelled.loc[code, "exec_target_raw"])
            ]
            if has_targets
            else []
        )
        audits.append(
            {
                "date": date,
                "vacancies": vacancies,
                "rerank_slots": slots,
                "changed_fills": len(selected_set - baseline_set),
                "selected_target": (
                    float(np.mean(selected_targets)) if selected_targets else np.nan
                ),
                "baseline_target": (
                    float(np.mean(baseline_targets)) if baseline_targets else np.nan
                ),
                "target_delta": (
                    float(np.mean(selected_targets) - np.mean(baseline_targets))
                    if selected_targets and baseline_targets
                    else np.nan
                ),
            }
        )
        selected = kept + selected_fills
        live_set = set(selected)
        for code in list(holding_ages):
            if code not in live_set:
                holding_ages.pop(code, None)
        for code in selected:
            holding_ages[code] = holding_ages.get(code, 0) + 1
        current_selected = selected
    return rows, pd.DataFrame(audits)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(DATASET)
    with MODEL.open("rb") as handle:
        payload = pickle.load(handle)
    m0_rows = load_alpha_rows(M0_ALPHA)
    assert_alpha_rows_within_research(m0_rows, context="V3 reranker validation")
    rows, audit = build_rows(dataset, payload, m0_rows)
    alpha_path = OUTPUT / "alpha_maxret095.jsonl"
    write_alpha(alpha_path, rows)
    audit.to_csv(OUTPUT / "fill_audit.csv", index=False)
    print(
        f"Mean vacancies={audit['vacancies'].mean():.2f}; "
        f"mean changed fills={audit['changed_fills'].mean():.2f}",
        flush=True,
    )

    backtest_rows = load_backtest_rows(alpha_path)
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
                    "model": "regression_v3",
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
