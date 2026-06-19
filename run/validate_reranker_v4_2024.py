"""Validate the frozen confidence-gated V4 reranker on 2024."""

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

from run.train_reranker_v4 import daily_percentile
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
MODEL = ROOT / "reranker_models_20260615/gated_v4/reranker_model.pkl"
M0_ALPHA = ROOT / "reranker_validation_20260615/m0w100/alpha_maxret095.jsonl"
OUTPUT = ROOT / "reranker_validation_20260615/gated_v4"


def build_rows(dataset, payload, m0_rows):
    regression = payload["regression"]
    classifier = payload["classifier"]
    features = payload["feature_columns"]
    gate = float(payload["gate"])
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
        baseline_rerank = baseline_fills[protected_count:]
        slots = min(vacancies, max_slots)
        eligible = [
            code
            for code in fill_candidates[protected_count:]
            if rank_map[code] < candidate_end and code in by_date[date].index
        ]
        active = False
        confidence = np.nan
        proposed = baseline_rerank
        if slots > 0 and len(eligible) >= slots:
            candidates = by_date[date].loc[eligible].copy()
            candidates["v3_vacancies"] = vacancies
            candidates["v3_rerank_slots"] = slots
            candidates["v3_was_held"] = candidates.index.isin(current_selected).astype(np.int8)
            candidates["v3_holding_age"] = [
                holding_ages.get(code, 0) for code in candidates.index
            ]
            candidates["v3_baseline_fill"] = candidates.index.isin(
                baseline_rerank
            ).astype(np.int8)
            candidates["pred_return"] = regression.predict(candidates[features])
            candidates["pred_win"] = classifier.predict(candidates[features])
            candidates["return_pct"] = daily_percentile(candidates["pred_return"])
            candidates["win_pct"] = daily_percentile(candidates["pred_win"])
            candidates["v4_score"] = (
                0.60 * candidates["return_pct"] + 0.40 * candidates["win_pct"]
            )
            proposed = candidates.nlargest(slots, "v4_score").index.astype(str).tolist()
            baseline_probability = candidates.loc[
                [code for code in baseline_rerank if code in candidates.index],
                "pred_win",
            ]
            if len(baseline_probability) == slots:
                confidence = (
                    candidates.loc[proposed, "pred_win"].mean()
                    - baseline_probability.mean()
                )
                active = confidence >= gate
        selected_rerank = proposed if active else baseline_rerank
        selected_fills = protected + selected_rerank
        selected_codes = kept + selected_fills
        if active:
            selected_fill_set = set(selected_fills)
            priority = selected_fills + [
                code for code in original if code not in selected_fill_set
            ]
        else:
            priority = original
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "codes": priority,
                "alpha": (
                    1.0 - np.arange(n, dtype=np.float64) / max(n - 1, 1)
                ).tolist(),
                "n_stocks": n,
                "reranker_v4": {
                    "active": bool(active),
                    "confidence": (
                        None if not np.isfinite(confidence) else float(confidence)
                    ),
                    "gate": gate,
                },
            }
        )
        audits.append(
            {
                "date": date,
                "vacancies": vacancies,
                "rerank_slots": slots,
                "active": active,
                "confidence": confidence,
                "changed_fills": len(set(selected_fills) - set(baseline_fills)),
            }
        )
        live_set = set(selected_codes)
        for code in list(holding_ages):
            if code not in live_set:
                holding_ages.pop(code, None)
        for code in selected_codes:
            holding_ages[code] = holding_ages.get(code, 0) + 1
        current_selected = selected_codes
    return rows, pd.DataFrame(audits)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(DATASET)
    with MODEL.open("rb") as handle:
        payload = pickle.load(handle)
    m0_rows = load_alpha_rows(M0_ALPHA)
    rows, audit = build_rows(dataset, payload, m0_rows)
    alpha_path = OUTPUT / "alpha_maxret095.jsonl"
    write_alpha(alpha_path, rows)
    audit.to_csv(OUTPUT / "gate_audit.csv", index=False)
    print(
        f"Active share={audit['active'].mean():.2%}; "
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
                    "model": "gated_v4",
                    "scenario": scenario_name,
                    "portfolio_value": capital,
                }
            )
            summary.append(result)
            print(
                f"{scenario_name}_{int(capital / 10000)}w: "
                f"ann={result['ann']:.2f}% sharpe={result['sharpe']:.3f} "
                f"mdd={result['mdd'] * 100:.2f}%",
                flush=True,
            )
    pd.DataFrame(summary).to_csv(OUTPUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
