"""Apply the frozen V4.1 date-level meta gate to an Alpha period."""

import argparse
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


V4_MODEL = ROOT / "reranker_models_20260615/gated_v4/reranker_model.pkl"
META_MODEL = ROOT / "reranker_models_20260615/meta_gate_v41/meta_gate_model.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--m0-alpha", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--forward-only", action="store_true")
    return parser.parse_args()


def meta_features(candidates, proposed, baseline, vacancies, slots):
    return {
        "vacancies": float(vacancies),
        "slots": float(slots),
        "candidate_count": float(len(candidates)),
        "proposal_overlap": len(set(proposed.index) & set(baseline.index)) / slots,
        "head_agreement": (
            len(
                set(candidates.nlargest(slots, "pred_return").index)
                & set(candidates.nlargest(slots, "pred_win").index)
            )
            / slots
        ),
        "pred_return_margin": float(
            proposed["pred_return"].mean() - baseline["pred_return"].mean()
        ),
        "pred_win_margin": float(
            proposed["pred_win"].mean() - baseline["pred_win"].mean()
        ),
        "score_margin": float(
            proposed["v4_score"].mean() - baseline["v4_score"].mean()
        ),
        "pred_return_std": float(candidates["pred_return"].std()),
        "pred_win_std": float(candidates["pred_win"].std()),
        "score_std": float(candidates["v4_score"].std()),
        "proposed_return_std": float(proposed["pred_return"].std(ddof=0)),
        "proposed_win_std": float(proposed["pred_win"].std(ddof=0)),
        "baseline_return_std": float(baseline["pred_return"].std(ddof=0)),
        "baseline_win_std": float(baseline["pred_win"].std(ddof=0)),
        **{
            column: float(candidates[column].iloc[0])
            for column in (
                "market_return_5d",
                "market_return_20d",
                "market_return_60d",
                "market_vol_20d",
                "market_vol_60d",
                "market_drawdown_60d",
                "market_ma20_gap",
                "market_ma60_gap",
            )
        },
    }


def build_rows(dataset, v4_payload, meta_payload, m0_rows):
    regression = v4_payload["regression"]
    classifier = v4_payload["classifier"]
    candidate_features = v4_payload["feature_columns"]
    meta_model = meta_payload["model"]
    meta_columns = meta_payload["feature_columns"]
    threshold = float(meta_payload["threshold"])
    candidate_end = int(v4_payload["candidate_end"])
    max_slots = int(v4_payload["max_reranked_fills"])
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
        active = False
        probability = np.nan
        proposed_codes = baseline_rerank

        eligible = [
            code
            for code in fill_candidates[protected_count:]
            if rank_map[code] < candidate_end and code in by_date[date].index
        ]
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
            candidates["pred_return"] = regression.predict(
                candidates[candidate_features]
            )
            candidates["pred_win"] = classifier.predict(
                candidates[candidate_features]
            )
            candidates["return_pct"] = daily_percentile(candidates["pred_return"])
            candidates["win_pct"] = daily_percentile(candidates["pred_win"])
            candidates["v4_score"] = (
                0.60 * candidates["return_pct"] + 0.40 * candidates["win_pct"]
            )
            proposed = candidates.nlargest(slots, "v4_score")
            baseline_codes = [
                code for code in baseline_rerank if code in candidates.index
            ]
            if len(baseline_codes) == slots:
                baseline = candidates.loc[baseline_codes]
                values = meta_features(
                    candidates, proposed, baseline, vacancies, slots
                )
                meta_frame = pd.DataFrame([values], columns=meta_columns)
                probability = float(meta_model.predict(meta_frame)[0])
                active = probability >= threshold
                proposed_codes = proposed.index.astype(str).tolist()
        selected_rerank = proposed_codes if active else baseline_rerank
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
                "reranker_v41": {
                    "active": bool(active),
                    "probability": (
                        None if not np.isfinite(probability) else probability
                    ),
                    "threshold": threshold,
                },
            }
        )
        audits.append(
            {
                "date": date,
                "active": active,
                "probability": probability,
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
    args = parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(ROOT / args.dataset)
    with V4_MODEL.open("rb") as handle:
        v4_payload = pickle.load(handle)
    with META_MODEL.open("rb") as handle:
        meta_payload = pickle.load(handle)
    m0_rows = load_alpha_rows(ROOT / args.m0_alpha)
    v41_rows, audit = build_rows(dataset, v4_payload, meta_payload, m0_rows)
    alpha_path = output / "alpha_v41.jsonl"
    write_alpha(alpha_path, v41_rows)
    audit.to_csv(output / "gate_audit.csv", index=False)
    print(
        f"Active share={audit['active'].mean():.2%}; "
        f"active dates={int(audit['active'].sum())}",
        flush=True,
    )

    row_sets = {
        "m0": load_backtest_rows(ROOT / args.m0_alpha),
        "v41": load_backtest_rows(alpha_path),
    }
    all_codes = sorted(
        {
            code
            for rows in row_sets.values()
            for row in rows
            for code in row["codes"]
        }
    )
    close, money = load_close_money(args.data_dir, all_codes, 1000.0, 1000)
    adv = recompute_adv(money, 20)
    idx_close, idx_daily = load_index_returns(args.data_dir, "hs300_index.csv", close.index)
    scenarios = (
        {"base": {"adv_cap": 0.05, "cost_mult": 1.0, "lag": 0}}
        if args.forward_only
        else {
            "base": {"adv_cap": 0.05, "cost_mult": 1.0, "lag": 0},
            "cap3": {"adv_cap": 0.03, "cost_mult": 1.0, "lag": 0},
            "cost2x": {"adv_cap": 0.05, "cost_mult": 2.0, "lag": 0},
            "lag1": {"adv_cap": 0.05, "cost_mult": 1.0, "lag": 1},
        }
    )
    summary = []
    for model_name, alpha_rows in row_sets.items():
        for scenario_name, scenario in scenarios.items():
            for capital in (500_000.0, 1_000_000.0):
                result, returns, diagnostics = run_constrained(
                    alpha_rows,
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
                returns.to_csv(output / f"returns_{tag}.csv", index=False)
                diagnostics.to_csv(output / f"diagnostics_{tag}.csv", index=False)
                print(
                    f"{tag}: ann={result['ann']:.2f}% "
                    f"sharpe={result['sharpe']:.3f} "
                    f"mdd={result['mdd'] * 100:.2f}%",
                    flush=True,
                )
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
