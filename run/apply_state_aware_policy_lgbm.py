"""Apply a trained state-aware policy model to an Alpha JSONL ranking.

The transform mirrors the marginal-fill dataset construction: retained holdings
and protected fills keep priority, while a small number of replacement slots are
chosen by the policy model.  The output remains a normal Alpha JSONL file for
the official open-ledger backtest.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import load_alpha_rows, write_alpha_rows
from alpha.transforms import percentile_map
from backtest.open_ledger import normalize_ts_code
from run.train_state_aware_policy_lgbm import build_matrix


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-alpha-jsonl", required=True)
    parser.add_argument("--policy-dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-alpha-jsonl", required=True)
    parser.add_argument("--target-frac", type=float, default=0.006)
    parser.add_argument("--hold-frac", type=float, default=0.10)
    parser.add_argument("--candidate-end", type=int, default=120)
    parser.add_argument("--max-reranked-fills", type=int, default=3)
    parser.add_argument("--min-policy-score", type=float, default=None)
    parser.add_argument(
        "--min-action-score",
        type=float,
        default=None,
        help="If the best policy score is below this threshold, do not alter the original Alpha row.",
    )
    parser.add_argument(
        "--min-score-margin",
        type=float,
        default=None,
        help=(
            "Require the best policy candidate to beat the original baseline replacement "
            "slot by this score margin; otherwise keep the baseline fill."
        ),
    )
    parser.add_argument(
        "--state-gate-mode",
        choices=("none", "risk_count"),
        default="none",
        help="Only allow policy replacement when the candidate/market state is risky.",
    )
    parser.add_argument("--state-gate-min-count", type=int, default=2)
    parser.add_argument("--state-gate-global-pressure", type=float, default=0.035)
    parser.add_argument("--state-gate-industry-share", type=float, default=0.35)
    parser.add_argument("--state-gate-industry-hhi", type=float, default=0.16)
    parser.add_argument("--state-gate-beta", type=float, default=1.10)
    parser.add_argument("--state-gate-specific-vol", type=float, default=0.07)
    parser.add_argument("--state-gate-ret20", type=float, default=0.18)
    return parser.parse_args(argv)


def load_model(path):
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    for key in ("model", "feature_columns", "fill_values"):
        if key not in payload:
            raise ValueError(f"model payload is missing {key}")
    return payload


def score_dataset(dataset, payload):
    frame = pd.read_parquet(dataset)
    if frame.empty:
        raise ValueError(f"empty policy dataset: {dataset}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    x, _ = build_matrix(frame, payload["feature_columns"], payload["fill_values"])
    frame["policy_score"] = payload["model"].predict(x)
    return frame


def _finite_mean(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.mean()) if len(arr) else np.nan


def compute_state_gate(group, baseline_rerank_fills, args):
    if args.state_gate_mode == "none":
        return 1, 0, {}
    if group is None or group.empty:
        return 0, 0, {}
    by_code = group.set_index("code", drop=False)
    rows = by_code.reindex([code for code in baseline_rerank_fills if code in by_code.index])
    if rows.empty:
        rows = group[group["eligible"].eq(1)].head(max(len(baseline_rerank_fills), 1))
    global_pressure = _finite_mean(group.get("global_us_hk_pressure", pd.Series(dtype=float)).head(1))
    top_industry_share = _finite_mean(group.get("top_industry_share", pd.Series(dtype=float)).head(1))
    top_industry_hhi = _finite_mean(group.get("top_industry_hhi", pd.Series(dtype=float)).head(1))
    candidate_industry_share = _finite_mean(rows.get("candidate_industry_top_share", pd.Series(dtype=float)))
    beta = _finite_mean(rows.get("beta_60d", pd.Series(dtype=float)))
    specific_vol = _finite_mean(rows.get("specific_vol_60d", pd.Series(dtype=float)))
    ret20 = _finite_mean(rows.get("ret_20d", pd.Series(dtype=float)))
    flags = {
        "global_pressure": int(np.isfinite(global_pressure) and global_pressure >= args.state_gate_global_pressure),
        "top_industry_share": int(np.isfinite(top_industry_share) and top_industry_share >= args.state_gate_industry_share),
        "top_industry_hhi": int(np.isfinite(top_industry_hhi) and top_industry_hhi >= args.state_gate_industry_hhi),
        "candidate_industry_share": int(
            np.isfinite(candidate_industry_share)
            and candidate_industry_share >= args.state_gate_industry_share
        ),
        "beta": int(np.isfinite(beta) and beta >= args.state_gate_beta),
        "specific_vol": int(np.isfinite(specific_vol) and specific_vol >= args.state_gate_specific_vol),
        "ret20": int(np.isfinite(ret20) and ret20 >= args.state_gate_ret20),
    }
    count = int(sum(flags.values()))
    metrics = {
        "state_gate_count": count,
        "state_gate_global_pressure": global_pressure,
        "state_gate_top_industry_share": top_industry_share,
        "state_gate_top_industry_hhi": top_industry_hhi,
        "state_gate_candidate_industry_share": candidate_industry_share,
        "state_gate_beta": beta,
        "state_gate_specific_vol": specific_vol,
        "state_gate_ret20": ret20,
    }
    metrics.update({f"state_flag_{key}": value for key, value in flags.items()})
    return int(count >= int(args.state_gate_min_count)), count, metrics


def reorder_one_day(row, group, current_selected, holding_ages, args):
    date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    original = [normalize_ts_code(code) for code in row.get("codes", [])]
    if not original:
        return row, current_selected, holding_ages, {"date": date, "changed": 0}
    n = len(original)
    target_n = max(1, int(n * float(args.target_frac)))
    hold_n = max(target_n, int(n * float(args.hold_frac)))
    rank_map = {code: rank for rank, code in enumerate(original)}
    kept = [code for code in current_selected if rank_map.get(code, n + 1) < hold_n]
    if len(kept) > target_n:
        kept = sorted(kept, key=lambda code: rank_map.get(code, n + 1))[:target_n]
    vacancies = max(target_n - len(kept), 0)
    fill_candidates = [code for code in original if code not in set(kept)]
    baseline_fills = fill_candidates[:vacancies]
    protected_count = max(vacancies - int(args.max_reranked_fills), 0)
    protected_fills = baseline_fills[:protected_count]
    slots = min(vacancies, int(args.max_reranked_fills))
    baseline_rerank_fills = baseline_fills[protected_count : protected_count + slots]
    state_gate_pass, state_gate_count, state_gate_metrics = compute_state_gate(
        group,
        baseline_rerank_fills,
        args,
    )
    chosen = []
    candidate_score = {}
    baseline_score = np.nan
    best_score = np.nan
    score_margin = np.nan
    margin_pass = 1
    action_score_pass = 1
    if slots > 0 and group is not None and not group.empty:
        eligible = group[group["eligible"].eq(1)].copy()
        eligible = eligible[eligible["code"].isin(original[: int(args.candidate_end)])]
        if args.min_policy_score is not None:
            eligible = eligible[eligible["policy_score"] >= float(args.min_policy_score)]
        candidate_score = {
            normalize_ts_code(code): float(score)
            for code, score in zip(eligible["code"], eligible["policy_score"])
            if np.isfinite(score)
        }
        baseline_scores = [
            candidate_score[code]
            for code in baseline_rerank_fills
            if code in candidate_score and np.isfinite(candidate_score[code])
        ]
        baseline_score = float(np.mean(baseline_scores)) if baseline_scores else np.nan
        eligible = eligible.sort_values(["policy_score", "candidate_position"], ascending=[False, True])
        if not eligible.empty:
            best_score = float(eligible["policy_score"].iloc[0])
        if args.min_action_score is not None and np.isfinite(best_score):
            action_score_pass = int(best_score >= float(args.min_action_score))
        elif args.min_action_score is not None:
            action_score_pass = 0
        if np.isfinite(best_score) and np.isfinite(baseline_score):
            score_margin = best_score - baseline_score
        if args.min_score_margin is not None and np.isfinite(score_margin):
            margin_pass = int(float(score_margin) >= float(args.min_score_margin))
        elif args.min_score_margin is not None:
            margin_pass = 0
        if margin_pass:
            iterator = eligible["code"].tolist()
        else:
            iterator = []
        for code in iterator:
            code = normalize_ts_code(code)
            if code not in chosen and code not in kept and code not in protected_fills:
                chosen.append(code)
            if len(chosen) >= slots:
                break
    if len(chosen) < slots:
        for code in baseline_fills[protected_count:]:
            if code not in chosen and code not in kept and code not in protected_fills:
                chosen.append(code)
            if len(chosen) >= slots:
                break
    policy_applied = int(
        not (args.min_score_margin is not None and not margin_pass)
        and bool(action_score_pass)
        and bool(state_gate_pass)
    )
    if not policy_applied:
        output = dict(row)
        output["date"] = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        selected = (kept + baseline_fills)[:target_n]
        selected_set = set(selected)
        next_ages = {code: age for code, age in holding_ages.items() if code in selected_set}
        for code in selected:
            next_ages[code] = next_ages.get(code, 0) + 1
        audit = {
            "date": date,
            "target_n": target_n,
            "kept_n": len(kept),
            "vacancies": vacancies,
            "protected_fills": len(protected_fills),
            "policy_chosen": 0,
            "changed": 0,
            "chosen_codes": "",
            "baseline_rerank_codes": ",".join(baseline_rerank_fills),
            "best_policy_score": best_score,
            "baseline_policy_score": baseline_score,
            "score_margin": score_margin,
            "margin_pass": margin_pass,
            "action_score_pass": action_score_pass,
            "policy_applied": policy_applied,
        }
        audit.update(state_gate_metrics)
        return output, selected, next_ages, audit
    priority = []
    for bucket in (kept, protected_fills, chosen):
        for code in bucket:
            if code not in priority:
                priority.append(code)
    for code in original:
        if code not in priority:
            priority.append(code)
    output = dict(row)
    output["date"] = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    output["codes"] = priority
    output["alpha"] = percentile_map(priority)

    selected = kept + protected_fills + chosen
    selected = selected[:target_n]
    selected_set = set(selected)
    next_ages = {code: age for code, age in holding_ages.items() if code in selected_set}
    for code in selected:
        next_ages[code] = next_ages.get(code, 0) + 1
    changed = sum(1 for a, b in zip(original[: len(priority)], priority) if a != b)
    audit = {
        "date": date,
        "target_n": target_n,
        "kept_n": len(kept),
        "vacancies": vacancies,
        "protected_fills": len(protected_fills),
        "policy_chosen": len(chosen),
        "changed": changed,
        "chosen_codes": ",".join(chosen),
        "baseline_rerank_codes": ",".join(baseline_rerank_fills),
        "best_policy_score": best_score,
        "baseline_policy_score": baseline_score,
        "score_margin": score_margin,
        "margin_pass": margin_pass,
        "action_score_pass": action_score_pass,
        "policy_applied": policy_applied,
    }
    audit.update(state_gate_metrics)
    return output, selected, next_ages, audit


def main(argv=None):
    args = parse_args(argv)
    payload = load_model(args.model)
    scored = score_dataset(args.policy_dataset, payload)
    by_date = {date: group for date, group in scored.groupby("date", sort=False)}
    rows = load_alpha_rows(args.input_alpha_jsonl)
    output_rows = []
    audits = []
    current_selected = []
    holding_ages = {}
    for row in rows:
        date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        group = by_date.get(date)
        output, current_selected, holding_ages, audit = reorder_one_day(
            row,
            group,
            current_selected,
            holding_ages,
            args,
        )
        output_rows.append(output)
        audits.append(audit)
    output_path = write_alpha_rows(args.output_alpha_jsonl, output_rows)
    audit_path = Path(args.output_alpha_jsonl).with_suffix(".policy_audit.csv")
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    print(
        {
            "output_alpha_jsonl": str(output_path),
            "audit_csv": str(audit_path),
            "dates": len(output_rows),
            "changed_days": int(sum(1 for item in audits if item["changed"] > 0)),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
