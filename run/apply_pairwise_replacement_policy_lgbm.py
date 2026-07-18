"""Apply a pairwise candidate-vs-baseline replacement policy to Alpha JSONL."""

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
from run.train_pairwise_replacement_policy_lgbm import build_matrix


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-alpha-jsonl", required=True)
    parser.add_argument("--pairwise-dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-alpha-jsonl", required=True)
    parser.add_argument("--target-frac", type=float, default=0.006)
    parser.add_argument("--hold-frac", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument(
        "--max-position-delta",
        type=float,
        default=None,
        help="Only allow replacing the baseline fill with a candidate within this many rank positions.",
    )
    parser.add_argument(
        "--rewrite-mode",
        choices=("promote", "slot_swap"),
        default="promote",
        help="promote puts chosen after kept holdings; slot_swap only replaces the baseline fill slot.",
    )
    parser.add_argument(
        "--industry-hhi-penalty",
        type=float,
        default=0.0,
        help="Subtract this times positive diff_top_industry_hhi from the pairwise score.",
    )
    parser.add_argument(
        "--top-industry-share-penalty",
        type=float,
        default=0.0,
        help="Subtract this times positive diff_top_industry_share from the pairwise score.",
    )
    parser.add_argument(
        "--candidate-industry-share-penalty",
        type=float,
        default=0.0,
        help="Subtract this times positive diff_candidate_industry_top_share from the pairwise score.",
    )
    parser.add_argument(
        "--concentration-penalty-condition",
        choices=(
            "always",
            "fragile_beta",
            "fragile_vol",
            "fragile_beta_or_vol",
            "pair_risk_positive",
            "fragile_or_pair_risk",
            "active_drawdown_negative",
            "global_pressure_high",
            "active_or_global",
            "active_and_global",
        ),
        default="always",
        help="State condition for applying concentration penalties.",
    )
    parser.add_argument("--penalty-beta-threshold", type=float, default=1.2)
    parser.add_argument("--penalty-specific-vol-threshold", type=float, default=0.08)
    parser.add_argument("--penalty-pair-risk-threshold", type=float, default=0.0)
    parser.add_argument("--penalty-active-drawdown-threshold", type=float, default=0.0)
    parser.add_argument("--penalty-global-pressure-threshold", type=float, default=0.04)
    parser.add_argument(
        "--specific-vol-worsen-penalty",
        type=float,
        default=0.0,
        help="Subtract this times positive diff_specific_vol_60d from the pairwise score.",
    )
    parser.add_argument(
        "--ret20-worsen-penalty",
        type=float,
        default=0.0,
        help="Subtract this times max(-diff_ret_20d, 0) from the pairwise score.",
    )
    parser.add_argument(
        "--pair-risk-worsen-penalty",
        type=float,
        default=0.0,
        help="Subtract this times positive pair_risk_delta from the pairwise score.",
    )
    parser.add_argument(
        "--beta-worsen-penalty",
        type=float,
        default=0.0,
        help="Subtract this times positive diff_beta_60d from the pairwise score.",
    )
    parser.add_argument(
        "--risk-guard-condition",
        choices=(
            "always",
            "when_decrowding",
            "when_pair_risk_positive",
            "when_decrowding_or_pair_risk",
            "active_drawdown_negative",
            "global_pressure_high",
            "active_or_global",
            "active_and_global",
            "risk_combo_ge",
            "fragile_and_risk_combo_ge",
            "crowd_and_svol",
            "pair_risk_and_svol",
        ),
        default="always",
        help="State condition for specific-vol and ret20 risk guards.",
    )
    parser.add_argument(
        "--risk-combo-min-count",
        type=int,
        default=2,
        help="For risk_combo_ge conditions, minimum number of worsening risk components.",
    )
    parser.add_argument(
        "--gate-max-pair-risk-delta",
        type=float,
        default=None,
        help="Reject a replacement when pair_risk_delta is above this value.",
    )
    parser.add_argument(
        "--gate-max-pair-downside-delta",
        type=float,
        default=None,
        help="Reject a replacement when pair_downside_delta is above this value.",
    )
    parser.add_argument(
        "--gate-max-diff-specific-vol-60d",
        type=float,
        default=None,
        help="Reject a replacement when diff_specific_vol_60d is above this value.",
    )
    parser.add_argument(
        "--gate-min-diff-ret20",
        type=float,
        default=None,
        help="Reject a replacement when diff_ret_20d is below this value.",
    )
    parser.add_argument(
        "--gate-condition",
        choices=(
            "always",
            "active_drawdown_negative",
            "global_pressure_high",
            "active_or_global",
            "active_and_global",
        ),
        default="always",
        help="State condition for hard replacement gates.",
    )
    return parser.parse_args(argv)


def load_model(path):
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    return payload


def score_pairwise(path, payload):
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    x, _ = build_matrix(frame, payload["feature_columns"], payload["fill_values"])
    frame["policy_score"] = payload["model"].predict(x)
    return frame


def apply_concentration_penalty(frame, args):
    frame = frame.copy()
    penalty = np.zeros(len(frame), dtype=np.float64)
    if float(args.industry_hhi_penalty) > 0 and "diff_top_industry_hhi" in frame.columns:
        hhi = pd.to_numeric(frame["diff_top_industry_hhi"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        penalty += float(args.industry_hhi_penalty) * np.maximum(hhi, 0.0)
    if float(args.top_industry_share_penalty) > 0 and "diff_top_industry_share" in frame.columns:
        share = (
            pd.to_numeric(frame["diff_top_industry_share"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        penalty += float(args.top_industry_share_penalty) * np.maximum(share, 0.0)
    if (
        float(args.candidate_industry_share_penalty) > 0
        and "diff_candidate_industry_top_share" in frame.columns
    ):
        candidate_share = (
            pd.to_numeric(frame["diff_candidate_industry_top_share"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        penalty += float(args.candidate_industry_share_penalty) * np.maximum(candidate_share, 0.0)
    condition = concentration_penalty_condition(frame, args)
    penalty = penalty * condition.astype(np.float64)
    guard_penalty, guard_active = risk_guard_penalty(frame, args)
    total_penalty = penalty + guard_penalty
    frame["raw_policy_score"] = frame["policy_score"]
    frame["concentration_penalty"] = penalty
    frame["concentration_penalty_active"] = condition.astype(int)
    frame["risk_guard_penalty"] = guard_penalty
    frame["risk_guard_active"] = guard_active.astype(int)
    frame["policy_score"] = frame["policy_score"] - total_penalty
    frame = apply_replacement_gate(frame, args)
    return frame


def apply_replacement_gate(frame, args):
    frame = frame.copy()
    gate_active = replacement_gate_condition(frame, args)
    rejected = pd.Series(False, index=frame.index)
    reasons = pd.Series("", index=frame.index, dtype=object)

    def add_rejection(mask, reason):
        nonlocal rejected, reasons
        mask = mask.fillna(False) & gate_active
        rejected = rejected | mask
        reasons.loc[mask & reasons.eq("")] = reason
        reasons.loc[mask & ~reasons.eq(reason) & ~reasons.eq("")] = reasons.loc[
            mask & ~reasons.eq(reason) & ~reasons.eq("")
        ] + ";" + reason

    if getattr(args, "gate_max_pair_risk_delta", None) is not None:
        values = numeric_column(frame, "pair_risk_delta")
        add_rejection(values > float(args.gate_max_pair_risk_delta), "pair_risk_delta")
    if getattr(args, "gate_max_pair_downside_delta", None) is not None:
        values = numeric_column(frame, "pair_downside_delta")
        add_rejection(values > float(args.gate_max_pair_downside_delta), "pair_downside_delta")
    if getattr(args, "gate_max_diff_specific_vol_60d", None) is not None:
        values = numeric_column(frame, "diff_specific_vol_60d")
        add_rejection(values > float(args.gate_max_diff_specific_vol_60d), "diff_specific_vol_60d")
    if getattr(args, "gate_min_diff_ret20", None) is not None:
        values = numeric_column(frame, "diff_ret_20d")
        add_rejection(values < float(args.gate_min_diff_ret20), "diff_ret_20d")

    frame["replacement_gate_active"] = gate_active.astype(int)
    frame["replacement_gate_rejected"] = rejected.astype(int)
    frame["replacement_gate_reason"] = reasons
    frame.loc[rejected, "policy_score"] = -np.inf
    return frame


def numeric_column(frame, column):
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def replacement_gate_condition(frame, args):
    mode = getattr(args, "gate_condition", "always")
    if mode == "always":
        return pd.Series(True, index=frame.index)
    state = state_condition(frame, args)
    if mode == "active_drawdown_negative":
        return state["active"]
    if mode == "global_pressure_high":
        return state["global"]
    if mode == "active_or_global":
        return state["active"] | state["global"]
    if mode == "active_and_global":
        return state["active"] & state["global"]
    raise ValueError(f"unknown replacement gate condition: {mode}")


def concentration_penalty_condition(frame, args):
    mode = getattr(args, "concentration_penalty_condition", "always")
    if mode == "always":
        return pd.Series(True, index=frame.index)
    state = state_condition(frame, args)
    beta = pd.to_numeric(frame.get("diag_portfolio_beta_60d"), errors="coerce").fillna(-np.inf)
    vol = pd.to_numeric(frame.get("diag_portfolio_specific_vol_60d"), errors="coerce").fillna(-np.inf)
    pair_risk = pd.to_numeric(frame.get("pair_risk_delta"), errors="coerce").fillna(-np.inf)
    fragile_beta = beta >= float(getattr(args, "penalty_beta_threshold", 1.2))
    fragile_vol = vol >= float(getattr(args, "penalty_specific_vol_threshold", 0.08))
    risk_positive = pair_risk >= float(getattr(args, "penalty_pair_risk_threshold", 0.0))
    if mode == "fragile_beta":
        return fragile_beta
    if mode == "fragile_vol":
        return fragile_vol
    if mode == "fragile_beta_or_vol":
        return fragile_beta | fragile_vol
    if mode == "pair_risk_positive":
        return risk_positive
    if mode == "fragile_or_pair_risk":
        return fragile_beta | fragile_vol | risk_positive
    if mode == "active_drawdown_negative":
        return state["active"]
    if mode == "global_pressure_high":
        return state["global"]
    if mode == "active_or_global":
        return state["active"] | state["global"]
    if mode == "active_and_global":
        return state["active"] & state["global"]
    raise ValueError(f"unknown concentration penalty condition: {mode}")


def state_condition(frame, args):
    if "diag_active_drawdown_trailing_return" in frame.columns:
        active_values = pd.to_numeric(
            frame["diag_active_drawdown_trailing_return"],
            errors="coerce",
        ).fillna(np.inf)
    else:
        active_values = pd.Series(np.inf, index=frame.index)
    if "diag_global_risk_pressure" in frame.columns:
        global_values = pd.to_numeric(
            frame["diag_global_risk_pressure"],
            errors="coerce",
        ).fillna(-np.inf)
    else:
        global_values = pd.Series(-np.inf, index=frame.index)
    active = active_values <= float(getattr(args, "penalty_active_drawdown_threshold", 0.0))
    global_pressure = global_values >= float(getattr(args, "penalty_global_pressure_threshold", 0.04))
    return {"active": active, "global": global_pressure}


def risk_guard_penalty(frame, args):
    vol_weight = float(getattr(args, "specific_vol_worsen_penalty", 0.0))
    ret_weight = float(getattr(args, "ret20_worsen_penalty", 0.0))
    pair_risk_weight = float(getattr(args, "pair_risk_worsen_penalty", 0.0))
    beta_weight = float(getattr(args, "beta_worsen_penalty", 0.0))
    if vol_weight <= 0.0 and ret_weight <= 0.0 and pair_risk_weight <= 0.0 and beta_weight <= 0.0:
        inactive = pd.Series(False, index=frame.index)
        return np.zeros(len(frame), dtype=np.float64), inactive
    mode = getattr(args, "risk_guard_condition", "always")
    state = state_condition(frame, args)
    decrowding = numeric_column(frame, "diff_candidate_industry_top_share").fillna(0.0) < 0.0
    pair_risk = numeric_column(frame, "pair_risk_delta").fillna(-np.inf) >= float(
        getattr(args, "penalty_pair_risk_threshold", 0.0)
    )
    if mode == "always":
        active = pd.Series(True, index=frame.index)
    elif mode == "when_decrowding":
        active = decrowding
    elif mode == "when_pair_risk_positive":
        active = pair_risk
    elif mode == "when_decrowding_or_pair_risk":
        active = decrowding | pair_risk
    elif mode == "active_drawdown_negative":
        active = state["active"]
    elif mode == "global_pressure_high":
        active = state["global"]
    elif mode == "active_or_global":
        active = state["active"] | state["global"]
    elif mode == "active_and_global":
        active = state["active"] & state["global"]
    elif mode == "risk_combo_ge":
        active = risk_combo_condition(frame, args)
    elif mode == "fragile_and_risk_combo_ge":
        active = (state["active"] | state["global"]) & risk_combo_condition(frame, args)
    elif mode == "crowd_and_svol":
        active = (
            numeric_column(frame, "diff_candidate_industry_top_share").fillna(0.0).gt(0.0)
            & numeric_column(frame, "diff_specific_vol_60d").fillna(0.0).gt(0.0)
        )
    elif mode == "pair_risk_and_svol":
        active = (
            numeric_column(frame, "pair_risk_delta").fillna(0.0).gt(float(getattr(args, "penalty_pair_risk_threshold", 0.0)))
            & numeric_column(frame, "diff_specific_vol_60d").fillna(0.0).gt(0.0)
        )
    else:
        raise ValueError(f"unknown risk guard condition: {mode}")
    diff_vol = numeric_column(frame, "diff_specific_vol_60d").fillna(0.0).to_numpy(dtype=np.float64)
    diff_ret20 = numeric_column(frame, "diff_ret_20d").fillna(0.0).to_numpy(dtype=np.float64)
    pair_risk_delta = numeric_column(frame, "pair_risk_delta").fillna(0.0).to_numpy(dtype=np.float64)
    diff_beta = numeric_column(frame, "diff_beta_60d").fillna(0.0).to_numpy(dtype=np.float64)
    penalty = (
        vol_weight * np.maximum(diff_vol, 0.0)
        + ret_weight * np.maximum(-diff_ret20, 0.0)
        + pair_risk_weight * np.maximum(pair_risk_delta, 0.0)
        + beta_weight * np.maximum(diff_beta, 0.0)
    )
    penalty = penalty * active.astype(np.float64).to_numpy()
    return penalty, active


def risk_combo_condition(frame, args):
    min_count = int(getattr(args, "risk_combo_min_count", 2))
    components = [
        numeric_column(frame, "pair_risk_delta").fillna(0.0).gt(float(getattr(args, "penalty_pair_risk_threshold", 0.0))),
        numeric_column(frame, "diff_specific_vol_60d").fillna(0.0).gt(0.0),
        numeric_column(frame, "diff_ret_20d").fillna(0.0).lt(0.0),
        numeric_column(frame, "diff_candidate_industry_top_share").fillna(0.0).gt(0.0),
        numeric_column(frame, "diff_beta_60d").fillna(0.0).gt(0.0),
    ]
    count = sum(component.astype(int) for component in components)
    return count >= min_count


def reorder_day(row, group, current_selected, holding_ages, args):
    date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    original = [normalize_ts_code(code) for code in row.get("codes", [])]
    if not original:
        return row, current_selected, holding_ages, {"date": date, "policy_applied": 0}
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
    baseline_code = baseline_fills[0] if baseline_fills else None
    chosen_code = None
    best_score = np.nan
    baseline_pos = rank_map.get(baseline_code, np.nan) if baseline_code else np.nan
    chosen_original_pos = np.nan
    if baseline_code and group is not None and not group.empty:
        candidates = group[group["baseline_code"].map(normalize_ts_code).eq(baseline_code)].copy()
        if candidates.empty:
            candidates = group.copy()
        candidates["code"] = candidates["code"].map(normalize_ts_code)
        candidates = candidates[candidates["code"].isin(original)]
        if args.max_position_delta is not None:
            candidates = candidates[
                candidates["candidate_position"].astype(float)
                <= float(baseline_pos) + float(args.max_position_delta)
            ]
        if not candidates.empty:
            best = candidates.sort_values(["policy_score", "candidate_position"], ascending=[False, True]).iloc[0]
            best_score = float(best["policy_score"])
            raw_best_score = float(best.get("raw_policy_score", best_score))
            concentration_penalty = float(best.get("concentration_penalty", 0.0))
            concentration_penalty_active = int(best.get("concentration_penalty_active", 0))
            risk_guard_penalty_value = float(best.get("risk_guard_penalty", 0.0))
            risk_guard_active = int(best.get("risk_guard_active", 0))
            replacement_gate_active = int(best.get("replacement_gate_active", 0))
            replacement_gate_rejected = int(best.get("replacement_gate_rejected", 0))
            replacement_gate_reason = str(best.get("replacement_gate_reason", ""))
            if np.isfinite(best_score) and best_score >= float(args.threshold):
                chosen_code = normalize_ts_code(best["code"])
                chosen_original_pos = rank_map.get(chosen_code, np.nan)
        else:
            raw_best_score = np.nan
            concentration_penalty = np.nan
            concentration_penalty_active = 0
            risk_guard_penalty_value = np.nan
            risk_guard_active = 0
            replacement_gate_active = 0
            replacement_gate_rejected = 0
            replacement_gate_reason = ""
    else:
        raw_best_score = np.nan
        concentration_penalty = np.nan
        concentration_penalty_active = 0
        risk_guard_penalty_value = np.nan
        risk_guard_active = 0
        replacement_gate_active = 0
        replacement_gate_rejected = 0
        replacement_gate_reason = ""
    if not chosen_code or chosen_code == baseline_code:
        selected = (kept + baseline_fills)[:target_n]
        next_ages = {code: age for code, age in holding_ages.items() if code in set(selected)}
        for code in selected:
            next_ages[code] = next_ages.get(code, 0) + 1
        output = dict(row)
        output["date"] = date
        return output, selected, next_ages, {
            "date": date,
            "policy_applied": 0,
            "best_score": best_score,
            "raw_best_score": raw_best_score,
            "concentration_penalty": concentration_penalty,
            "concentration_penalty_active": concentration_penalty_active,
            "risk_guard_penalty": risk_guard_penalty_value,
            "risk_guard_active": risk_guard_active,
            "replacement_gate_active": replacement_gate_active,
            "replacement_gate_rejected": replacement_gate_rejected,
            "replacement_gate_reason": replacement_gate_reason,
            "threshold": float(args.threshold),
            "baseline_code": baseline_code or "",
            "baseline_position": baseline_pos,
            "chosen_code": "",
            "chosen_original_position": chosen_original_pos,
            "position_delta": np.nan,
            "vacancies": vacancies,
            "target_n": target_n,
            "kept_n": len(kept),
            "rewrite_mode": args.rewrite_mode,
        }
    if args.rewrite_mode == "slot_swap":
        priority = []
        inserted = False
        for code in original:
            if code == chosen_code:
                continue
            if code == baseline_code:
                if chosen_code not in priority:
                    priority.append(chosen_code)
                inserted = True
            elif code not in priority:
                priority.append(code)
        if not inserted and chosen_code not in priority:
            priority.insert(rank_map.get(baseline_code, len(priority)), chosen_code)
    else:
        priority = []
        for bucket in (kept, [chosen_code]):
            for code in bucket:
                if code and code not in priority:
                    priority.append(code)
        for code in original:
            if code not in priority:
                priority.append(code)
    removed = {chosen_code, baseline_code}
    selected = (kept + [chosen_code] + [c for c in baseline_fills if c not in removed])[:target_n]
    next_ages = {code: age for code, age in holding_ages.items() if code in set(selected)}
    for code in selected:
        next_ages[code] = next_ages.get(code, 0) + 1
    output = dict(row)
    output["date"] = date
    output["codes"] = priority
    output["alpha"] = percentile_map(priority)
    return output, selected, next_ages, {
        "date": date,
        "policy_applied": 1,
        "best_score": best_score,
        "raw_best_score": raw_best_score,
        "concentration_penalty": concentration_penalty,
        "concentration_penalty_active": concentration_penalty_active,
        "risk_guard_penalty": risk_guard_penalty_value,
        "risk_guard_active": risk_guard_active,
        "replacement_gate_active": replacement_gate_active,
        "replacement_gate_rejected": replacement_gate_rejected,
        "replacement_gate_reason": replacement_gate_reason,
        "threshold": float(args.threshold),
        "baseline_code": baseline_code or "",
        "baseline_position": baseline_pos,
        "chosen_code": chosen_code,
        "chosen_original_position": chosen_original_pos,
        "position_delta": float(chosen_original_pos - baseline_pos)
        if np.isfinite(chosen_original_pos) and np.isfinite(baseline_pos)
        else np.nan,
        "vacancies": vacancies,
        "target_n": target_n,
        "kept_n": len(kept),
        "rewrite_mode": args.rewrite_mode,
    }


def main(argv=None):
    args = parse_args(argv)
    payload = load_model(args.model)
    pairwise = score_pairwise(args.pairwise_dataset, payload)
    pairwise = apply_concentration_penalty(pairwise, args)
    by_date = {date: group for date, group in pairwise.groupby("date", sort=False)}
    rows = load_alpha_rows(args.input_alpha_jsonl)
    out_rows = []
    audits = []
    current_selected = []
    holding_ages = {}
    for row in rows:
        date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        out, current_selected, holding_ages, audit = reorder_day(
            row,
            by_date.get(date),
            current_selected,
            holding_ages,
            args,
        )
        out_rows.append(out)
        audits.append(audit)
    output = write_alpha_rows(args.output_alpha_jsonl, out_rows)
    audit_path = Path(args.output_alpha_jsonl).with_suffix(".pairwise_audit.csv")
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    print({"output": str(output), "audit": str(audit_path), "applied_days": int(sum(a["policy_applied"] for a in audits))})


if __name__ == "__main__":
    main()
