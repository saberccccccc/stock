"""Compare two pairwise policy audits at the replacement-event level."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "candidate_position",
    "baseline_position",
    "baseline_weight",
    "ledger_cost",
    "pair_risk_delta",
    "pair_rank_delta",
    "diag_portfolio_beta_60d",
    "diag_portfolio_specific_vol_60d",
    "diag_turnover",
    "diag_cost",
    "diff_candidate_industry_top_share",
    "cand_candidate_industry_top_share",
    "base_candidate_industry_top_share",
    "diff_ret_1d",
    "diff_ret_5d",
    "diff_ret_20d",
    "diff_drawdown_20d",
    "diff_beta_60d",
    "diff_specific_vol_60d",
    "diff_vol_20d",
    "diff_vol_60d",
    "diff_money_ma20",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-audit", required=True)
    p.add_argument("--challenger-audit", required=True)
    p.add_argument("--pairwise-dataset", required=True)
    p.add_argument("--reference-name", default="reference")
    p.add_argument("--challenger-name", default="challenger")
    p.add_argument("--split-name", required=True)
    p.add_argument("--industry-csv", default="data/stock_industry.csv")
    p.add_argument("--output-dir", required=True)
    return p.parse_args(argv)


def normalize_code(code):
    code = str(code).strip()
    if not code or code.lower() == "nan":
        return ""
    if "." in code:
        left, right = code.split(".", 1)
        if left.lower() in {"sh", "sz", "bj"}:
            return f"{right}.{left.upper()}"
        return f"{left}.{right.upper()}"
    return code


def load_audit(path, prefix):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    for col in ("baseline_code", "chosen_code"):
        if col in frame.columns:
            frame[col] = frame[col].map(normalize_code)
    keep = [
        "date",
        "policy_applied",
        "best_score",
        "raw_best_score",
        "concentration_penalty",
        "baseline_code",
        "chosen_code",
        "position_delta",
        "vacancies",
        "target_n",
        "kept_n",
    ]
    keep = [c for c in keep if c in frame.columns]
    return frame[keep].add_prefix(f"{prefix}_").rename(columns={f"{prefix}_date": "date"})


def load_pairwise(path):
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame["code"] = frame["code"].map(normalize_code)
    frame["baseline_code"] = frame["baseline_code"].map(normalize_code)
    return frame


def load_industry(path):
    p = Path(path)
    if not p.exists():
        return {}
    frame = pd.read_csv(p)
    code_col = "code" if "code" in frame.columns else frame.columns[0]
    industry_col = "industry" if "industry" in frame.columns else None
    if industry_col is None:
        return {}
    out = {}
    for _, row in frame.iterrows():
        code = normalize_code(row.get(code_col, ""))
        industry = row.get(industry_col, "")
        if code and isinstance(industry, str) and industry.strip():
            out[code] = industry.strip()
    return out


def safe_mean(frame, cols):
    rows = []
    for col in cols:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        rows.append(
            {
                "metric": col,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
                "n": int(values.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def attach_features(events, pairwise, code_col, baseline_col):
    if events.empty:
        return events.copy()
    keys = events[["date", code_col, baseline_col]].rename(
        columns={code_col: "code", baseline_col: "baseline_code"}
    ).reset_index(drop=True)
    keys["code"] = keys["code"].map(normalize_code)
    keys["baseline_code"] = keys["baseline_code"].map(normalize_code)
    features = pairwise[["date", "code", "baseline_code"] + [c for c in FEATURE_COLS if c in pairwise.columns]]
    merged = keys.merge(features, on=["date", "code", "baseline_code"], how="left")
    feature_cols = [c for c in FEATURE_COLS if c in pairwise.columns]
    missing = merged[feature_cols].isna().all(axis=1) if feature_cols else pd.Series(False, index=merged.index)
    if missing.any():
        # Once policies diverge, the audited baseline slot can differ from the
        # baseline used to build inference rows.  Candidate state features are
        # still meaningful by date+code, so use that as a fallback for
        # attribution only.
        fallback_features = (
            pairwise[["date", "code"] + feature_cols]
            .sort_values(["date", "code"])
            .drop_duplicates(["date", "code"], keep="first")
        )
        fallback = keys.loc[missing, ["date", "code"]].merge(fallback_features, on=["date", "code"], how="left")
        for col in feature_cols:
            merged.loc[missing, col] = fallback[col].to_numpy()
    out = events.reset_index(drop=True).join(merged.drop(columns=["date", "code", "baseline_code"]))
    return out


def industry_counts(events, industry_map, code_col):
    if events.empty or code_col not in events.columns:
        return pd.DataFrame(columns=["industry", "count"])
    rows = []
    for code in events[code_col].map(normalize_code):
        rows.append(industry_map.get(code, "UNKNOWN"))
    return (
        pd.Series(rows, name="industry")
        .value_counts()
        .reset_index()
        .rename(columns={"count": "count"})
    )


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = load_audit(args.reference_audit, "ref")
    ch = load_audit(args.challenger_audit, "chg")
    pairwise = load_pairwise(args.pairwise_dataset)
    industry_map = load_industry(args.industry_csv)

    merged = ref.merge(ch, on="date", how="outer").sort_values("date")
    for col in ("ref_policy_applied", "chg_policy_applied"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
    merged["same_chosen"] = (
        merged["ref_chosen_code"].fillna("").map(normalize_code)
        == merged["chg_chosen_code"].fillna("").map(normalize_code)
    )
    merged["decision_changed"] = (
        (merged["ref_policy_applied"] != merged["chg_policy_applied"]) | (~merged["same_chosen"])
    ).astype(int)
    changed = merged[merged["decision_changed"].eq(1)].copy()
    only_ref = changed[(changed["ref_policy_applied"].eq(1)) & (changed["chg_policy_applied"].eq(0))].copy()
    only_ch = changed[(changed["ref_policy_applied"].eq(0)) & (changed["chg_policy_applied"].eq(1))].copy()
    both_diff = changed[
        (changed["ref_policy_applied"].eq(1))
        & (changed["chg_policy_applied"].eq(1))
        & (~changed["same_chosen"])
    ].copy()

    ref_events = attach_features(changed[changed["ref_policy_applied"].eq(1)], pairwise, "ref_chosen_code", "ref_baseline_code")
    ch_events = attach_features(changed[changed["chg_policy_applied"].eq(1)], pairwise, "chg_chosen_code", "chg_baseline_code")
    ref_summary = safe_mean(ref_events, FEATURE_COLS)
    ch_summary = safe_mean(ch_events, FEATURE_COLS)
    metric_delta = ref_summary.merge(ch_summary, on="metric", suffixes=("_ref", "_challenger"))
    if not metric_delta.empty:
        metric_delta["mean_delta_challenger_minus_ref"] = metric_delta["mean_challenger"] - metric_delta["mean_ref"]

    merged.to_csv(out / "daily_decision_diff.csv", index=False)
    changed.to_csv(out / "changed_decisions.csv", index=False)
    ref_events.to_csv(out / "reference_changed_events_with_features.csv", index=False)
    ch_events.to_csv(out / "challenger_changed_events_with_features.csv", index=False)
    metric_delta.to_csv(out / "changed_event_feature_delta.csv", index=False)
    industry_counts(ref_events, industry_map, "ref_chosen_code").to_csv(out / "reference_changed_industries.csv", index=False)
    industry_counts(ch_events, industry_map, "chg_chosen_code").to_csv(out / "challenger_changed_industries.csv", index=False)

    summary = {
        "split_name": args.split_name,
        "reference_name": args.reference_name,
        "challenger_name": args.challenger_name,
        "days": int(len(merged)),
        "changed_days": int(changed["date"].nunique()),
        "reference_applied_days": int(merged["ref_policy_applied"].sum()),
        "challenger_applied_days": int(merged["chg_policy_applied"].sum()),
        "only_reference_days": int(len(only_ref)),
        "only_challenger_days": int(len(only_ch)),
        "both_applied_different_choice_days": int(len(both_diff)),
        "feature_delta": metric_delta.to_dict(orient="records"),
    }
    (out / "policy_diff_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
