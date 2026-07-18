"""Summarize attribution for ledger-path V3 replacements.

The report focuses on whether the reranker improves portfolio construction:
what it replaced, risk deltas, industry breadth, contribution concentration,
and whether performance comes from a few dates.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--ledger-dataset", required=True)
    parser.add_argument("--baseline-diagnostics", required=True)
    parser.add_argument("--candidate-diagnostics", required=True)
    parser.add_argument("--baseline-returns", required=True)
    parser.add_argument("--candidate-returns", required=True)
    parser.add_argument("--industry-csv", default="data/stock_industry.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    return parser.parse_args(argv)


def normalize_code(code):
    if not isinstance(code, str) or not code.strip():
        return ""
    code = code.strip()
    if "." in code:
        left, right = code.split(".", 1)
        if left.lower() in {"sh", "sz", "bj"}:
            return f"{right}.{left.upper()}"
        return f"{left}.{right.upper()}"
    return code


def finite(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


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
        code = normalize_code(str(row.get(code_col, "")))
        industry = row.get(industry_col)
        if code and isinstance(industry, str) and industry.strip():
            out[code] = industry.strip()
    return out


def read_diag(path):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


def read_returns(path):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


def t_stat(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    std = values.std(ddof=1)
    if std <= 1e-12:
        return np.nan
    return float(values.mean() / (std / math.sqrt(len(values))))


def summarize_replacements(applied, ledger, industry_map):
    merged = applied.merge(
        ledger,
        left_on=["date", "chosen_code", "baseline_code"],
        right_on=["date", "code", "baseline_code"],
        how="left",
        suffixes=("", "_ledger"),
    )
    for col in ("chosen_code", "baseline_code"):
        merged[f"{col}_industry"] = merged[col].map(lambda x: industry_map.get(normalize_code(str(x)), "UNKNOWN"))
    metric_cols = [
        "ledger_path_utility",
        "ledger_weighted_raw_edge",
        "pair_path_raw_edge",
        "pair_risk_delta",
        "pair_downside_delta",
        "pair_quick_fade_delta",
        "pair_rank_delta",
        "baseline_weight",
        "cand_ret_20d",
        "base_ret_20d",
        "diff_ret_20d",
        "cand_beta_60d",
        "base_beta_60d",
        "diff_beta_60d",
        "cand_specific_vol_60d",
        "base_specific_vol_60d",
        "diff_specific_vol_60d",
        "cand_candidate_industry_top_share",
        "base_candidate_industry_top_share",
        "diff_candidate_industry_top_share",
    ]
    summary = {
        "applied_days": int(len(merged)),
        "mean_position_delta": float(merged["position_delta"].mean()) if len(merged) else np.nan,
        "median_position_delta": float(merged["position_delta"].median()) if len(merged) else np.nan,
    }
    for col in metric_cols:
        if col in merged.columns:
            summary[f"mean_{col}"] = float(pd.to_numeric(merged[col], errors="coerce").mean())
            summary[f"median_{col}"] = float(pd.to_numeric(merged[col], errors="coerce").median())
    chosen_industry = (
        merged.groupby("chosen_code_industry")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    baseline_industry = (
        merged.groupby("baseline_code_industry")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    industry_pair = (
        merged.groupby(["baseline_code_industry", "chosen_code_industry"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return merged, summary, chosen_industry, baseline_industry, industry_pair


def summarize_diag_delta(base_diag, cand_diag):
    cols = [
        "gross_weight",
        "market_mult",
        "portfolio_beta_60d",
        "portfolio_beta_per_gross_60d",
        "portfolio_specific_vol_60d",
        "avg_live_age",
        "turnover",
        "desired_turnover",
        "executed_turnover",
        "cost",
        "desired_new_names",
        "band_skipped",
    ]
    merged = base_diag.merge(cand_diag, on="date", suffixes=("_base", "_v3"))
    rows = []
    for col in cols:
        b = pd.to_numeric(merged.get(f"{col}_base"), errors="coerce")
        v = pd.to_numeric(merged.get(f"{col}_v3"), errors="coerce")
        if b is None or v is None:
            continue
        delta = v - b
        rows.append(
            {
                "metric": col,
                "base_mean": float(b.mean()),
                "v3_mean": float(v.mean()),
                "delta_mean": float(delta.mean()),
                "delta_median": float(delta.median()),
                "delta_t": t_stat(delta),
            }
        )
    return pd.DataFrame(rows), merged


def summarize_return_delta(base_ret, cand_ret, applied_dates):
    merged = base_ret.merge(cand_ret, on="date", suffixes=("_base", "_v3"))
    merged["return_delta"] = pd.to_numeric(merged["return_v3"], errors="coerce") - pd.to_numeric(
        merged["return_base"], errors="coerce"
    )
    merged["active_delta"] = pd.to_numeric(merged["active_return_v3"], errors="coerce") - pd.to_numeric(
        merged["active_return_base"], errors="coerce"
    )
    applied_set = set(applied_dates)
    merged["after_applied_signal"] = merged["date"].isin(applied_set).astype(int)
    sorted_delta = merged.sort_values("return_delta", ascending=False)
    total = float(merged["return_delta"].sum())
    top5 = float(sorted_delta["return_delta"].head(5).sum())
    bottom5 = float(sorted_delta["return_delta"].tail(5).sum())
    summary = {
        "days": int(len(merged)),
        "sum_return_delta": total,
        "mean_return_delta": float(merged["return_delta"].mean()),
        "t_return_delta": t_stat(merged["return_delta"]),
        "positive_delta_days": int((merged["return_delta"] > 0).sum()),
        "positive_delta_rate": float((merged["return_delta"] > 0).mean()),
        "top5_delta_sum": top5,
        "bottom5_delta_sum": bottom5,
        "top5_share_of_positive_total": float(top5 / merged.loc[merged["return_delta"] > 0, "return_delta"].sum())
        if (merged["return_delta"] > 0).any()
        else np.nan,
    }
    return merged, summary


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit_csv)
    audit["date"] = pd.to_datetime(audit["date"]).dt.strftime("%Y-%m-%d")
    applied = audit[audit["policy_applied"].eq(1)].copy()
    applied["chosen_code"] = applied["chosen_code"].map(lambda x: normalize_code(str(x)))
    applied["baseline_code"] = applied["baseline_code"].map(lambda x: normalize_code(str(x)))
    ledger = pd.read_parquet(args.ledger_dataset)
    ledger["date"] = pd.to_datetime(ledger["date"]).dt.strftime("%Y-%m-%d")
    ledger["code"] = ledger["code"].map(lambda x: normalize_code(str(x)))
    ledger["baseline_code"] = ledger["baseline_code"].map(lambda x: normalize_code(str(x)))
    industry_map = load_industry(args.industry_csv)
    repl, repl_summary, chosen_industry, baseline_industry, industry_pair = summarize_replacements(
        applied,
        ledger,
        industry_map,
    )
    base_diag = read_diag(args.baseline_diagnostics)
    cand_diag = read_diag(args.candidate_diagnostics)
    diag_delta, diag_merged = summarize_diag_delta(base_diag, cand_diag)
    base_ret = read_returns(args.baseline_returns)
    cand_ret = read_returns(args.candidate_returns)
    return_delta, return_summary = summarize_return_delta(base_ret, cand_ret, applied["date"])

    repl.to_csv(output / "replacement_events.csv", index=False)
    chosen_industry.to_csv(output / "chosen_industry_counts.csv", index=False)
    baseline_industry.to_csv(output / "baseline_industry_counts.csv", index=False)
    industry_pair.to_csv(output / "industry_pair_counts.csv", index=False)
    diag_delta.to_csv(output / "portfolio_state_delta.csv", index=False)
    return_delta.to_csv(output / "return_delta_by_day.csv", index=False)
    summary = {
        "split": args.split_name,
        "replacement_summary": repl_summary,
        "return_delta_summary": return_summary,
        "top_chosen_industries": chosen_industry.head(10).to_dict(orient="records"),
        "top_baseline_industries": baseline_industry.head(10).to_dict(orient="records"),
        "top_industry_pairs": industry_pair.head(10).to_dict(orient="records"),
        "portfolio_state_delta": diag_delta.to_dict(orient="records"),
    }
    (output / "attribution_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
