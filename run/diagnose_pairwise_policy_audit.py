"""Diagnose pairwise replacement audit files against open-ledger returns."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--returns-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    return parser.parse_args(argv)


def bucket_position_delta(values):
    bins = [-np.inf, 0, 5, 10, 20, 40, np.inf]
    labels = ["same_or_above", "1_5", "6_10", "11_20", "21_40", "40_plus"]
    return pd.cut(values, bins=bins, labels=labels)


def summarize(audit, returns=None):
    audit = audit.copy()
    audit["date"] = pd.to_datetime(audit["date"]).dt.strftime("%Y-%m-%d")
    audit["position_delta_bucket"] = bucket_position_delta(pd.to_numeric(audit["position_delta"], errors="coerce"))
    rows = []
    rows.append({"metric": "days", "value": len(audit)})
    rows.append({"metric": "applied_days", "value": int(audit["policy_applied"].sum())})
    rows.append({"metric": "apply_rate", "value": float(audit["policy_applied"].mean())})
    applied = audit[audit["policy_applied"].eq(1)]
    if not applied.empty:
        rows.append({"metric": "mean_best_score_applied", "value": float(applied["best_score"].mean())})
        rows.append({"metric": "mean_position_delta", "value": float(applied["position_delta"].mean())})
        rows.append({"metric": "median_position_delta", "value": float(applied["position_delta"].median())})
        rows.append({"metric": "mean_vacancies_applied", "value": float(applied["vacancies"].mean())})
    if returns is not None and not returns.empty:
        ret = returns.copy()
        date_col = "date" if "date" in ret.columns else "trade_date"
        ret["date"] = pd.to_datetime(ret[date_col]).dt.strftime("%Y-%m-%d")
        ret_col = "daily_return" if "daily_return" in ret.columns else "return"
        merged = audit.merge(ret[["date", ret_col]], on="date", how="left")
        rows.append({"metric": "mean_return_applied_days", "value": float(merged.loc[merged.policy_applied.eq(1), ret_col].mean())})
        rows.append({"metric": "mean_return_no_apply_days", "value": float(merged.loc[merged.policy_applied.eq(0), ret_col].mean())})
    bucket = (
        applied.groupby("position_delta_bucket", observed=False)
        .agg(
            days=("policy_applied", "size"),
            mean_score=("best_score", "mean"),
            mean_delta=("position_delta", "mean"),
        )
        .reset_index()
    )
    return pd.DataFrame(rows), bucket


def main(argv=None):
    args = parse_args(argv)
    audit = pd.read_csv(args.audit_csv)
    returns = pd.read_csv(args.returns_csv) if args.returns_csv and Path(args.returns_csv).exists() else None
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary, bucket = summarize(audit, returns)
    summary.to_csv(output / "audit_summary.csv", index=False)
    bucket.to_csv(output / "position_delta_bucket.csv", index=False)
    result = {
        "split_name": args.split_name,
        "audit_csv": str(args.audit_csv),
        "returns_csv": str(args.returns_csv) if args.returns_csv else None,
        "summary": summary.to_dict(orient="records"),
    }
    (output / "audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
