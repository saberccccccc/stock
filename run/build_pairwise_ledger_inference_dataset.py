"""Build feature-only pairwise rows for ledger-path V3 inference.

This script intentionally does not read future return labels.  By default it
also avoids execution lookahead: for a signal date it uses the latest baseline
open-ledger diagnostic available on or before that signal date, then estimates
the baseline fill's planned weight from the known target_n and market_mult.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = (
    "candidate_rank_pct",
    "candidate_industry_top_share",
    "top_industry_share",
    "top_industry_hhi",
    "global_us_hk_pressure",
    "global_defensive_pressure",
    "global_hk_risk_pressure",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "vol_60d",
    "drawdown_20d",
    "beta_60d",
    "specific_vol_60d",
    "money_ma20",
    "was_held",
    "holding_age",
    "protected_fill",
    "baseline_fill",
)

DIAG_FEATURES = (
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
    "active_drawdown_trailing_return",
    "global_risk_pressure",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dataset", required=True)
    parser.add_argument("--diagnostics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--max-pairs-per-day", type=int, default=80)
    parser.add_argument("--base-cost-bps", type=float, default=25.0)
    parser.add_argument("--risk-cost", type=float, default=0.0025)
    parser.add_argument("--min-baseline-weight", type=float, default=0.002)
    parser.add_argument(
        "--diag-timing",
        choices=("previous", "next"),
        default="previous",
        help=(
            "previous is no-lookahead and uses diagnostics on/before the signal date; "
            "next reproduces research-only diagnostics after baseline execution."
        ),
    )
    parser.add_argument("--max-weight", type=float, default=0.05)
    return parser.parse_args(argv)


def finite(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def parse_holdings(text):
    out = {}
    if not isinstance(text, str) or not text:
        return out
    for part in text.split(";"):
        if "=" not in part:
            continue
        code, weight = part.split("=", 1)
        value = finite(weight)
        if np.isfinite(value):
            out[str(code).strip()] = float(value)
    return out


def risk_load(row):
    vol = max(finite(row.get("specific_vol_60d"), 0.0), 0.0)
    beta = finite(row.get("beta_60d"), 1.0)
    crowd = max(finite(row.get("candidate_industry_top_share"), 0.0), 0.0)
    ret20 = max(finite(row.get("ret_20d"), 0.0), 0.0)
    return (
        np.clip(vol / 0.20, 0.0, 3.0) * 0.35
        + np.clip((beta - 1.0) / 1.0, 0.0, 2.0) * 0.25
        + np.clip(crowd / 0.50, 0.0, 2.0) * 0.25
        + np.clip(ret20 / 0.30, 0.0, 2.0) * 0.15
    )


def load_diag(path):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["holdings_map"] = frame["holdings"].map(parse_holdings)
    keep = ["date", "holdings_map"] + [c for c in DIAG_FEATURES if c in frame.columns]
    return frame[keep].sort_values("date").drop_duplicates("date", keep="last")


def diag_lookup(diag_dates, timing):
    diag_dates = np.asarray(pd.to_datetime(diag_dates), dtype="datetime64[ns]")

    def lookup(signal_date):
        value = np.datetime64(pd.Timestamp(signal_date).to_datetime64())
        if timing == "next":
            pos = int(np.searchsorted(diag_dates, value, side="right"))
        else:
            pos = int(np.searchsorted(diag_dates, value, side="right")) - 1
        if pos < 0:
            return None
        if pos >= len(diag_dates):
            return None
        return pd.Timestamp(diag_dates[pos]).normalize()

    return lookup


def planned_baseline_weight(base, holdings, diag_row, args):
    held_weight = finite(holdings.get(str(base["code"])), 0.0)
    target_n = max(int(finite(base.get("target_n"), 1.0)), 1)
    market_mult = finite(getattr(diag_row, "market_mult", np.nan))
    gross = finite(getattr(diag_row, "gross_weight", np.nan))
    if not np.isfinite(market_mult):
        market_mult = gross if np.isfinite(gross) and gross > 0 else 1.0
    planned = min(float(args.max_weight), max(float(market_mult), 0.0) / target_n)
    return max(held_weight, planned)


def build_dataset(policy, diag, args):
    rows = []
    diag_by_date = {pd.Timestamp(row.date).normalize(): row for row in diag.itertuples(index=False)}
    lookup_exec_date = diag_lookup(diag["date"], args.diag_timing)
    eligible = policy[policy["eligible"].eq(1)].copy()
    for signal_date, group in eligible.groupby("date", sort=True):
        exec_date = lookup_exec_date(signal_date)
        if exec_date is None or exec_date not in diag_by_date:
            continue
        diag_row = diag_by_date[exec_date]
        holdings = getattr(diag_row, "holdings_map")
        if not holdings:
            continue
        group = group.sort_values("candidate_position")
        baseline = group[group["baseline_fill"].eq(1)].head(1)
        if baseline.empty:
            baseline = group.head(1)
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        base_code = str(base["code"])
        if args.diag_timing == "next":
            base_weight = finite(holdings.get(base_code), 0.0)
        else:
            base_weight = planned_baseline_weight(base, holdings, diag_row, args)
        if base_weight < float(args.min_baseline_weight):
            continue
        base_risk = risk_load(base)
        for _, cand in group.head(int(args.max_pairs_per_day)).iterrows():
            cand_code = str(cand["code"])
            cand_risk = risk_load(cand)
            risk_delta = cand_risk - base_risk
            already_held_weight = finite(holdings.get(cand_code), 0.0)
            rank_delta = max(0.0, finite(cand.get("candidate_position"), 0.0) - finite(base.get("candidate_position"), 0.0))
            turnover_weight = 0.0 if already_held_weight > 0 else base_weight
            rec = {
                "split": args.split_name,
                "date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                "execution_date": exec_date.strftime("%Y-%m-%d"),
                "code": cand_code,
                "baseline_code": base_code,
                "candidate_position": int(cand["candidate_position"]),
                "baseline_position": int(base["candidate_position"]),
                "is_baseline": int(cand_code == base_code),
                "label_available": 0,
                "eligible": 1,
                "baseline_weight": float(base_weight),
                "candidate_already_held_weight": float(already_held_weight),
                "ledger_cost": float(turnover_weight * float(args.base_cost_bps) / 10000.0),
                "pair_risk_delta": float(risk_delta),
                "pair_rank_delta": float(rank_delta),
            }
            for col in DIAG_FEATURES:
                rec[f"diag_{col}"] = finite(getattr(diag_row, col, np.nan))
            for col in FEATURES:
                cval = finite(cand.get(col))
                bval = finite(base.get(col))
                rec[f"cand_{col}"] = cval
                rec[f"base_{col}"] = bval
                rec[f"diff_{col}"] = cval - bval if np.isfinite(cval) and np.isfinite(bval) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse_args(argv)
    policy = pd.read_parquet(args.policy_dataset)
    policy["date"] = pd.to_datetime(policy["date"]).dt.normalize()
    diag = load_diag(args.diagnostics_csv)
    dataset = build_dataset(policy, diag, args)
    if dataset.empty:
        raise ValueError("ledger inference dataset is empty")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output / "pairwise_ledger_inference_dataset.parquet", index=False)
    summary = {
        "policy_dataset": str(args.policy_dataset),
        "diagnostics_csv": str(args.diagnostics_csv),
        "split_name": args.split_name,
        "rows": int(len(dataset)),
        "dates": int(dataset["date"].nunique()),
        "mean_baseline_weight": float(dataset["baseline_weight"].mean()),
        "mean_pair_risk_delta": float(dataset["pair_risk_delta"].mean()),
        "feature_only": True,
        "uses_future_labels": False,
        "uses_future_execution_diagnostics": bool(args.diag_timing == "next"),
        "params": vars(args),
    }
    (output / "pairwise_ledger_inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
