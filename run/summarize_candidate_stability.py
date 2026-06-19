"""Summarize candidate-vs-base stability by month and market state."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.market_state import load_index_states


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize candidate stability")
    parser.add_argument("--compare-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index-file", default="data/raw/hs300_index.csv")
    parser.add_argument("--ma-window", type=int, default=60)
    parser.add_argument("--crash-ret", type=float, default=-0.03)
    parser.add_argument("--tail-start", default="2026-04-01")
    parser.add_argument("--tail-end", default="2026-05-18")
    return parser.parse_args()


def load_state(index_file, ma_window, crash_ret):
    return load_index_states(index_file, ma_window=ma_window, crash_ret=crash_ret)


def summarize_group(df, group_cols):
    return df.groupby(group_cols).agg(
        days=("diff", "size"),
        diff_sum=("diff", "sum"),
        diff_mean=("diff", "mean"),
        win_rate=("diff", lambda x: float((x > 0).mean())),
        base_return=("return_base", lambda x: float((1.0 + x).prod() - 1.0)),
        candidate_return=("return_candidate", lambda x: float((1.0 + x).prod() - 1.0)),
    ).reset_index()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(Path(args.compare_dir) / "daily_return_compare.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    for col in ["return_base", "return_candidate", "diff", "portfolio_value"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    state = load_state(args.index_file, args.ma_window, args.crash_ret)
    daily = daily.merge(state, on="date", how="left")
    daily["market_state"] = daily["market_state"].fillna("unknown")
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    daily["year"] = daily["date"].dt.year
    tail_start = pd.Timestamp(args.tail_start)
    tail_end = pd.Timestamp(args.tail_end)
    if tail_end < tail_start:
        raise ValueError("tail-end must be on or after tail-start")
    daily["is_tail_period"] = daily["date"].between(tail_start, tail_end)

    monthly = summarize_group(daily, ["portfolio_value", "month"])
    monthly["candidate_minus_base_return"] = monthly["candidate_return"] - monthly["base_return"]
    monthly.to_csv(out_dir / "monthly_stability.csv", index=False)

    state_summary = summarize_group(daily, ["portfolio_value", "market_state"])
    state_summary["candidate_minus_base_return"] = state_summary["candidate_return"] - state_summary["base_return"]
    state_summary.to_csv(out_dir / "market_state_stability.csv", index=False)

    tail_summary = summarize_group(daily, ["portfolio_value", "is_tail_period"])
    tail_summary["candidate_minus_base_return"] = tail_summary["candidate_return"] - tail_summary["base_return"]
    tail_summary.to_csv(out_dir / "tail_dependency.csv", index=False)

    overall = summarize_group(daily, ["portfolio_value"])
    overall["candidate_minus_base_return"] = overall["candidate_return"] - overall["base_return"]
    monthly_win = monthly.groupby("portfolio_value").agg(
        months=("month", "size"),
        positive_months=("candidate_minus_base_return", lambda x: int((x > 0).sum())),
        monthly_win_rate=("candidate_minus_base_return", lambda x: float((x > 0).mean())),
        best_month_diff=("candidate_minus_base_return", "max"),
        worst_month_diff=("candidate_minus_base_return", "min"),
    ).reset_index()
    result = overall.merge(monthly_win, on="portfolio_value", how="left")
    result.to_csv(out_dir / "overall_stability.csv", index=False)
    print(result.to_string(index=False), flush=True)
    print(state_summary.to_string(index=False), flush=True)
    print(tail_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
