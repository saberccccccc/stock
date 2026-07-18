"""Compare Alpha rankings with execution-neutral future open returns."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import load_alpha_rows
from backtest.open_ledger import load_ohlc_money
from core.research_protocol import assert_alpha_rows_within_research, assert_research_end_date


def parse_specs(raw):
    specs = []
    for item in raw.split(","):
        if not item.strip():
            continue
        name, path = item.split("=", 1)
        specs.append((name.strip(), Path(path.strip())))
    if not specs:
        raise ValueError("No alpha specs provided")
    return specs


def ranked_open_return(row, open_df, top_n, horizon, execution_lag=0):
    signal_date = pd.Timestamp(row["date"])
    signal_pos = open_df.index.get_indexer([signal_date])[0]
    if signal_pos < 0:
        return np.nan
    entry_pos = signal_pos + 1 + int(execution_lag)
    exit_pos = entry_pos + int(horizon)
    if exit_pos >= len(open_df.index):
        return np.nan

    codes = [code for code in row["codes"][: int(top_n)] if code in open_df.columns]
    if not codes:
        return np.nan
    entry = open_df.loc[open_df.index[entry_pos], codes].to_numpy(dtype=float)
    exit_ = open_df.loc[open_df.index[exit_pos], codes].to_numpy(dtype=float)
    valid = np.isfinite(entry) & np.isfinite(exit_) & (entry > 0)
    if not valid.any():
        return np.nan
    return float(np.mean(exit_[valid] / entry[valid] - 1.0))


def summarize_daily(daily):
    rows = []
    for (model, lag, horizon), group in daily.groupby(["model", "execution_lag", "horizon"]):
        values = group["return"].dropna().to_numpy(float)
        if not len(values):
            continue
        std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        sharpe = float(np.mean(values) / std * np.sqrt(252)) if std > 0 else np.nan
        months = group.assign(month=group["date"].dt.to_period("M")).groupby("month")["return"].sum()
        rows.append({
            "model": model,
            "execution_lag": int(lag),
            "horizon": int(horizon),
            "days": int(len(values)),
            "mean_return_pct": float(np.mean(values) * 100),
            "median_return_pct": float(np.median(values) * 100),
            "daily_sharpe": sharpe,
            "positive_day_rate": float(np.mean(values > 0)),
            "positive_month_rate": float(np.mean(months > 0)),
        })
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare future open returns of Alpha rankings")
    parser.add_argument("--alpha-specs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--execution-lags", default="0,1")
    parser.add_argument("--money-scale", type=float, default=1000.0)
    parser.add_argument("--max-data-date", default="2024-12-31")
    return parser.parse_args()


def main():
    args = parse_args()
    specs = parse_specs(args.alpha_specs)
    row_sets = {}
    all_codes = set()
    for name, path in specs:
        rows = load_alpha_rows(path, timestamp_dates=True)
        assert_alpha_rows_within_research(rows, context=f"open-return comparison {name}")
        row_sets[name] = rows
        all_codes.update(code for row in rows for code in row["codes"][: args.top_n])

    open_df, _, _ = load_ohlc_money(args.data_dir, sorted(all_codes), args.money_scale, 1000)
    cutoff = assert_research_end_date(args.max_data_date, context="open-return comparison")
    open_df = open_df.loc[open_df.index <= cutoff]
    horizons = [int(value) for value in args.horizons.split(",") if value.strip()]
    lags = [int(value) for value in args.execution_lags.split(",") if value.strip()]

    daily_rows = []
    for model, rows in row_sets.items():
        for row in rows:
            for lag in lags:
                for horizon in horizons:
                    daily_rows.append({
                        "model": model,
                        "date": pd.Timestamp(row["date"]),
                        "execution_lag": lag,
                        "horizon": horizon,
                        "return": ranked_open_return(row, open_df, args.top_n, horizon, lag),
                    })
    daily = pd.DataFrame(daily_rows)
    summary = summarize_daily(daily)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "daily_top_open_returns.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
