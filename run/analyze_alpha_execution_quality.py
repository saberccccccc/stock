"""Compare Alpha files for chase risk and cross-day rank stability."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run.transform_alpha_for_execution import load_alpha_rows, load_signal_returns


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Alpha execution quality")
    parser.add_argument("--inputs", required=True, help="Comma-separated Alpha JSONL files")
    parser.add_argument("--names", default=None, help="Comma-separated display names")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def summarize(name, rows, signal_returns, top_n):
    daily = []
    previous = set()
    for row in rows:
        selected = list(row.get("codes", []))[:top_n]
        returns = [
            signal_returns.get(pd.Timestamp(row["date"]), {}).get(code)
            for code in selected
        ]
        returns = np.asarray([value for value in returns if value is not None], dtype=float)
        selected_set = set(selected)
        daily.append(
            {
                "name": name,
                "date": pd.Timestamp(row["date"]),
                "mean_signal_return": float(np.mean(returns)) if len(returns) else np.nan,
                "median_signal_return": float(np.median(returns)) if len(returns) else np.nan,
                "share_ge_070": float(np.mean(returns >= 0.07)) if len(returns) else np.nan,
                "share_ge_095": float(np.mean(returns >= 0.095)) if len(returns) else np.nan,
                "overlap_previous": (
                    len(selected_set & previous) / max(len(selected_set), 1)
                    if previous
                    else np.nan
                ),
            }
        )
        previous = selected_set
    return pd.DataFrame(daily)


def main():
    args = parse_args()
    paths = [value.strip() for value in args.inputs.split(",") if value.strip()]
    names = (
        [value.strip() for value in args.names.split(",")]
        if args.names
        else [Path(path).parent.name for path in paths]
    )
    if len(paths) != len(names):
        raise ValueError("names length must match inputs length")

    row_sets = [load_alpha_rows(path) for path in paths]
    all_codes = {code for rows in row_sets for row in rows for code in row.get("codes", [])}
    start = min(rows[0]["date"] for rows in row_sets if rows)
    end = max(rows[-1]["date"] for rows in row_sets if rows)
    signal_returns = load_signal_returns(args.data_dir, all_codes, start, end)
    daily = pd.concat(
        [
            summarize(name, rows, signal_returns, args.top_n)
            for name, rows in zip(names, row_sets)
        ],
        ignore_index=True,
    )
    summary = (
        daily.groupby("name", sort=False)
        .agg(
            dates=("date", "count"),
            mean_signal_return=("mean_signal_return", "mean"),
            median_signal_return=("median_signal_return", "mean"),
            share_ge_070=("share_ge_070", "mean"),
            share_ge_095=("share_ge_095", "mean"),
            overlap_previous=("overlap_previous", "mean"),
        )
        .reset_index()
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    daily.to_csv(output.with_name(f"{output.stem}_daily.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
