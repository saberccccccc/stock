"""Compare Alpha files for chase risk and cross-day rank stability."""

import argparse
from pathlib import Path

import pandas as pd

from alpha.diagnostics import execution_quality_daily, summarize_execution_quality
from alpha.io import load_alpha_rows
from alpha.transforms import load_signal_returns


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Alpha execution quality")
    parser.add_argument("--inputs", required=True, help="Comma-separated Alpha JSONL files")
    parser.add_argument("--names", default=None, help="Comma-separated display names")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


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

    row_sets = [load_alpha_rows(path, timestamp_dates=True) for path in paths]
    all_codes = {code for rows in row_sets for row in rows for code in row.get("codes", [])}
    start = min(rows[0]["date"] for rows in row_sets if rows)
    end = max(rows[-1]["date"] for rows in row_sets if rows)
    signal_returns = load_signal_returns(args.data_dir, all_codes, start, end)
    daily = pd.concat(
        [
            execution_quality_daily(name, rows, signal_returns, args.top_n)
            for name, rows in zip(names, row_sets)
        ],
        ignore_index=True,
    )
    summary = summarize_execution_quality(daily)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    daily.to_csv(output.with_name(f"{output.stem}_daily.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
