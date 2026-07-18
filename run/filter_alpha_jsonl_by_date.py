"""Filter alpha JSONL rows by inclusive date range.

This keeps the alpha row schema unchanged and is intended for reproducible
APM split construction, e.g. test_2025 and forward_2026.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Filter alpha JSONL by date.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if end < start:
        raise ValueError("--end-date must be >= --start-date")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    seen = 0
    first_date = None
    last_date = None
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            seen += 1
            row = json.loads(line)
            dt = pd.Timestamp(str(row["date"])[:10])
            if start <= dt <= end:
                kept += 1
                first_date = dt if first_date is None else min(first_date, dt)
                last_date = dt if last_date is None else max(last_date, dt)
                dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(
        f"Filtered {kept}/{seen} rows to {output_path} "
        f"range={first_date.date() if first_date is not None else 'NA'}.."
        f"{last_date.date() if last_date is not None else 'NA'}"
    )


if __name__ == "__main__":
    main()
