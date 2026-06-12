"""Blend two saved alpha rankings using cross-sectional percentile scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Blend alpha JSONL rankings")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-weight", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[pd.Timestamp(row["date"])] = row
    return rows


def percentile_map(codes):
    n = len(codes)
    if n <= 1:
        return {code: 1.0 for code in codes}
    return {code: 1.0 - rank / (n - 1) for rank, code in enumerate(codes)}


def main():
    args = parse_args()
    weight = float(args.left_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("left-weight must be between 0 and 1")

    left = load_rows(args.left)
    right = load_rows(args.right)
    common_dates = sorted(set(left) & set(right))
    if not common_dates:
        raise ValueError("No common dates between alpha files")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for date in common_dates:
            left_pct = percentile_map(left[date]["codes"])
            right_pct = percentile_map(right[date]["codes"])
            codes = sorted(set(left_pct) & set(right_pct))
            scores = np.asarray(
                [
                    weight * left_pct[code] + (1.0 - weight) * right_pct[code]
                    for code in codes
                ],
                dtype=np.float64,
            )
            order = np.argsort(scores)[::-1]
            ordered_codes = np.asarray(codes, dtype=object)[order].tolist()
            ordered_scores = scores[order].tolist()
            row = {
                "date": date.strftime("%Y-%m-%d"),
                "codes": ordered_codes,
                "alpha": ordered_scores,
                "n_stocks": len(ordered_codes),
                "blend_left_weight": weight,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(common_dates)} blended dates to {output}")


if __name__ == "__main__":
    main()
