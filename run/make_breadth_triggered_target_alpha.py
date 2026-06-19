"""Shrink alpha top lists when market breadth is weak on the signal date.

Breadth is computed from same-day close-to-close stock returns, which is known
after the signal-day close and before next-open execution.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import iter_alpha_rows, load_alpha_dates, write_alpha_rows
from alpha.market_overlays import compute_breadth, rolling_breadth_map, shrink_target_row
from core.research_protocol import (
    assert_alpha_dates_within_forward,
    assert_alpha_dates_within_research,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Market-breadth-triggered target shrink transform")
    parser.add_argument("--alpha-jsonl", required=True)
    parser.add_argument("--output-alpha", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--base-target-frac", type=float, default=0.006)
    parser.add_argument("--risk-target-frac", type=float, required=True)
    parser.add_argument("--breadth-window", type=int, default=3)
    parser.add_argument("--breadth-below", type=float, required=True)
    parser.add_argument("--start-pad-days", type=int, default=30)
    parser.add_argument("--breadth-output", default=None)
    parser.add_argument("--allow-forward", action="store_true")
    return parser.parse_args()


def transform_row(row, triggered, args, breadth_value):
    out, details = shrink_target_row(
        row,
        triggered=triggered,
        base_target_frac=args.base_target_frac,
        risk_target_frac=args.risk_target_frac,
    )
    if details is None:
        return row
    out["breadth_target_transform"] = {
        **details,
        "breadth_window": int(args.breadth_window),
        "breadth_below": float(args.breadth_below),
        "breadth_value": None if pd.isna(breadth_value) else float(breadth_value),
    }
    return out


def main():
    args = parse_args()
    if not 0 < args.risk_target_frac <= args.base_target_frac:
        raise ValueError("risk-target-frac must be in (0, base-target-frac]")
    dates = load_alpha_dates(args.alpha_jsonl)
    if args.allow_forward:
        assert_alpha_dates_within_forward(dates, context="breadth target forward transform")
    else:
        assert_alpha_dates_within_research(dates, context="breadth target research transform")
    start = min(dates) - pd.Timedelta(days=max(int(args.start_pad_days), int(args.breadth_window) * 3))
    end = max(dates)
    breadth = compute_breadth(args.data_dir, start, end)
    breadth_map = rolling_breadth_map(breadth, args.breadth_window)
    if args.breadth_output:
        out_b = Path(args.breadth_output)
        out_b.parent.mkdir(parents=True, exist_ok=True)
        breadth.to_csv(out_b, index=False)

    output = Path(args.output_alpha)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    triggered_rows = 0
    def transformed_rows():
        nonlocal rows, triggered_rows
        for row in iter_alpha_rows(args.alpha_jsonl):
            date = pd.Timestamp(row["date"]).normalize()
            value = breadth_map.get(date, np.nan)
            triggered = bool(pd.notna(value) and float(value) <= float(args.breadth_below))
            out = transform_row(row, triggered, args, value)
            rows += 1
            triggered_rows += int(triggered)
            yield out

    write_alpha_rows(output, transformed_rows())

    print(
        json.dumps(
            {
                "output": str(output),
                "rows": rows,
                "triggered_rows": triggered_rows,
                "risk_target_frac": args.risk_target_frac,
                "breadth_window": args.breadth_window,
                "breadth_below": args.breadth_below,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
