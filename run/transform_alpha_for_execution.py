"""Transform saved Alpha rankings using only signal-day and prior information."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import load_alpha_rows, write_alpha_rows
from alpha.transforms import (
    load_signal_returns,
    load_stall_signals,
    transform_execution_rows,
)

# Backward-compatible import for older tests/scripts.
transform_rows = transform_execution_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Make Alpha rankings more executable")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--stability-window", type=int, default=0)
    parser.add_argument("--current-weight", type=float, default=1.0)
    parser.add_argument("--max-signal-return", type=float, default=None)
    parser.add_argument("--stall-surge-return", type=float, default=None)
    parser.add_argument("--stall-recent-abs-return", type=float, default=0.03)
    parser.add_argument("--stall-max-range", type=float, default=0.08)
    parser.add_argument("--stall-surge-lookback", type=int, default=20)
    parser.add_argument("--stall-recent-window", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_alpha_rows(args.input, timestamp_dates=True)
    if not rows:
        raise ValueError("Input Alpha file is empty")
    codes = {code for row in rows for code in row.get("codes", [])}
    signal_returns = load_signal_returns(
        args.data_dir,
        codes,
        rows[0]["date"],
        rows[-1]["date"],
    )
    stall_config = None
    stall_signals = {}
    if args.stall_surge_return is not None:
        stall_config = {
            "surge_return": args.stall_surge_return,
            "recent_abs_return": args.stall_recent_abs_return,
            "max_range": args.stall_max_range,
            "surge_lookback": args.stall_surge_lookback,
            "recent_window": args.stall_recent_window,
        }
        stall_signals = load_stall_signals(
            args.data_dir,
            codes,
            rows[0]["date"],
            rows[-1]["date"],
            **stall_config,
        )
    transformed = transform_execution_rows(
        rows,
        signal_returns,
        stability_window=args.stability_window,
        current_weight=args.current_weight,
        max_signal_return=args.max_signal_return,
        stall_signals=stall_signals,
        stall_config=stall_config,
    )
    output = write_alpha_rows(args.output, transformed)
    demoted = sum(row["execution_transform"]["demoted_count"] for row in transformed)
    stall_demoted = sum(
        row["execution_transform"]["stall_demoted_count"] for row in transformed
    )
    print(
        json.dumps(
            {
                "input": args.input,
                "output": str(Path(output)),
                "dates": len(transformed),
                "codes": len(codes),
                "demoted_total": demoted,
                "stall_demoted_total": stall_demoted,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
