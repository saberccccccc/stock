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


def compute_breadth(data_dir, start_date, end_date):
    frames = []
    for path in Path(data_dir).glob("*.csv"):
        if not path.name.endswith((".SZ.csv", ".SH.csv", ".BJ.csv")):
            continue
        try:
            df = pd.read_csv(path, usecols=["trade_date", "close"])
        except Exception:
            continue
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        if len(df) < 2:
            continue
        df = df.sort_values("trade_date")
        close = pd.to_numeric(df["close"], errors="coerce")
        ret = close.pct_change()
        tmp = pd.DataFrame(
            {
                "date": df["trade_date"].dt.normalize(),
                "up": ret > 0,
                "valid": ret.notna(),
            }
        )
        frames.append(tmp[tmp["valid"]])
    if not frames:
        raise ValueError(f"No stock data found in {data_dir}")
    all_rows = pd.concat(frames, ignore_index=True)
    breadth = (
        all_rows.groupby("date", sort=True)
        .agg(up_ratio=("up", "mean"), breadth_n=("up", "size"))
        .reset_index()
    )
    return breadth


def transform_row(row, triggered, args, breadth_value):
    codes = list(row.get("codes", []))
    alpha = list(row.get("alpha", []))
    n = len(codes)
    if n == 0:
        return row
    base_n = max(int(n * float(args.base_target_frac)), 1)
    risk_n = max(int(n * float(args.risk_target_frac)), 1)
    keep_n = risk_n if triggered else base_n
    effective_target = float(args.risk_target_frac if triggered else args.base_target_frac)

    if triggered and risk_n < base_n:
        kept = list(range(risk_n))
        demoted = list(range(risk_n, base_n))
        rest = list(range(base_n, n))
        order = kept + rest + demoted
        codes = [codes[i] for i in order]
        if len(alpha) == n:
            alpha = [alpha[i] for i in order]
        else:
            alpha = [float(n - i) for i in range(n)]

    out = dict(row)
    out["codes"] = codes
    out["alpha"] = alpha
    out["breadth_target_transform"] = {
        "triggered": bool(triggered),
        "base_target_frac": float(args.base_target_frac),
        "risk_target_frac": float(args.risk_target_frac),
        "effective_target_frac": effective_target,
        "base_n": int(base_n),
        "risk_n": int(risk_n),
        "keep_n": int(keep_n),
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
    roll_col = f"up_ma{int(args.breadth_window)}"
    breadth[roll_col] = breadth["up_ratio"].rolling(int(args.breadth_window)).mean()
    if args.breadth_output:
        out_b = Path(args.breadth_output)
        out_b.parent.mkdir(parents=True, exist_ok=True)
        breadth.to_csv(out_b, index=False)
    breadth_map = {
        pd.Timestamp(row.date).normalize(): getattr(row, roll_col)
        for row in breadth.itertuples(index=False)
    }

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
