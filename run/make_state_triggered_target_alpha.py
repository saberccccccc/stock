"""Shrink alpha top lists on dates triggered by recent strategy state.

The transform is intentionally label-free for a given signal date: it computes
state from prior executed portfolio returns and then demotes names outside the
smaller target bucket while preserving the original ranking order.
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
from alpha.market_overlays import shrink_target_row
from core.research_protocol import (
    assert_alpha_dates_within_forward,
    assert_alpha_dates_within_research,
)


def parse_args():
    parser = argparse.ArgumentParser(description="State-triggered target shrink alpha transform")
    parser.add_argument("--alpha-jsonl", required=True)
    parser.add_argument("--returns-csv", required=True)
    parser.add_argument("--output-alpha", required=True)
    parser.add_argument("--base-target-frac", type=float, default=0.006)
    parser.add_argument("--risk-target-frac", type=float, required=True)
    parser.add_argument("--trigger-window", type=int, default=5)
    parser.add_argument("--trigger-cumret-below", type=float, default=None)
    parser.add_argument("--trigger-drawdown-above", type=float, default=None)
    parser.add_argument("--min-trigger-days", type=int, default=1)
    parser.add_argument("--allow-forward", action="store_true")
    return parser.parse_args()


def load_returns(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["return"] = pd.to_numeric(df["return"], errors="coerce").fillna(0.0)
    df = df.sort_values("date").reset_index(drop=True)
    df["nav"] = (1.0 + df["return"]).cumprod()
    df["prior_nav"] = df["nav"].shift(1)
    df["prior_nav"] = df["prior_nav"].fillna(1.0)
    return df


def build_trigger_by_signal_date(args):
    returns = load_returns(args.returns_csv)
    by_date = {}
    for i, row in returns.iterrows():
        # returns date is execution/holding result after the previous signal.
        # For the same calendar signal date, only use returns strictly before it.
        prior = returns.iloc[max(0, i - args.trigger_window):i]
        if len(prior) < max(int(args.min_trigger_days), 1):
            by_date[pd.Timestamp(row["date"]).normalize()] = {
                "triggered": False,
                "cumret": np.nan,
                "drawdown": np.nan,
            }
            continue
        nav = (1.0 + prior["return"]).cumprod()
        cumret = float(nav.iloc[-1] - 1.0)
        drawdown = float(1.0 - nav.iloc[-1] / nav.cummax().iloc[-1])
        triggered = False
        if args.trigger_cumret_below is not None and cumret <= float(args.trigger_cumret_below):
            triggered = True
        if args.trigger_drawdown_above is not None and drawdown >= float(args.trigger_drawdown_above):
            triggered = True
        by_date[pd.Timestamp(row["date"]).normalize()] = {
            "triggered": bool(triggered),
            "cumret": cumret,
            "drawdown": drawdown,
        }
    return by_date


def transform_row(row, args, state):
    triggered = bool(state.get("triggered", False))
    out, details = shrink_target_row(
        row,
        triggered=triggered,
        base_target_frac=args.base_target_frac,
        risk_target_frac=args.risk_target_frac,
    )
    if details is None:
        return row
    out["state_target_transform"] = {
        **details,
        "prior_window_cumret": state.get("cumret"),
        "prior_window_drawdown": state.get("drawdown"),
    }
    return out


def main():
    args = parse_args()
    if args.trigger_cumret_below is None and args.trigger_drawdown_above is None:
        raise ValueError("At least one trigger condition must be provided")
    if not 0 < args.risk_target_frac <= args.base_target_frac:
        raise ValueError("risk-target-frac must be in (0, base-target-frac]")

    dates = load_alpha_dates(args.alpha_jsonl)
    if args.allow_forward:
        assert_alpha_dates_within_forward(dates, context="state target forward transform")
    else:
        assert_alpha_dates_within_research(dates, context="state target research transform")

    trigger_by_date = build_trigger_by_signal_date(args)
    output = Path(args.output_alpha)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    triggered_rows = 0
    def transformed_rows():
        nonlocal rows, triggered_rows
        for row in iter_alpha_rows(args.alpha_jsonl):
            date = pd.Timestamp(row["date"]).normalize()
            state = trigger_by_date.get(date, {"triggered": False, "cumret": np.nan, "drawdown": np.nan})
            out = transform_row(row, args, state)
            rows += 1
            triggered_rows += int(out["state_target_transform"]["triggered"])
            yield out

    write_alpha_rows(output, transformed_rows())

    print(
        json.dumps(
            {
                "output": str(output),
                "rows": rows,
                "triggered_rows": triggered_rows,
                "risk_target_frac": args.risk_target_frac,
                "trigger_window": args.trigger_window,
                "trigger_cumret_below": args.trigger_cumret_below,
                "trigger_drawdown_above": args.trigger_drawdown_above,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
