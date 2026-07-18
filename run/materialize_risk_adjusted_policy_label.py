"""Materialize a risk-adjusted utility target column into a ledger path dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--target-col", default="ledger_sweep_utility")
    parser.add_argument("--beta-cost", type=float, default=0.0)
    parser.add_argument("--specific-vol-cost", type=float, default=0.0)
    parser.add_argument("--industry-concentration-cost", type=float, default=0.0)
    parser.add_argument("--active-drawdown-cost", type=float, default=0.0)
    parser.add_argument("--momentum-plateau-cost", type=float, default=0.0)
    return parser.parse_args(argv)


def positive(series):
    return pd.to_numeric(series, errors="coerce").clip(lower=0.0).fillna(0.0)


def active_drawdown_input(frame):
    active = -pd.to_numeric(
        frame.get("diag_active_drawdown_trailing_return", 0.0),
        errors="coerce",
    ).fillna(0.0)
    risk = positive(frame.get("pair_risk_delta", 0.0))
    return active.clip(lower=0.0) * risk


def add_target(frame, args):
    out = frame.copy()
    weight = pd.to_numeric(out["baseline_weight"], errors="coerce").fillna(0.0)
    target = pd.to_numeric(out["ledger_path_utility"], errors="coerce").fillna(0.0)
    target = target - weight * float(args.beta_cost) * positive(out.get("pair_beta_delta", 0.0))
    target = target - weight * float(args.specific_vol_cost) * positive(out.get("pair_specific_vol_delta", 0.0))
    target = (
        target
        - weight
        * float(args.industry_concentration_cost)
        * positive(out.get("pair_industry_concentration_delta", 0.0))
    )
    target = target - weight * float(args.active_drawdown_cost) * active_drawdown_input(out)
    target = (
        target
        - weight
        * float(args.momentum_plateau_cost)
        * positive(out.get("pair_momentum_plateau_delta", 0.0))
    )
    out[args.target_col] = target.astype(float)
    return out


def main(argv=None):
    args = parse_args(argv)
    frame = pd.read_parquet(args.input_dataset)
    out = add_target(frame, args)
    output = Path(args.output_dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    summary = {
        "input_dataset": str(args.input_dataset),
        "output_dataset": str(output),
        "target_col": args.target_col,
        "rows": int(len(out)),
        "dates": int(pd.to_datetime(out["date"]).nunique()),
        "mean_target": float(out[args.target_col].mean()),
        "positive_rate": float((out[args.target_col] > 0).mean()),
        "params": vars(args),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
