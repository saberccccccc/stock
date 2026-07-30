"""Research-alpha bridge to the project's existing realistic ledger runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from alpha.io import load_alpha_dates
from backtest.market_data_contract import ExecutionMarketDataContract
from core.research_protocol import RESEARCH_END_DATE, SELECTION_SPLITS, get_split_spec


def validate_experiment_alpha(alpha_path, split):
    if split not in SELECTION_SPLITS:
        raise ValueError(f"research experiment split is unsupported: {split}")
    dates = load_alpha_dates(alpha_path)
    spec = get_split_spec(split)
    start, end = spec.start, spec.end
    observed = [date for date in dates if start <= date <= end]
    if not observed:
        raise ValueError(f"alpha has no dates inside {split}")
    if max(observed) > RESEARCH_END_DATE:
        raise ValueError("research alpha exceeds frozen boundary")
    return {"signal_start": str(min(observed).date()), "signal_end": str(max(observed).date()), "days": len(observed)}


def build_ledger_command(
    *,
    alpha_path,
    experiment_id,
    split,
    output_dir,
    python=None,
    market_data: ExecutionMarketDataContract | None = None,
):
    start, end, _ = get_split_spec(split).command_dates()
    command = [
        python or sys.executable,
        "run/sweep_open_price_ledger_params.py",
        "--alpha-specs", f"{experiment_id}={Path(alpha_path).resolve()}",
        "--output-dir", str(output_dir),
        "--data-dir", "data/raw",
        "--stresses", "normal,lag1,cost2x,capacity_3pct",
        "--portfolio-values", "500000,1000000",
        "--target-fracs", "0.006",
        "--hold-fracs", "0.10",
        "--rebalance-bands", "0.20",
        "--max-new-names-list", "5",
        "--execution-constraint-mode", "realistic",
        "--start-date", start,
        "--end-date", end,
        "--max-data-date", end,
    ]
    command.extend((market_data or ExecutionMarketDataContract()).cli_args())
    return command
