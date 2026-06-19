"""Screen surge-then-stall rank demotions after the active training queue."""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.research_protocol import assert_alpha_rows_within_research
from run.backtest_retention_execution_constraints import (
    load_alpha_rows as load_backtest_rows,
    load_close_money,
    recompute_adv,
    run_constrained,
)
from run.backtest_temporal_retention import load_index_returns
from run.transform_alpha_for_execution import (
    load_alpha_rows,
    load_signal_returns,
    load_stall_signals,
    transform_rows,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Screen surge-then-stall execution rules")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--wait-pid-file", default=None)
    return parser.parse_args()


def wait_for_pid_file(pid_file):
    if not pid_file:
        return
    path = ROOT / pid_file
    if not path.exists():
        return
    pid = int(path.read_text(encoding="utf-8").strip())
    print(f"Waiting for training queue PID {pid}", flush=True)
    while True:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Get-Process -Id {pid} -ErrorAction SilentlyContinue",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            break
        time.sleep(60)
    print("Training queue finished; starting stall screen", flush=True)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_run_args(portfolio_value):
    return SimpleNamespace(
        max_weight=0.05,
        market_timing_mode="legacy",
        market_min_mult=0.20,
        market_max_mult=1.00,
        legacy_bear_mult=0.70,
        legacy_crash_mult=0.30,
        commission_rate=0.0001,
        stamp_tax_rate=0.0005,
        slippage_rate=0.0005,
        portfolio_value=float(portfolio_value),
        adv_participation_cap=0.05,
        min_adv_cny=3_000_000.0,
        limit_threshold=0.095,
        execution_lag=0,
        lot_size=100,
        min_commission_cny=5.0,
        rebalance_band=0.20,
        allow_forward=False,
    )


def main():
    args = parse_args()
    wait_for_pid_file(args.wait_pid_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = load_alpha_rows(args.input)
    if not source_rows:
        raise ValueError("Input Alpha file is empty")
    assert_alpha_rows_within_research(source_rows, context="stall execution screen")

    codes = {code for row in source_rows for code in row.get("codes", [])}
    start = source_rows[0]["date"]
    end = source_rows[-1]["date"]
    signal_returns = load_signal_returns(args.data_dir, codes, start, end)
    stall10 = load_stall_signals(
        args.data_dir, codes, start, end, surge_return=0.10
    )
    stall15 = load_stall_signals(
        args.data_dir, codes, start, end, surge_return=0.15
    )

    variants = [
        ("baseline", None, None, None),
        ("maxret095", 0.095, None, None),
        ("stall10", None, stall10, {"surge_return": 0.10}),
        ("stall15", None, stall15, {"surge_return": 0.15}),
        ("combo095_stall10", 0.095, stall10, {"surge_return": 0.10}),
        ("combo095_stall15", 0.095, stall15, {"surge_return": 0.15}),
    ]
    alpha_paths = {}
    for name, max_return, stalls, stall_config in variants:
        if name == "baseline":
            alpha_paths[name] = Path(args.input)
            continue
        transformed = transform_rows(
            source_rows,
            signal_returns,
            max_signal_return=max_return,
            stall_signals=stalls,
            stall_config=stall_config,
        )
        path = output_dir / f"{name}.jsonl"
        write_rows(path, transformed)
        alpha_paths[name] = path
        total = sum(row["execution_transform"]["demoted_count"] for row in transformed)
        stall_total = sum(
            row["execution_transform"]["stall_demoted_count"] for row in transformed
        )
        print(f"Generated {name}: demoted={total}, stall={stall_total}", flush=True)
        del transformed
        gc.collect()

    del source_rows, signal_returns, stall10, stall15
    gc.collect()

    close, money = load_close_money(args.data_dir, sorted(codes), 1000.0, 1000)
    adv = recompute_adv(money, 20)
    idx_close, idx_daily = load_index_returns(args.data_dir, "hs300_index.csv", close.index)

    summary = []
    for name, path in alpha_paths.items():
        rows = load_backtest_rows(path)
        for portfolio_value in (500_000.0, 1_000_000.0):
            run_args = make_run_args(portfolio_value)
            result, returns_df, diag_df = run_constrained(
                rows,
                close,
                adv,
                0.006,
                0.10,
                run_args,
                idx_close,
                idx_daily,
            )
            result.update(
                {
                    "variant": name,
                    "portfolio_value": portfolio_value,
                    "alpha_jsonl": str(path),
                }
            )
            summary.append(result)
            tag = f"{name}_{int(portfolio_value / 10000)}w"
            returns_df.to_csv(output_dir / f"returns_{tag}.csv", index=False)
            diag_df.to_csv(output_dir / f"diagnostics_{tag}.csv", index=False)
            print(
                f"{tag}: ann={result['ann']:.2f}% sharpe={result['sharpe']:.3f} "
                f"mdd={result['mdd'] * 100:.2f}%",
                flush=True,
            )
        del rows
        gc.collect()

    frame = pd.DataFrame(summary)
    frame.to_csv(output_dir / "stall_execution_summary.csv", index=False)
    report = [
        "# Surge-then-stall execution screen",
        "",
        "Validation period: 2024",
        "",
        "| variant | capital | ann | Sharpe | mdd | blocked buys |",
        "|---|---:|---:|---:|---:|---:|",
        "",
    ]
    for row in summary:
        report.insert(
            -1,
            f"| {row['variant']} | {row['portfolio_value']:,.0f} | "
            f"{row['ann']:.2f}% | {row['sharpe']:.3f} | "
            f"{row['mdd'] * 100:.2f}% | {row['blocked_buy']} |",
        )
    (output_dir / "stall_execution_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"Saved summary to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
