"""Confirm the 2024-locked candidate on the reserved historical period."""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.research_protocol import RESEARCH_END_DATE, assert_alpha_rows_within_research
from run.backtest_retention_execution_constraints import (
    load_alpha_rows as load_backtest_rows,
    load_close_money,
    recompute_adv,
    run_constrained,
)
from run.backtest_temporal_retention import load_index_returns
from run.backtest_v9_retention import compute_v9_alpha_rows, load_v9_samples_and_predictor
from run.transform_alpha_for_execution import (
    load_alpha_rows,
    load_signal_returns,
    transform_rows,
)


LOCKED_MODEL = "a4_e6"
LOCKED_CHECKPOINT = "checkpoints_loss_ablation_A4/epochs/epoch_006.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Confirm locked candidate")
    parser.add_argument(
        "--output-dir",
        default="locked_candidate_confirmation_20260614",
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=40)
    return parser.parse_args()


def write_alpha(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps({**row, "date": str(row["date"])}, ensure_ascii=False)
                + "\n"
            )


def generate_locked_alpha(output_dir, device, progress_every):
    raw_path = output_dir / LOCKED_MODEL / "alpha_raw_test.jsonl"
    if raw_path.exists() and raw_path.stat().st_size > 0:
        return raw_path
    args = SimpleNamespace(
        checkpoint=str(ROOT / LOCKED_CHECKPOINT),
        split="test",
        start_date="2025-01-01",
        end_date=str(RESEARCH_END_DATE.date()),
        output_dir=str(output_dir / LOCKED_MODEL),
        predictor_mode="average",
        window=3,
        target_fracs="0.006",
        hold_fracs="0.10",
        market_timing_mode="legacy",
        market_min_mult=0.20,
        market_max_mult=1.00,
        weight_mode="equal",
        max_weight=0.05,
        commission_rate=0.0001,
        stamp_tax_rate=0.0005,
        slippage_rate=0.0005,
        index_file="hs300_index.csv",
        device=device,
        progress_every=progress_every,
        limit_dates=None,
        ablate_fundamental=False,
    )
    _, samples, predictor = load_v9_samples_and_predictor(args)
    print(
        f"Generating locked confirmation Alpha: model={LOCKED_MODEL} "
        f"samples={len(samples)}",
        flush=True,
    )
    rows = compute_v9_alpha_rows(samples, predictor, progress_every)
    assert_alpha_rows_within_research(rows, context="locked historical confirmation")
    write_alpha(raw_path, rows)
    del rows, samples, predictor
    gc.collect()
    return raw_path


def apply_filter(raw_path, filtered_path, data_dir):
    if filtered_path.exists() and filtered_path.stat().st_size > 0:
        return
    rows = load_alpha_rows(raw_path)
    codes = {code for row in rows for code in row["codes"]}
    signal_returns = load_signal_returns(
        data_dir,
        codes,
        rows[0]["date"],
        rows[-1]["date"],
    )
    transformed = transform_rows(
        rows,
        signal_returns,
        max_signal_return=0.095,
    )
    write_alpha(filtered_path, transformed)
    print(
        f"Applied 9.5% filter: demoted="
        f"{sum(row['execution_transform']['demoted_count'] for row in transformed)}",
        flush=True,
    )


def find_frozen_alpha():
    matches = list(
        Path.home().joinpath("Documents").glob(
            "*/alpha_execution_screen_20260613/frozen_maxret095_test.jsonl"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"Frozen test Alpha matches: {matches}")
    return matches[0]


def make_run_args(portfolio_value, scenario):
    mult = scenario["cost_mult"]
    return SimpleNamespace(
        max_weight=0.05,
        market_timing_mode="legacy",
        market_min_mult=0.20,
        market_max_mult=1.00,
        legacy_bear_mult=0.70,
        legacy_crash_mult=0.30,
        commission_rate=0.0001 * mult,
        stamp_tax_rate=0.0005 * mult,
        slippage_rate=0.0005 * mult,
        portfolio_value=float(portfolio_value),
        adv_participation_cap=scenario["adv_cap"],
        min_adv_cny=3_000_000.0,
        limit_threshold=0.095,
        execution_lag=scenario["lag"],
        lot_size=100,
        min_commission_cny=5.0,
        rebalance_band=0.20,
        allow_forward=False,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = generate_locked_alpha(
        output_dir,
        args.device,
        args.progress_every,
    )
    filtered_path = output_dir / LOCKED_MODEL / "alpha_maxret095_test.jsonl"
    apply_filter(raw_path, filtered_path, args.data_dir)

    alpha_paths = {
        LOCKED_MODEL: filtered_path,
        "frozen_v9": find_frozen_alpha(),
    }
    rows_by_model = {
        model: load_backtest_rows(path)
        for model, path in alpha_paths.items()
    }
    all_codes = sorted(
        {
            code
            for rows in rows_by_model.values()
            for row in rows
            for code in row["codes"]
        }
    )
    print(f"Loading shared confirmation market data: codes={len(all_codes)}", flush=True)
    close, money = load_close_money(args.data_dir, all_codes, 1000.0, 1000)
    adv = recompute_adv(money, 20)
    idx_close, idx_daily = load_index_returns(args.data_dir, "hs300_index.csv", close.index)

    scenarios = {
        "base": {"adv_cap": 0.05, "cost_mult": 1.0, "lag": 0},
        "cap3": {"adv_cap": 0.03, "cost_mult": 1.0, "lag": 0},
        "cost2x": {"adv_cap": 0.05, "cost_mult": 2.0, "lag": 0},
        "lag1": {"adv_cap": 0.05, "cost_mult": 1.0, "lag": 1},
    }
    summary = []
    for model, rows in rows_by_model.items():
        for scenario_name, scenario in scenarios.items():
            for capital in (500_000.0, 1_000_000.0):
                result, returns_df, diag_df = run_constrained(
                    rows,
                    close,
                    adv,
                    0.006,
                    0.10,
                    make_run_args(capital, scenario),
                    idx_close,
                    idx_daily,
                )
                result.update(
                    {
                        "model": model,
                        "scenario": scenario_name,
                        "portfolio_value": capital,
                    }
                )
                summary.append(result)
                tag = f"{model}_{scenario_name}_{int(capital / 10000)}w"
                returns_df.to_csv(output_dir / f"returns_{tag}.csv", index=False)
                diag_df.to_csv(output_dir / f"diagnostics_{tag}.csv", index=False)
                print(
                    f"{tag}: ann={result['ann']:.2f}% "
                    f"sharpe={result['sharpe']:.3f} "
                    f"mdd={result['mdd'] * 100:.2f}%",
                    flush=True,
                )

    frame = pd.DataFrame(summary)
    frame.to_csv(output_dir / "confirmation_summary.csv", index=False)
    lines = [
        "# Locked Candidate Historical Confirmation",
        "",
        "The model was locked using 2024 results before this period was evaluated.",
        "",
        "| scenario | model | capital | annualized | Sharpe | max drawdown |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in frame.sort_values(
        ["scenario", "portfolio_value", "model"]
    ).to_dict("records"):
        lines.append(
            f"| {row['scenario']} | {row['model']} | "
            f"{row['portfolio_value']:,.0f} | {row['ann']:.2f}% | "
            f"{row['sharpe']:.3f} | {row['mdd'] * 100:.2f}% |"
        )
    (output_dir / "confirmation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Saved confirmation to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
