"""Generate and strictly validate selected V9 checkpoints on 2024 only."""

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

from core.research_protocol import assert_alpha_rows_within_research
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


CANDIDATES = {
    "m0_nomulti_e6": "checkpoints_loss_ablation_M0_nomulti/epochs/epoch_006.pt",
    "d001_e6": "checkpoints_loss_ablation_D001/epochs/epoch_006.pt",
    "d003_e6": "checkpoints_loss_ablation_D003/epochs/epoch_006.pt",
    "d005_e6": "checkpoints_loss_ablation_D005/epochs/epoch_006.pt",
    "t001_e5": "checkpoints_loss_ablation_T001/epochs/epoch_005.pt",
    "d003_t001_e6": "checkpoints_loss_ablation_D003_T001/epochs/epoch_006.pt",
    "m1_nomulti_topfocus_w005_e6": (
        "checkpoints_loss_ablation_M1_nomulti_topfocus_w005/epochs/epoch_006.pt"
    ),
    "a0_e6": "checkpoints_loss_ablation_A0/epochs/epoch_006.pt",
    "a1_e6": "checkpoints_loss_ablation_A1/epochs/epoch_006.pt",
    "a2_e6": "checkpoints_loss_ablation_A2/epochs/epoch_006.pt",
    "a3_e6": "checkpoints_loss_ablation_A3/epochs/epoch_006.pt",
    "a5_e6": "checkpoints_loss_ablation_A5/epochs/epoch_006.pt",
    "a0_e9": "checkpoints_loss_ablation_A0_low_lr_e10/epochs/epoch_009.pt",
    "a4_e6": "checkpoints_loss_ablation_A4/epochs/epoch_006.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate selected V9 candidates")
    parser.add_argument(
        "--output-dir",
        default="candidate_model_validation_20260614",
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=80)
    parser.add_argument(
        "--alpha-only",
        action="store_true",
        help="Generate raw and maxret095 Alpha without running the legacy close backtest.",
    )
    parser.add_argument(
        "--only",
        choices=sorted(CANDIDATES),
        default=None,
        help="Validate only one registered candidate and omit the frozen reference.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Validate an arbitrary checkpoint path instead of a registered candidate.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Model name to use with --checkpoint.",
    )
    parser.add_argument(
        "--frozen-alpha",
        default="backtest_results_test_plan_v9_avgw3_val/v9_daily_alpha_top_order.jsonl",
    )
    return parser.parse_args()


def write_alpha(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {**row, "date": str(row["date"])}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def generation_args(checkpoint, output_dir, device, progress_every):
    return SimpleNamespace(
        checkpoint=str(checkpoint),
        split="val",
        start_date="2024-01-01",
        # The shared date filter uses a right-open interval.
        end_date="2025-01-01",
        output_dir=str(output_dir),
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


def generate_candidate_alpha(name, checkpoint, output_dir, device, progress_every):
    raw_path = output_dir / name / "alpha_raw.jsonl"
    if raw_path.exists() and raw_path.stat().st_size > 0:
        print(f"SKIP alpha {name}: {raw_path}", flush=True)
        return raw_path

    args = generation_args(checkpoint, output_dir / name, device, progress_every)
    _, samples, predictor = load_v9_samples_and_predictor(args)
    print(f"Generating {name}: samples={len(samples)} checkpoint={checkpoint}", flush=True)
    rows = compute_v9_alpha_rows(samples, predictor, progress_every)
    assert_alpha_rows_within_research(rows, context=f"{name} candidate Alpha")
    write_alpha(raw_path, rows)
    del rows, samples, predictor
    gc.collect()
    return raw_path


def apply_chase_filter(raw_path, filtered_path, data_dir):
    if filtered_path.exists() and filtered_path.stat().st_size > 0:
        print(f"SKIP transform: {filtered_path}", flush=True)
        return
    rows = load_alpha_rows(raw_path)
    codes = {code for row in rows for code in row.get("codes", [])}
    returns = load_signal_returns(data_dir, codes, rows[0]["date"], rows[-1]["date"])
    transformed = transform_rows(rows, returns, max_signal_return=0.095)
    write_alpha(filtered_path, transformed)
    demoted = sum(row["execution_transform"]["demoted_count"] for row in transformed)
    print(f"Filtered {raw_path}: demoted={demoted}", flush=True)
    del rows, returns, transformed
    gc.collect()


def make_run_args(portfolio_value, scenario):
    cost_mult = scenario["cost_mult"]
    return SimpleNamespace(
        max_weight=0.05,
        market_timing_mode="legacy",
        market_min_mult=0.20,
        market_max_mult=1.00,
        legacy_bear_mult=0.70,
        legacy_crash_mult=0.30,
        commission_rate=0.0001 * cost_mult,
        stamp_tax_rate=0.0005 * cost_mult,
        slippage_rate=0.0005 * cost_mult,
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

    alpha_paths = {}
    if args.checkpoint:
        if not args.name:
            raise ValueError("--name is required when --checkpoint is provided")
        candidates = {args.name: args.checkpoint}
    else:
        candidates = (
            {args.only: CANDIDATES[args.only]}
            if args.only
            else CANDIDATES
        )
    for name, checkpoint in candidates.items():
        raw_path = generate_candidate_alpha(
            name,
            ROOT / checkpoint,
            output_dir,
            args.device,
            args.progress_every,
        )
        filtered_path = output_dir / name / "alpha_maxret095.jsonl"
        apply_chase_filter(raw_path, filtered_path, args.data_dir)
        alpha_paths[(name, "raw")] = raw_path
        alpha_paths[(name, "maxret095")] = filtered_path

    if args.alpha_only:
        print(f"Saved candidate Alpha to {output_dir}", flush=True)
        return

    frozen_path = ROOT / args.frozen_alpha
    if not args.only and frozen_path.exists():
        frozen_filtered = output_dir / "frozen_v9" / "alpha_maxret095.jsonl"
        apply_chase_filter(frozen_path, frozen_filtered, args.data_dir)
        alpha_paths[("frozen_v9", "raw")] = frozen_path
        alpha_paths[("frozen_v9", "maxret095")] = frozen_filtered

    all_rows = {
        key: load_backtest_rows(path)
        for key, path in alpha_paths.items()
    }
    for key, rows in all_rows.items():
        assert_alpha_rows_within_research(rows, context=f"{key} constrained validation")

    all_codes = sorted(
        {
            code
            for rows in all_rows.values()
            for row in rows
            for code in row["codes"]
        }
    )
    print(f"Loading shared market data: codes={len(all_codes)}", flush=True)
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
    for (model, transform), rows in all_rows.items():
        for scenario_name, scenario in scenarios.items():
            for portfolio_value in (500_000.0, 1_000_000.0):
                run_args = make_run_args(portfolio_value, scenario)
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
                        "model": model,
                        "transform": transform,
                        "scenario": scenario_name,
                        "portfolio_value": portfolio_value,
                        "checkpoint": CANDIDATES.get(model, "operational reference"),
                    }
                )
                summary.append(result)
                tag = f"{model}_{transform}_{scenario_name}_{int(portfolio_value / 10000)}w"
                returns_df.to_csv(output_dir / f"returns_{tag}.csv", index=False)
                diag_df.to_csv(output_dir / f"diagnostics_{tag}.csv", index=False)
                print(
                    f"{tag}: ann={result['ann']:.2f}% sharpe={result['sharpe']:.3f} "
                    f"mdd={result['mdd'] * 100:.2f}%",
                    flush=True,
                )

    frame = pd.DataFrame(summary)
    frame.to_csv(output_dir / "candidate_validation_summary.csv", index=False)
    filtered = frame[frame["transform"] == "maxret095"].copy()
    filtered = filtered.sort_values(
        ["scenario", "portfolio_value", "sharpe"],
        ascending=[True, True, False],
    )
    lines = [
        "# Candidate Model Validation",
        "",
        "Validation period: 2024 only.",
        "",
        "## 9.5% Filtered Results",
        "",
        "| scenario | model | capital | annualized | Sharpe | max drawdown | blocked buys | unfilled turnover |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in filtered.to_dict("records"):
        lines.append(
            f"| {row['scenario']} | {row['model']} | {row['portfolio_value']:,.0f} | "
            f"{row['ann']:.2f}% | {row['sharpe']:.3f} | {row['mdd'] * 100:.2f}% | "
            f"{row.get('blocked_buy', 0)} | {row.get('avg_unfilled_turnover', 0):.3f} |"
        )
    (output_dir / "candidate_validation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Saved candidate validation to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
