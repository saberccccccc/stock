"""Sweep risk-adjusted utility label variants for pairwise replacement policy.

This script does not use forward rows for selection.  It trains on the 2024
validation dataset, evaluates 2025 test for selection, and writes 2026 forward
only as observation.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run.train_pairwise_replacement_policy_lgbm import (
    evaluate,
    feature_columns,
    parse_thresholds,
    train,
)


TARGET_COL = "ledger_sweep_utility"


GRID = [
    {
        "name": "legacy_path",
        "beta": 0.0,
        "specific_vol": 0.0,
        "industry": 0.0,
        "active_drawdown": 0.0,
        "momentum_plateau": 0.0,
    },
    {
        "name": "riskadj_v1",
        "beta": 0.02,
        "specific_vol": 0.10,
        "industry": 0.02,
        "active_drawdown": 0.10,
        "momentum_plateau": 0.05,
    },
    {
        "name": "riskadj_half_beta_svol",
        "beta": 0.01,
        "specific_vol": 0.05,
        "industry": 0.02,
        "active_drawdown": 0.10,
        "momentum_plateau": 0.05,
    },
    {
        "name": "riskadj_svol_only_mild",
        "beta": 0.0,
        "specific_vol": 0.03,
        "industry": 0.0,
        "active_drawdown": 0.0,
        "momentum_plateau": 0.0,
    },
    {
        "name": "riskadj_beta_only_mild",
        "beta": 0.005,
        "specific_vol": 0.0,
        "industry": 0.0,
        "active_drawdown": 0.0,
        "momentum_plateau": 0.0,
    },
    {
        "name": "riskadj_industry_plateau",
        "beta": 0.0,
        "specific_vol": 0.0,
        "industry": 0.03,
        "active_drawdown": 0.0,
        "momentum_plateau": 0.08,
    },
    {
        "name": "riskadj_active_state",
        "beta": 0.0,
        "specific_vol": 0.0,
        "industry": 0.0,
        "active_drawdown": 0.30,
        "momentum_plateau": 0.0,
    },
    {
        "name": "riskadj_active_state_mild",
        "beta": 0.0,
        "specific_vol": 0.0,
        "industry": 0.0,
        "active_drawdown": 0.10,
        "momentum_plateau": 0.0,
    },
    {
        "name": "riskadj_active_state_strong",
        "beta": 0.0,
        "specific_vol": 0.0,
        "industry": 0.0,
        "active_drawdown": 0.60,
        "momentum_plateau": 0.0,
    },
    {
        "name": "riskadj_active_state_beta_tiny",
        "beta": 0.003,
        "specific_vol": 0.0,
        "industry": 0.0,
        "active_drawdown": 0.30,
        "momentum_plateau": 0.0,
    },
    {
        "name": "riskadj_active_state_svol_tiny",
        "beta": 0.0,
        "specific_vol": 0.01,
        "industry": 0.0,
        "active_drawdown": 0.30,
        "momentum_plateau": 0.0,
    },
    {
        "name": "riskadj_active_state_plateau",
        "beta": 0.0,
        "specific_vol": 0.0,
        "industry": 0.0,
        "active_drawdown": 0.30,
        "momentum_plateau": 0.03,
    },
    {
        "name": "riskadj_tiny_combo",
        "beta": 0.003,
        "specific_vol": 0.015,
        "industry": 0.01,
        "active_drawdown": 0.10,
        "momentum_plateau": 0.02,
    },
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--forward-dataset", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", choices=("binary", "lambdarank", "regression"), default="binary")
    parser.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--positive-threshold", type=float, default=0.0)
    parser.add_argument("--rank-label-bins", type=int, default=5)
    parser.add_argument("--num-boost-round", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=7)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--lambda-l2", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260704)
    return parser.parse_args(argv)


def load_dataset(path):
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


def positive(series):
    return pd.to_numeric(series, errors="coerce").clip(lower=0.0).fillna(0.0)


def active_drawdown_input(frame):
    active = -pd.to_numeric(
        frame.get("diag_active_drawdown_trailing_return", 0.0),
        errors="coerce",
    ).fillna(0.0)
    active = active.clip(lower=0.0)
    risk = positive(frame.get("pair_risk_delta", 0.0))
    return active * risk


def with_target(frame, cfg):
    out = frame.copy()
    weight = pd.to_numeric(out["baseline_weight"], errors="coerce").fillna(0.0)
    target = pd.to_numeric(out["ledger_path_utility"], errors="coerce").fillna(0.0)
    target = target - weight * float(cfg["beta"]) * positive(out.get("pair_beta_delta", 0.0))
    target = target - weight * float(cfg["specific_vol"]) * positive(out.get("pair_specific_vol_delta", 0.0))
    target = target - weight * float(cfg["industry"]) * positive(out.get("pair_industry_concentration_delta", 0.0))
    target = target - weight * float(cfg["active_drawdown"]) * active_drawdown_input(out)
    target = target - weight * float(cfg["momentum_plateau"]) * positive(out.get("pair_momentum_plateau_delta", 0.0))
    out[TARGET_COL] = target.astype(float)
    return out


def train_args(args):
    return SimpleNamespace(
        target_col=TARGET_COL,
        raw_edge_col="pair_path_raw_edge",
        objective=args.objective,
        positive_threshold=args.positive_threshold,
        rank_label_bins=args.rank_label_bins,
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        lambda_l2=args.lambda_l2,
        seed=args.seed,
    )


def best_selection(summary):
    sel = summary[summary["split"].eq("test_2025")].copy()
    if sel.empty:
        return None
    sel = sel.sort_values(
        ["mean_net_edge", "t_stat", "applied_win_rate", "apply_rate"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return sel.iloc[0].to_dict()


def write_markdown(result, path):
    ranked = result.sort_values(
        ["selection_mean_net_edge", "selection_t_stat", "selection_applied_win_rate"],
        ascending=[False, False, False],
        na_position="last",
    )
    lines = [
        "# Risk-Adjusted Label Sweep",
        "",
        "Selection uses 2025 test rows only after training on 2024 validation. 2026 forward is observation-only and is not used for ranking.",
        "",
        "| rank | config | threshold | test edge | test t | test applied win | test apply | forward edge | forward applied win |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['config']} | {row['selection_threshold']:.2f} | "
            f"{row['selection_mean_net_edge']:.6f} | {row['selection_t_stat']:.3f} | "
            f"{row['selection_applied_win_rate']:.2%} | {row['selection_apply_rate']:.2%} | "
            f"{row.get('forward_mean_net_edge', np.nan):.6f} | "
            f"{row.get('forward_applied_win_rate', np.nan):.2%} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_val = load_dataset(args.val_dataset)
    base_test = load_dataset(args.test_dataset)
    base_forward = load_dataset(args.forward_dataset) if args.forward_dataset else None
    rows = []
    detailed = []
    thresholds = parse_thresholds(args.thresholds)
    for cfg in GRID:
        val = with_target(base_val, cfg)
        test = with_target(base_test, cfg)
        forward = with_target(base_forward, cfg) if base_forward is not None else None
        targs = train_args(args)
        cols = feature_columns(val)
        model, fill_values, params = train(val, cols, targs)
        summaries = []
        for split, frame in [("train_2024", val), ("test_2025", test), ("forward_2026", forward)]:
            if frame is None:
                continue
            for threshold in thresholds:
                _, summary = evaluate(frame, model, cols, fill_values, split, threshold, targs)
                summary["config"] = cfg["name"]
                summaries.append(summary)
                detailed.append(summary)
        summary_df = pd.DataFrame(summaries)
        best = best_selection(summary_df)
        if best is None:
            continue
        fwd = summary_df[
            summary_df["split"].eq("forward_2026")
            & np.isclose(summary_df["threshold"].astype(float), float(best["threshold"]))
        ]
        fwd_row = fwd.iloc[0].to_dict() if not fwd.empty else {}
        record = {
            "config": cfg["name"],
            "objective": args.objective,
            "selection_threshold": float(best["threshold"]),
            "selection_mean_net_edge": float(best["mean_net_edge"]),
            "selection_t_stat": float(best["t_stat"]) if np.isfinite(best["t_stat"]) else np.nan,
            "selection_applied_win_rate": float(best["applied_win_rate"]),
            "selection_apply_rate": float(best["apply_rate"]),
            "forward_mean_net_edge": float(fwd_row.get("mean_net_edge", np.nan)),
            "forward_applied_win_rate": float(fwd_row.get("applied_win_rate", np.nan)),
            "forward_apply_rate": float(fwd_row.get("apply_rate", np.nan)),
        }
        record.update({f"w_{k}": v for k, v in cfg.items() if k != "name"})
        rows.append(record)
    result = pd.DataFrame(rows)
    detail = pd.DataFrame(detailed)
    result.to_csv(output / "risk_adjusted_label_sweep_summary.csv", index=False)
    detail.to_csv(output / "risk_adjusted_label_sweep_detail.csv", index=False)
    write_markdown(result, output / "risk_adjusted_label_sweep_summary.md")
    meta = {
        "val_dataset": str(args.val_dataset),
        "test_dataset": str(args.test_dataset),
        "forward_dataset": str(args.forward_dataset) if args.forward_dataset else None,
        "objective": args.objective,
        "thresholds": thresholds,
        "selection_protocol": "Train on 2024 validation; select by 2025 test. 2026 forward is observation-only.",
        "rows": len(result),
    }
    (output / "risk_adjusted_label_sweep_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
