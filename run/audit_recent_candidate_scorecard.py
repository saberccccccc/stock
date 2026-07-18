"""Audit recent open-ledger candidate evidence and build a unified scorecard.

This script discovers ``open_ledger_summary.csv`` files under report roots and
checks whether core candidates have the expected split/stress/capital coverage.
It is intentionally read-only with respect to backtest results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOTS = [
    "reports/state_aware_policy_training_20260710_capped",
    "reports/state_aware_policy_training_20260705",
    "reports/state_aware_policy_training_20260704",
]

DEFAULT_CANDIDATES = [
    "ledger_path_v3_t0001_nolookahead",
    "nolookahead_stateobs",
    "rank_utility_v1_t05",
    "stateobs_activeglobal_volg001",
    "gate_v2_risk_nonworse",
    "ledger_path_v3_nolookahead_cond_pairrisk_volg001",
    "ledger_path_v3_capital_aware_hybrid_50_75no_100volg001",
]

EXPECTED_SPLITS = ["val_2024", "test_2025", "forward_2026"]
EXPECTED_SCENARIOS = ["normal", "lag1", "cost2x", "capacity_3pct"]
EXPECTED_CAPITALS = [500000.0, 1000000.0]

METRIC_COLUMNS = [
    "ann",
    "sharpe",
    "mdd",
    "active_ann",
    "information_ratio",
    "avg_executed_turnover",
    "total_cost",
    "blocked_buy",
    "blocked_sell",
    "adv_blocked",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=None, help="Report root to scan. Can repeat.")
    parser.add_argument("--candidate", action="append", default=None, help="Candidate name to audit. Can repeat.")
    parser.add_argument("--output-dir", default="reports/recent_candidate_scorecard_20260710")
    parser.add_argument("--include-all", action="store_true", help="Keep discovered non-core candidates too.")
    return parser.parse_args(argv)


def normalize_split(part: str) -> str | None:
    text = part.lower()
    if text in {"val_2024", "validation_2024", "validation"}:
        return "val_2024"
    if text in {"test_2025", "test_2025_20260518", "test"}:
        return "test_2025"
    if text in {"forward_2026", "forward_shadow", "forward"}:
        return "forward_2026"
    return None


def normalize_scenario(part: str) -> str | None:
    text = part.lower()
    if text in set(EXPECTED_SCENARIOS):
        return text
    return None


def infer_candidate_split_scenario(path: Path, root: Path):
    rel = path.relative_to(root)
    parts = list(rel.parts[:-1])
    split_idx = None
    split = None
    for idx, part in enumerate(parts):
        maybe = normalize_split(part)
        if maybe:
            split_idx = idx
            split = maybe
            break
    scenario = None
    for part in parts:
        maybe = normalize_scenario(part)
        if maybe:
            scenario = maybe
            break
    if split_idx is None:
        return None
    candidate_parts = parts[:split_idx]
    if not candidate_parts:
        return None
    candidate = normalize_candidate_name("/".join(candidate_parts))
    return candidate, split, scenario or "normal"


def normalize_candidate_name(name: str) -> str:
    text = name.replace("\\", "/")
    aliases = {
        "ledger_path_v3_nolookahead_riskguard_sweep/cond_pairrisk_volg001": "ledger_path_v3_nolookahead_cond_pairrisk_volg001",
        "stress_ledger_path_v3_t0001_nolookahead": "ledger_path_v3_t0001_nolookahead",
        "ledger_path_v3_t0001_nolookahead_open_ledger": "ledger_path_v3_t0001_nolookahead",
        "stress_ledger_path_v3_t0001_nolookahead": "ledger_path_v3_t0001_nolookahead",
        "nolookahead_stateobs_open_ledger": "nolookahead_stateobs",
        "rank_utility_v1_t05_open_ledger": "rank_utility_v1_t05",
        "stateobs_activeglobal_volg001_open_ledger": "stateobs_activeglobal_volg001",
        "gate_v2_risk_nonworse_open_ledger": "gate_v2_risk_nonworse",
    }
    if text in aliases:
        return aliases[text]
    for suffix in ("_open_ledger", "_inference_open_ledger"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def numeric_or_nan(row, column):
    try:
        return float(row.get(column, np.nan))
    except (TypeError, ValueError):
        return np.nan


def discover_rows(roots, core_candidates, include_all=False):
    records = []
    for root_raw in roots:
        root = Path(root_raw)
        if not root.exists():
            continue
        for path in root.rglob("open_ledger_summary.csv"):
            inferred = infer_candidate_split_scenario(path, root)
            if inferred is None:
                continue
            candidate, split, scenario = inferred
            if not include_all and candidate not in core_candidates:
                continue
            frame = pd.read_csv(path)
            for _, row in frame.iterrows():
                record = {
                    "candidate": candidate,
                    "split": split,
                    "scenario": scenario,
                    "portfolio_value": numeric_or_nan(row, "portfolio_value"),
                    "source": str(path),
                    "root": str(root),
                    "signal_start": str(row.get("signal_start", "")),
                    "signal_end": str(row.get("signal_end", "")),
                    "backtest_start": str(row.get("backtest_start", "")),
                    "backtest_end": str(row.get("backtest_end", "")),
                    "execution_constraint_mode": str(row.get("execution_constraint_mode", "")),
                    "limit_threshold": numeric_or_nan(row, "limit_threshold"),
                    "adv_participation_cap": numeric_or_nan(row, "adv_participation_cap"),
                    "min_adv_cny": numeric_or_nan(row, "min_adv_cny"),
                }
                for column in METRIC_COLUMNS:
                    record[column] = numeric_or_nan(row, column)
                records.append(record)
    return pd.DataFrame(records)


def summarize_candidates(long_df):
    if long_df.empty:
        return pd.DataFrame()
    selection = long_df[long_df["split"].isin(["val_2024", "test_2025"])].copy()
    grouped = []
    for candidate, group in selection.groupby("candidate", sort=False):
        grouped.append(
            {
                "candidate": candidate,
                "selection_rows": int(len(group)),
                "splits": ",".join(sorted(group["split"].unique())),
                "scenarios": ",".join(sorted(group["scenario"].unique())),
                "capital_count": int(group["portfolio_value"].nunique(dropna=True)),
                "mean_ann": float(group["ann"].mean()),
                "mean_sharpe": float(group["sharpe"].mean()),
                "min_sharpe": float(group["sharpe"].min()),
                "worst_mdd": float(group["mdd"].max()),
                "mean_active_ann": float(group["active_ann"].mean()),
                "mean_information_ratio": float(group["information_ratio"].mean()),
                "mean_turnover": float(group["avg_executed_turnover"].mean()),
                "mean_total_cost": float(group["total_cost"].mean()),
            }
        )
    out = pd.DataFrame(grouped)
    if out.empty:
        return out
    return out.sort_values(
        ["mean_information_ratio", "mean_sharpe", "mean_ann", "worst_mdd", "mean_turnover"],
        ascending=[False, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_coverage(long_df, candidates):
    keys = []
    available = set()
    if not long_df.empty:
        for row in long_df[["candidate", "split", "scenario", "portfolio_value"]].itertuples(index=False):
            available.add((row.candidate, row.split, row.scenario, float(row.portfolio_value)))
    for candidate in candidates:
        for split in EXPECTED_SPLITS:
            for scenario in EXPECTED_SCENARIOS:
                for capital in EXPECTED_CAPITALS:
                    present = (candidate, split, scenario, capital) in available
                    keys.append(
                        {
                            "candidate": candidate,
                            "split": split,
                            "scenario": scenario,
                            "portfolio_value": capital,
                            "status": "present" if present else "missing",
                        }
                    )
    return pd.DataFrame(keys)


def write_markdown(long_df, summary_df, coverage_df, output_path):
    lines = [
        "# Recent Candidate Evidence Audit",
        "",
        "Selection ranking uses only 2024 val + 2025 test. 2026 forward rows are listed as observation evidence only.",
        "",
        "## Candidate Selection Summary",
        "",
    ]
    if summary_df.empty:
        lines.append("No selection rows found.")
    else:
        lines.extend(
            [
                "| rank | candidate | rows | splits | scenarios | capital count | mean IR | mean active ann | mean ann | mean Sharpe | min Sharpe | worst MDD | turnover | cost |",
                "|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for idx, row in summary_df.iterrows():
            lines.append(
                f"| {idx + 1} | {row['candidate']} | {int(row['selection_rows'])} | "
                f"{row['splits']} | {row['scenarios']} | {int(row['capital_count'])} | "
                f"{row['mean_information_ratio']:.3f} | {row['mean_active_ann']:.2f}% | "
                f"{row['mean_ann']:.2f}% | {row['mean_sharpe']:.3f} | "
                f"{row['min_sharpe']:.3f} | {row['worst_mdd']:.2%} | "
                f"{row['mean_turnover']:.3f} | {row['mean_total_cost']:.4f} |"
            )
    lines.extend(["", "## Coverage Gaps", ""])
    gaps = coverage_df[coverage_df["status"].eq("missing")]
    if gaps.empty:
        lines.append("All expected candidate/split/scenario/capital cells are present.")
    else:
        compact = (
            gaps.groupby(["candidate", "split", "scenario"], sort=False)["portfolio_value"]
            .apply(lambda values: ",".join(f"{v/10000:.0f}w" for v in values))
            .reset_index(name="missing_capitals")
        )
        lines.extend(["| candidate | split | scenario | missing capitals |", "|---|---|---|---|"])
        for _, row in compact.iterrows():
            lines.append(
                f"| {row['candidate']} | {row['split']} | {row['scenario']} | {row['missing_capitals']} |"
            )
    lines.extend(["", "## Normal Rows Snapshot", ""])
    if "scenario" not in long_df.columns:
        normal = pd.DataFrame()
    else:
        normal = long_df[long_df["scenario"].eq("normal")].copy()
    if normal.empty:
        lines.append("No normal rows found.")
    else:
        normal = normal.sort_values(["candidate", "split", "portfolio_value"])
        lines.extend(
            [
                "| candidate | split | capital | signal | backtest | ann | Sharpe | MDD | IR | cost |",
                "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in normal.iterrows():
            lines.append(
                f"| {row['candidate']} | {row['split']} | {row['portfolio_value']/10000:.0f}w | "
                f"{row['signal_start']}~{row['signal_end']} | {row['backtest_start']}~{row['backtest_end']} | "
                f"{row['ann']:.2f}% | {row['sharpe']:.3f} | {row['mdd']:.2%} | "
                f"{row['information_ratio']:.3f} | {row['total_cost']:.4f} |"
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    roots = args.root or DEFAULT_ROOTS
    candidates = args.candidate or DEFAULT_CANDIDATES
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = discover_rows(roots, set(candidates), include_all=args.include_all)
    coverage_df = build_coverage(long_df, candidates)
    summary_df = summarize_candidates(long_df)

    long_df.to_csv(out_dir / "recent_candidate_scorecard_long.csv", index=False)
    coverage_df.to_csv(out_dir / "recent_candidate_coverage.csv", index=False)
    summary_df.to_csv(out_dir / "recent_candidate_selection_summary.csv", index=False)
    write_markdown(long_df, summary_df, coverage_df, out_dir / "recent_candidate_evidence_audit.md")

    print(
        {
            "output_dir": str(out_dir),
            "rows": int(len(long_df)),
            "missing_cells": int((coverage_df["status"] == "missing").sum()),
            "candidates": candidates,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
