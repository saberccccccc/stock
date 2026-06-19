"""Summarize selected open-ledger candidates into CSV/Markdown."""

import argparse
from pathlib import Path

import pandas as pd


CANDIDATES = [
    {
        "name": "main_candidate",
        "role": "official_baseline",
        "normal": {
            "val": "v9_avgw3_open_ledger_20260617/main_candidate_single_diag/val/open_ledger_summary.csv",
            "test": "v9_avgw3_open_ledger_20260617/main_candidate_single_diag/test/open_ledger_summary.csv",
        },
        "lag1": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_main_candidate_lag1/sweep_summary.csv",
        },
        "cost2x": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_main_candidate_cost2x/sweep_summary.csv",
        },
    },
    {
        "name": "negfilter_r030_100_drop3",
        "role": "first_attack_candidate",
        "normal": {
            "val": "v9_avgw3_open_ledger_20260617/negfilter_drop3_single_diag/val/open_ledger_summary.csv",
            "test": "v9_avgw3_open_ledger_20260617/negfilter_drop3_single_diag/test/open_ledger_summary.csv",
        },
        "lag1": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_negfilter_r030_100_drop3_lag1/sweep_summary.csv",
        },
        "cost2x": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_negfilter_r030_100_drop3_cost2x/sweep_summary.csv",
        },
    },
    {
        "name": "edge_r030_100",
        "role": "first_stability_candidate",
        "normal": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_edge_r030_100_w095/sweep_summary.csv",
        },
        "lag1": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_edge_r030_100_w095_lag1/sweep_summary.csv",
        },
        "cost2x": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_edge_r030_100_w095_cost2x/sweep_summary.csv",
        },
    },
    {
        "name": "market_switch",
        "role": "conservative_watch_candidate",
        "normal": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_market_switch_base_bear_edge_normal_r030_100/sweep_summary.csv",
        },
        "lag1": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_market_switch_base_bear_edge_normal_r030_100_lag1/sweep_summary.csv",
        },
        "cost2x": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_market_switch_base_bear_edge_normal_r030_100_cost2x/sweep_summary.csv",
        },
    },
    {
        "name": "edge_r030_120",
        "role": "stability_watch_candidate",
        "normal": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_edge_r030_120_w095/sweep_summary.csv",
        },
        "lag1": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_edge_r030_120_w095_lag1/sweep_summary.csv",
        },
        "cost2x": {
            "val_test": "v9_avgw3_open_ledger_20260617/sweep_edge_r030_120_w095_cost2x/sweep_summary.csv",
        },
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/open_ledger_candidate_summary_20260617")
    return parser.parse_args()


def load_rows(path, split_hint=None):
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        split = split_hint
        if split is None:
            explicit_split = str(row.get("split", "")).strip().lower()
            if explicit_split in {"val", "test"}:
                split = explicit_split
            else:
                n = int(row.get("n_return_days", 0))
                if n <= 0:
                    raise ValueError(f"Cannot determine val/test split for row in {path}")
                split = "val" if n <= 260 else "test"
        rows.append(
            {
                "split": split,
                "portfolio_value": float(row["portfolio_value"]),
                "ann": float(row["ann"]),
                "sharpe": float(row["sharpe"]),
                "mdd": float(row["mdd"]),
                "avg_turnover": float(row.get("avg_turnover", row.get("turn_mean", 0.0))),
                "blocked_buy": float(row.get("blocked_buy", 0.0)),
                "blocked_sell": float(row.get("blocked_sell", 0.0)),
            }
        )
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for cand in CANDIDATES:
        for scenario in ["normal", "lag1", "cost2x"]:
            for key, path in cand.get(scenario, {}).items():
                split_hint = key if key in ("val", "test") else None
                for row in load_rows(path, split_hint):
                    row.update({"candidate": cand["name"], "role": cand["role"], "scenario": scenario})
                    records.append(row)
    df = pd.DataFrame(records)
    df = df.sort_values(["scenario", "split", "portfolio_value", "candidate"])
    df.to_csv(out_dir / "candidate_summary_long.csv", index=False)

    normal = df[df["scenario"] == "normal"].copy()
    pivot = normal.pivot_table(
        index=["candidate", "role", "portfolio_value"],
        columns="split",
        values=["ann", "sharpe", "mdd"],
        aggfunc="first",
    )
    pivot.columns = ["_".join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()
    pivot.to_csv(out_dir / "candidate_summary_normal_wide.csv", index=False)

    md = ["# Open-Ledger Candidate Summary 2026-06-17", ""]
    md.append("## Normal Scenario")
    md.append("")
    md.append("| candidate | role | capital | val ann | val sharpe | val mdd | test ann | test sharpe | test mdd |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in pivot.iterrows():
        md.append(
            f"| {row['candidate']} | {row['role']} | {row['portfolio_value']/10000:.0f}w | "
            f"{row.get('ann_val', float('nan')):.2f}% | {row.get('sharpe_val', float('nan')):.3f} | {row.get('mdd_val', float('nan')):.2%} | "
            f"{row.get('ann_test', float('nan')):.2f}% | {row.get('sharpe_test', float('nan')):.3f} | {row.get('mdd_test', float('nan')):.2%} |"
        )
    md.append("")
    md.append("## Current Decision")
    md.append("")
    md.append("- official baseline: `main_candidate`")
    md.append("- first attack candidate: `negfilter_r030_100_drop3`")
    md.append("- first stability candidate: `edge_r030_100`")
    md.append("- use `negfilter_r030_100_drop3` for forward/live observation, not immediate replacement.")
    (out_dir / "candidate_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
