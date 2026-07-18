"""Build an Active Portfolio Management scorecard from open-ledger summaries."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RANK_COLUMNS = [
    "information_ratio",
    "active_ann",
    "sharpe",
    "ann",
    "mdd",
    "avg_executed_turnover",
    "total_cost",
]

REQUIRED_COVERAGE_COLUMNS = [
    "signal_start",
    "signal_end",
    "backtest_start",
    "backtest_end",
]

DEFAULT_SELECTION_SPLITS = {
    "val",
    "validation",
    "validation_2024",
    "val_2024",
    "test",
    "test_2025",
    "test_2025_20260518",
}


def parse_input_spec(raw):
    """Parse candidate:split:scenario:path.

    Path is allowed to contain additional ':' characters, which keeps Windows
    drive letters usable.
    """
    parts = str(raw).split(":", 3)
    if len(parts) != 4:
        raise ValueError(
            "input must be formatted as candidate:split:scenario:path"
        )
    candidate, split, scenario, path = parts
    return {
        "candidate": candidate.strip(),
        "split": split.strip(),
        "scenario": scenario.strip(),
        "path": Path(path.strip()),
    }


def _numeric_or_nan(row, key):
    value = row.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def require_coverage_columns(frame, path):
    missing = [column for column in REQUIRED_COVERAGE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required coverage columns: {', '.join(missing)}"
        )


def load_scorecard_rows(specs):
    rows = []
    for spec in specs:
        path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        require_coverage_columns(frame, path)
        for _, row in frame.iterrows():
            record = {
                "candidate": spec["candidate"],
                "split": spec["split"],
                "scenario": spec["scenario"],
                "source": str(path),
                "signal_start": str(row.get("signal_start", "")),
                "signal_end": str(row.get("signal_end", "")),
                "backtest_start": str(row.get("backtest_start", "")),
                "backtest_end": str(row.get("backtest_end", "")),
                "portfolio_value": _numeric_or_nan(row, "portfolio_value"),
                "target_frac": _numeric_or_nan(row, "target_frac"),
                "hold_frac": _numeric_or_nan(row, "hold_frac"),
                "n_return_days": _numeric_or_nan(row, "n_return_days"),
                "ann": _numeric_or_nan(row, "ann"),
                "sharpe": _numeric_or_nan(row, "sharpe"),
                "mdd": _numeric_or_nan(row, "mdd"),
                "benchmark_ann": _numeric_or_nan(row, "benchmark_ann"),
                "active_ann": _numeric_or_nan(row, "active_ann"),
                "tracking_error": _numeric_or_nan(row, "tracking_error"),
                "information_ratio": _numeric_or_nan(row, "information_ratio"),
                "beta_to_benchmark": _numeric_or_nan(row, "beta_to_benchmark"),
                "avg_portfolio_beta_60d": _numeric_or_nan(row, "avg_portfolio_beta_60d"),
                "avg_portfolio_beta_per_gross_60d": _numeric_or_nan(row, "avg_portfolio_beta_per_gross_60d"),
                "avg_portfolio_specific_vol_60d": _numeric_or_nan(row, "avg_portfolio_specific_vol_60d"),
                "avg_executed_turnover": _numeric_or_nan(row, "avg_executed_turnover"),
                "total_cost": _numeric_or_nan(row, "total_cost"),
                "blocked_buy": _numeric_or_nan(row, "blocked_buy"),
                "blocked_sell": _numeric_or_nan(row, "blocked_sell"),
                "adv_blocked": _numeric_or_nan(row, "adv_blocked"),
            }
            if np.isnan(record["active_ann"]):
                record["active_ann"] = record["ann"]
            if np.isnan(record["information_ratio"]):
                record["information_ratio"] = record["sharpe"]
            rows.append(record)
    return pd.DataFrame(rows)


def rank_scorecard(frame):
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    ascending = {
        "information_ratio": False,
        "active_ann": False,
        "sharpe": False,
        "ann": False,
        "mdd": True,
        "avg_executed_turnover": True,
        "total_cost": True,
    }
    existing = [col for col in RANK_COLUMNS if col in ranked.columns]
    ranked = ranked.sort_values(
        existing,
        ascending=[ascending[col] for col in existing],
        na_position="last",
    )
    ranked["apm_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def split_selection_observation(frame, selection_splits=None):
    if frame.empty:
        return frame.copy(), frame.copy()
    allowed = set(selection_splits or DEFAULT_SELECTION_SPLITS)
    scoped = frame.copy()
    split_values = scoped["split"].astype(str).str.strip()
    is_selection = split_values.isin(allowed)
    selection = scoped[is_selection].copy()
    observation = scoped[~is_selection].copy()
    return selection, observation


def aggregate_candidate_selection(frame):
    if frame.empty:
        return pd.DataFrame()
    grouped = []
    for candidate, group in frame.groupby("candidate", sort=False):
        record = {
            "candidate": candidate,
            "selection_rows": int(len(group)),
            "splits": ",".join(sorted(group["split"].astype(str).unique())),
            "scenarios": ",".join(sorted(group["scenario"].astype(str).unique())),
            "capital_count": int(group["portfolio_value"].nunique(dropna=True)),
            "mean_information_ratio": float(group["information_ratio"].mean()),
            "mean_active_ann": float(group["active_ann"].mean()),
            "mean_ann": float(group["ann"].mean()),
            "mean_sharpe": float(group["sharpe"].mean()),
            "min_sharpe": float(group["sharpe"].min()),
            "worst_mdd": float(group["mdd"].max()),
            "mean_executed_turnover": float(group["avg_executed_turnover"].mean()),
            "mean_total_cost": float(group["total_cost"].mean()),
        }
        grouped.append(record)
    summary = pd.DataFrame(grouped)
    if summary.empty:
        return summary
    summary = summary.sort_values(
        [
            "mean_information_ratio",
            "mean_active_ann",
            "mean_sharpe",
            "mean_ann",
            "worst_mdd",
            "mean_executed_turnover",
            "mean_total_cost",
        ],
        ascending=[False, False, False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    summary["candidate_selection_rank"] = np.arange(1, len(summary) + 1)
    cols = ["candidate_selection_rank"] + [
        col for col in summary.columns if col != "candidate_selection_rank"
    ]
    return summary[cols]


def write_markdown(ranked, output_path, title="Active Management Scorecard", note=None):
    lines = [
        f"# {title}",
        "",
        note
        or (
            "Ranking priority: information ratio, active return, Sharpe, annualized "
            "return, drawdown, executed turnover, and total cost."
        ),
        "",
        "| rank | candidate | split | scenario | capital | signal | backtest | IR | active ann | ann | sharpe | mdd | realized beta | avg beta | turnover | cost |",
        "|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ranked.iterrows():
        capital = row.get("portfolio_value", np.nan)
        capital_text = "" if np.isnan(capital) else f"{capital / 10000:.0f}w"
        lines.append(
            f"| {int(row['apm_rank'])} | {row['candidate']} | {row['split']} | "
            f"{row['scenario']} | {capital_text} | "
            f"{row.get('signal_start', '')}~{row.get('signal_end', '')} | "
            f"{row.get('backtest_start', '')}~{row.get('backtest_end', '')} | "
            f"{row.get('information_ratio', np.nan):.3f} | "
            f"{row.get('active_ann', np.nan):.2f}% | "
            f"{row.get('ann', np.nan):.2f}% | "
            f"{row.get('sharpe', np.nan):.3f} | "
            f"{row.get('mdd', np.nan):.2%} | "
            f"{row.get('beta_to_benchmark', np.nan):.3f} | "
            f"{row.get('avg_portfolio_beta_60d', np.nan):.3f} | "
            f"{row.get('avg_executed_turnover', np.nan):.3f} | "
            f"{row.get('total_cost', np.nan):.4f} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_candidate_summary_markdown(summary, output_path):
    lines = [
        "# Candidate Selection Summary",
        "",
        "This candidate-level ranking is computed only from selection splits.",
        "Forward/observation rows are excluded from these ranks.",
        "",
        "| rank | candidate | rows | splits | scenarios | capital count | mean IR | mean active ann | mean ann | mean Sharpe | min Sharpe | worst MDD | turnover | cost |",
        "|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['candidate_selection_rank'])} | {row['candidate']} | "
            f"{int(row['selection_rows'])} | {row['splits']} | {row['scenarios']} | "
            f"{int(row['capital_count'])} | "
            f"{row.get('mean_information_ratio', np.nan):.3f} | "
            f"{row.get('mean_active_ann', np.nan):.2f}% | "
            f"{row.get('mean_ann', np.nan):.2f}% | "
            f"{row.get('mean_sharpe', np.nan):.3f} | "
            f"{row.get('min_sharpe', np.nan):.3f} | "
            f"{row.get('worst_mdd', np.nan):.2%} | "
            f"{row.get('mean_executed_turnover', np.nan):.3f} | "
            f"{row.get('mean_total_cost', np.nan):.4f} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="candidate:split:scenario:path to an open_ledger_summary.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--selection-split",
        action="append",
        default=None,
        help=(
            "Split allowed for model/rule selection. Defaults to common val/test "
            "names. Splits not listed are written as observation-only."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    specs = [parse_input_spec(raw) for raw in args.input]
    if not specs:
        raise SystemExit("at least one --input is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_df = load_scorecard_rows(specs)
    selection_df, observation_df = split_selection_observation(
        long_df, selection_splits=args.selection_split
    )
    ranked = rank_scorecard(selection_df)
    long_df.to_csv(output_dir / "apm_scorecard_long.csv", index=False)
    selection_df.to_csv(output_dir / "apm_scorecard_selection_input.csv", index=False)
    observation_df.to_csv(output_dir / "apm_scorecard_observation.csv", index=False)
    ranked.to_csv(output_dir / "apm_scorecard_ranked.csv", index=False)
    candidate_summary = aggregate_candidate_selection(selection_df)
    candidate_summary.to_csv(output_dir / "apm_candidate_selection_summary.csv", index=False)
    write_markdown(
        ranked,
        output_dir / "apm_scorecard.md",
        title="Active Management Selection Scorecard",
        note=(
            "Selection ranking excludes observation-only splits such as forward. "
            "Use 2024 validation and 2025 test evidence for model/rule selection; "
            "forward rows are written separately for monitoring."
        ),
    )
    write_candidate_summary_markdown(
        candidate_summary,
        output_dir / "apm_candidate_selection_summary.md",
    )
    print(f"wrote APM scorecard to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
