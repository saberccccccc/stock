"""Build the legacy diagnostic leaderboard from historical result files.

This module predates formal experiment manifests. Its output is never eligible
for registry promotion; use ``run/scorecard_from_registry.py`` for governed
comparisons.
"""

import argparse
from pathlib import Path

import pandas as pd

from experiments.registry import DEFAULT_CANDIDATES, resolve_source_path


LEADERBOARD_COLUMNS = [
    "candidate",
    "split",
    "stress",
    "signal_start",
    "signal_end",
    "backtest_start",
    "backtest_end",
    "capital",
    "ann",
    "sharpe",
    "mdd",
    "exec_to",
    "blocked_buy",
    "evidence_class",
    "formal_eligible",
]

REQUIRED_COVERAGE_COLUMNS = [
    "signal_start",
    "signal_end",
    "backtest_start",
    "backtest_end",
]


def capital_label(portfolio_value):
    value = float(portfolio_value)
    if value >= 10_000 and value % 10_000 == 0:
        return f"{int(value / 10_000)}w"
    return f"{value:g}"


def infer_split(row):
    days = int(float(row.get("n_return_days", 0)))
    return "val" if days <= 260 else "test"


def _num(row, key, default=0.0):
    value = row.get(key, default)
    if value == "" or pd.isna(value):
        return default
    return float(value)


def _filter_frame(frame, target_frac=None, hold_frac=None, portfolio_value=None):
    filters = {
        "target_frac": target_frac,
        "hold_frac": hold_frac,
        "portfolio_value": portfolio_value,
    }
    for column, value in filters.items():
        if value is None:
            continue
        if column not in frame.columns:
            raise ValueError(f"Cannot filter missing column: {column}")
        numeric = pd.to_numeric(frame[column])
        frame = frame[(numeric - float(value)).abs() < 1e-12]
    return frame


def require_coverage_columns(frame, path):
    missing = [column for column in REQUIRED_COVERAGE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required coverage columns: {', '.join(missing)}"
        )


def rows_from_summary(
    path,
    candidate,
    split=None,
    stress="normal",
    target_frac=None,
    hold_frac=None,
    portfolio_value=None,
):
    frame = pd.read_csv(path)
    require_coverage_columns(frame, path)
    frame = _filter_frame(
        frame,
        target_frac=target_frac,
        hold_frac=hold_frac,
        portfolio_value=portfolio_value,
    )
    records = []
    for _, row in frame.iterrows():
        file_split = row.get("split")
        row_split = split or (file_split if isinstance(file_split, str) and file_split else None) or infer_split(row)
        records.append(
            {
                "candidate": candidate,
                "split": row_split,
                "stress": stress,
                "signal_start": str(row["signal_start"]),
                "signal_end": str(row["signal_end"]),
                "backtest_start": str(row["backtest_start"]),
                "backtest_end": str(row["backtest_end"]),
                "capital": capital_label(row["portfolio_value"]),
                "ann": _num(row, "ann"),
                "sharpe": _num(row, "sharpe"),
                "mdd": _num(row, "mdd"),
                "exec_to": _num(row, "avg_executed_turnover", _num(row, "avg_turnover")),
                "blocked_buy": int(_num(row, "blocked_buy")),
                "evidence_class": "legacy_registered",
                "formal_eligible": False,
            }
        )
    return records


def build_leaderboard(candidates=DEFAULT_CANDIDATES, root="."):
    records = []
    missing = []
    for candidate in candidates:
        for source in candidate.result_sources:
            path = resolve_source_path(source, root)
            if not path.exists():
                missing.append({"candidate": candidate.name, "path": str(path)})
                continue
            records.extend(
                rows_from_summary(
                    path,
                    candidate=candidate.name,
                    split=source.split,
                    stress=source.stress,
                    target_frac=source.target_frac,
                    hold_frac=source.hold_frac,
                    portfolio_value=source.portfolio_value,
                )
            )
    frame = pd.DataFrame(records, columns=LEADERBOARD_COLUMNS)
    if not frame.empty:
        split_order = {"val": 0, "test": 1, "forward": 2}
        stress_order = {"normal": 0, "lag1": 1, "cost2x": 2, "capacity_3pct": 3}
        frame["_split_order"] = frame["split"].map(split_order).fillna(99)
        frame["_stress_order"] = frame["stress"].map(stress_order).fillna(99)
        frame["_capital_order"] = frame["capital"].str.rstrip("w").astype(float)
        frame = frame.sort_values(
            ["_split_order", "_stress_order", "_capital_order", "candidate"]
        ).drop(columns=["_split_order", "_stress_order", "_capital_order"])
    return frame.reset_index(drop=True), missing


def write_leaderboard(output_csv, candidates=DEFAULT_CANDIDATES, root="."):
    frame, missing = build_leaderboard(candidates=candidates, root=root)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame, missing


def parse_args():
    parser = argparse.ArgumentParser(description="Build registered candidate leaderboard")
    parser.add_argument("--output-csv", default="reports/candidate_leaderboard_20260617/candidate_leaderboard_from_registry.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    frame, missing = write_leaderboard(args.output_csv, root=args.root)
    if missing:
        for item in missing:
            print(f"missing {item['candidate']}: {item['path']}", flush=True)
        if args.fail_on_missing:
            raise SystemExit(2)
    print(f"wrote {args.output_csv} rows={len(frame)} missing={len(missing)}", flush=True)


if __name__ == "__main__":
    main()
