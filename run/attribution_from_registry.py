"""Build a compact attribution coverage report from registry/attributions.csv.

Attribution is evidence about why a portfolio policy changed results.  It is
separate from the performance scorecard and never uses forward evidence to
make a promotion decision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SELECTION_SPLITS = ["val_2024", "test_2025"]
REQUIRED_STRESSES = ["normal", "lag1", "cost2x", "capacity_3pct"]
REQUIRED_CAPITALS = ["0050w", "0100w"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributions-csv", default="registry/attributions.csv")
    parser.add_argument("--output-dir", default="reports/official_registry_attribution_20260710")
    return parser.parse_args(argv)


def read_registry(path):
    full = ROOT / path
    if not full.exists():
        raise FileNotFoundError(full)
    return pd.read_csv(full, keep_default_na=False)


def coverage(frame):
    observed = {
        (str(row.split), str(row.scenario), str(row.capital))
        for row in frame.itertuples(index=False)
    }
    missing = [
        f"{split}/{stress}/{capital}"
        for split in SELECTION_SPLITS
        for stress in REQUIRED_STRESSES
        for capital in REQUIRED_CAPITALS
        if (split, stress, capital) not in observed
    ]
    return len(observed), missing


def main(argv=None):
    args = parse_args(argv)
    registry = read_registry(args.attributions_csv)
    rows = []
    coverage_rows = []
    for item in registry.itertuples(index=False):
        path_text = str(item.path).strip()
        if not path_text:
            coverage_rows.append({"candidate": item.candidate_id, "registered": False, "selection_missing_count": 16, "selection_missing": "not_registered"})
            continue
        path = ROOT / path_text
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"split", "scenario", "capital", "ret_sum_return_delta", "ret_t_return_delta", "repl_applied_days"}
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")
        frame = frame.copy()
        frame["candidate"] = item.candidate_id
        rows.append(frame)
        selection = frame.loc[frame["split"].isin(SELECTION_SPLITS)]
        observed_count, missing = coverage(selection)
        coverage_rows.append({"candidate": item.candidate_id, "registered": True, "selection_observed_rows": observed_count, "selection_missing_count": len(missing), "selection_missing": ";".join(missing)})

    long_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_dir / "attribution_long.csv", index=False, encoding="utf-8-sig")
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(out_dir / "attribution_coverage.csv", index=False, encoding="utf-8-sig")
    if long_df.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            long_df.groupby(["candidate", "split"], as_index=False)
            .agg(
                rows=("split", "size"),
                mean_return_delta=("ret_mean_return_delta", "mean"),
                mean_return_delta_t=("ret_t_return_delta", "mean"),
                mean_replacements=("repl_applied_days", "mean"),
                mean_pair_risk_delta=("repl_mean_pair_risk_delta", "mean"),
                mean_specific_vol_delta=("repl_mean_diff_specific_vol_60d", "mean"),
                mean_industry_top_share_delta=("repl_mean_diff_candidate_industry_top_share", "mean"),
            )
        )
    summary.to_csv(out_dir / "attribution_summary.csv", index=False, encoding="utf-8-sig")
    print({"output_dir": str(out_dir), "long_rows": int(len(long_df)), "registered_candidates": int(len(rows))}, flush=True)


if __name__ == "__main__":
    main()
