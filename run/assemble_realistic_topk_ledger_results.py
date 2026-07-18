"""Assemble realistic top-k open-ledger proposal results into a selector table."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)
os.chdir(ROOT)


REQUIRED_DATE_COLUMNS = ("signal_start", "signal_end", "backtest_start", "backtest_end")
DEFAULT_METRIC_WEIGHTS = {
    "ann": 0.01,
    "sharpe": 0.20,
    "mdd": -1.50,
    "avg_executed_turnover": -0.20,
    "total_cost": -25.0,
    "blocked_buy": -0.001,
    "adv_blocked": -0.001,
    "new_stock_buy_blocked": -0.001,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-proposal", default="baseline")
    parser.add_argument("--metric-weight", action="append", default=None, help="metric:weight override")
    return parser.parse_args(argv)


def parse_metric_weights(items):
    weights = dict(DEFAULT_METRIC_WEIGHTS)
    for item in items or []:
        if ":" not in item:
            raise ValueError(f"metric weight must be metric:weight, got {item!r}")
        metric, weight = item.split(":", 1)
        weights[metric] = float(weight)
    return weights


def require_columns(frame, columns, path):
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_manifest(root_dir):
    path = Path(root_dir) / "realistic_topk_ledger_dataset_v2_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary(path, proposal, stress, split_name):
    frame = pd.read_csv(path)
    require_columns(frame, REQUIRED_DATE_COLUMNS, path)
    for col in REQUIRED_DATE_COLUMNS:
        if frame[col].isna().any() or frame[col].astype(str).str.len().eq(0).any():
            raise ValueError(f"{path} has empty {col}")
    frame = frame.copy()
    frame.insert(0, "split", split_name)
    frame.insert(1, "proposal", proposal)
    frame.insert(2, "stress", stress)
    frame.insert(3, "summary_path", str(path))
    return frame


def discover_summaries(root_dir, split_name):
    root = Path(root_dir)
    base = root / "open_ledger"
    if not base.exists():
        raise FileNotFoundError(base)
    rows = []
    for proposal_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        proposal = proposal_dir.name
        for stress_dir in sorted(path for path in proposal_dir.iterdir() if path.is_dir()):
            summary = stress_dir / "open_ledger_summary.csv"
            if not summary.exists():
                continue
            rows.append(read_summary(summary, proposal, stress_dir.name, split_name))
    if not rows:
        raise ValueError(f"no open_ledger_summary.csv files found under {base}")
    return pd.concat(rows, axis=0, ignore_index=True)


def numeric(frame, column, default=0.0):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def compute_utility(frame, weights):
    total = np.zeros(len(frame), dtype=np.float64)
    for metric, weight in weights.items():
        total += numeric(frame, metric, 0.0).to_numpy(dtype=np.float64) * float(weight)
    return pd.Series(total, index=frame.index, dtype=float)


def add_baseline_deltas(frame, baseline_proposal):
    keys = ["split", "stress", "portfolio_value", "target_frac", "hold_frac"]
    require_columns(frame, keys + ["ledger_utility"], "assembled summary")
    baseline = frame[frame["proposal"].eq(baseline_proposal)][keys + ["ledger_utility"]].rename(
        columns={"ledger_utility": "baseline_ledger_utility"}
    )
    if baseline.empty:
        raise ValueError(f"baseline proposal {baseline_proposal!r} is missing")
    if baseline.duplicated(keys).any():
        raise ValueError(f"baseline proposal {baseline_proposal!r} has duplicate stress/account rows")
    out = frame.merge(baseline, on=keys, how="left")
    if out["baseline_ledger_utility"].isna().any():
        missing = out[out["baseline_ledger_utility"].isna()][keys].head(5).to_dict("records")
        raise ValueError(f"missing baseline rows for {missing}")
    out["ledger_utility_delta_vs_baseline"] = out["ledger_utility"] - out["baseline_ledger_utility"]
    return out


def summarize(frame):
    grouped = frame.groupby(["split", "proposal", "stress"], dropna=False)
    summary = grouped.agg(
        rows=("ledger_utility_delta_vs_baseline", "size"),
        mean_delta=("ledger_utility_delta_vs_baseline", "mean"),
        mean_ann=("ann", "mean"),
        mean_sharpe=("sharpe", "mean"),
        worst_mdd=("mdd", "max"),
        mean_turnover=("avg_executed_turnover", "mean"),
        mean_cost=("total_cost", "mean"),
    ).reset_index()
    return summary.sort_values(["split", "stress", "mean_delta"], ascending=[True, True, False])


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.root_dir)
    if str(manifest.get("split_name")) != str(args.split_name):
        raise ValueError(f"manifest split_name={manifest.get('split_name')} does not match {args.split_name}")
    weights = parse_metric_weights(args.metric_weight)
    frame = discover_summaries(args.root_dir, args.split_name)
    frame["ledger_utility"] = compute_utility(frame, weights)
    frame = add_baseline_deltas(frame, args.baseline_proposal)
    summary = summarize(frame)
    frame.to_parquet(output / "realistic_topk_ledger_results.parquet", index=False)
    frame.to_csv(output / "realistic_topk_ledger_results.csv", index=False)
    summary.to_csv(output / "realistic_topk_ledger_summary.csv", index=False)
    meta = {
        "root_dir": str(args.root_dir),
        "split_name": args.split_name,
        "rows": int(len(frame)),
        "proposals": sorted(frame["proposal"].astype(str).unique().tolist()),
        "stresses": sorted(frame["stress"].astype(str).unique().tolist()),
        "metric_weights": weights,
        "baseline_proposal": args.baseline_proposal,
        "required_date_columns": list(REQUIRED_DATE_COLUMNS),
        "selection_protocol": "Use 2024 validation and 2025 test only for selection. 2026 forward is observation-only.",
    }
    (output / "assemble_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
