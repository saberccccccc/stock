"""Summarize epoch metrics and correlations for registered loss experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize loss ablation metrics")
    parser.add_argument(
        "--config",
        default="configs/loss_ablation_20260613.json",
    )
    parser.add_argument(
        "--output",
        default="reports/loss_ablation_summary_20260613.csv",
    )
    return parser.parse_args()


def safe_corr(left, right):
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.std(left[mask]) <= 1e-12 or np.std(right[mask]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def main():
    args = parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    rows = []
    for experiment in config["experiments"]:
        path = ROOT / experiment["output_dir"] / "epochs" / "epoch_metrics.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            row = {
                "experiment": experiment["id"],
                "epoch": record["epoch"],
                "train_loss": record["train_loss"],
                **{
                    f"loss_{key}": value
                    for key, value in record.get("train_components", {}).items()
                },
                **record["val_metrics"],
            }
            rows.append(row)
    if not rows:
        raise ValueError("No completed experiment metrics found")

    frame = pd.DataFrame(rows)
    target = "rawtopstable_h5_top0p6"
    correlations = []
    for experiment, group in frame.groupby("experiment", sort=False):
        for column in [c for c in frame.columns if c.startswith("loss_")]:
            correlations.append({
                "experiment": experiment,
                "loss_component": column,
                "correlation_with_raw_top30_stability": safe_corr(
                    group[column].to_numpy(dtype=float),
                    group[target].to_numpy(dtype=float),
                ),
            })
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    pd.DataFrame(correlations).to_csv(
        output.with_name(f"{output.stem}_correlations.csv"),
        index=False,
    )
    print(frame.groupby("experiment")[target].agg(["max", "mean"]).to_string())


if __name__ == "__main__":
    main()
