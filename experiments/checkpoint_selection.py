"""Checkpoint selection helpers for epoch metrics JSONL files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_GATES = {
    "alpha": ("min", 0.07),
    "rawtopret_h5_top0p6": ("min", 0.0),
}

DEFAULT_RANK_METRICS = (
    ("rawtopstable_h5_top0p6", "max", 1.0),
    ("rawtopret_h5_top0p6", "max", 0.5),
    ("alpha", "max", 0.1),
)


@dataclass(frozen=True)
class SelectionRule:
    gates: dict[str, tuple[str, float]]
    rank_metrics: tuple[tuple[str, str, float], ...]


TOP_FIRST_RULE = SelectionRule(
    gates=dict(DEFAULT_GATES),
    rank_metrics=DEFAULT_RANK_METRICS,
)


def load_epoch_metrics(path: str | Path) -> pd.DataFrame:
    rows = []
    path = Path(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        row = {
            "epoch": int(record["epoch"]),
            "train_loss": float(record.get("train_loss", 0.0)),
            "learning_rate": float(record.get("learning_rate", 0.0)),
            "selection_metric": record.get("selection_metric", ""),
            "selection_score": float(record.get("selection_score", 0.0)),
            "checkpoint": record.get("checkpoint", ""),
        }
        row.update({f"loss_{k}": float(v) for k, v in record.get("train_components", {}).items()})
        row.update({k: float(v) for k, v in record.get("val_metrics", {}).items()})
        rows.append(row)
    if not rows:
        raise ValueError(f"No epoch metrics found: {path}")
    return pd.DataFrame(rows)


def _check_gate(frame: pd.DataFrame, metric: str, direction: str, threshold: float) -> pd.Series:
    if metric not in frame.columns:
        raise ValueError(f"Missing gate metric: {metric}")
    if direction == "min":
        return frame[metric] >= threshold
    if direction == "max":
        return frame[metric] <= threshold
    raise ValueError(f"Unknown gate direction for {metric}: {direction}")


def apply_gates(frame: pd.DataFrame, rule: SelectionRule = TOP_FIRST_RULE) -> pd.DataFrame:
    selected = frame.copy()
    selected["passed_gates"] = True
    failed = [[] for _ in range(len(selected))]
    for metric, (direction, threshold) in rule.gates.items():
        mask = _check_gate(selected, metric, direction, threshold)
        selected["passed_gates"] &= mask
        for idx, passed in enumerate(mask.tolist()):
            if not passed:
                failed[idx].append(f"{metric} {direction} {threshold:g}")
    selected["failed_gates"] = ["; ".join(items) for items in failed]
    return selected


def score_checkpoints(frame: pd.DataFrame, rule: SelectionRule = TOP_FIRST_RULE) -> pd.DataFrame:
    scored = apply_gates(frame, rule)
    scored["selection_rank_score"] = 0.0
    for metric, direction, weight in rule.rank_metrics:
        if metric not in scored.columns:
            raise ValueError(f"Missing rank metric: {metric}")
        sign = 1.0 if direction == "max" else -1.0
        scored["selection_rank_score"] += sign * float(weight) * scored[metric]
    scored["_gate_order"] = scored["passed_gates"].map({True: 0, False: 1})
    return scored.sort_values(
        ["_gate_order", "selection_rank_score", "epoch"],
        ascending=[True, False, True],
    ).drop(columns=["_gate_order"]).reset_index(drop=True)


def select_checkpoints(path: str | Path, rule: SelectionRule = TOP_FIRST_RULE) -> pd.DataFrame:
    return score_checkpoints(load_epoch_metrics(path), rule)


def parse_args():
    parser = argparse.ArgumentParser(description="Rank checkpoints from epoch_metrics.jsonl")
    parser.add_argument("metrics_jsonl")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    ranked = select_checkpoints(args.metrics_jsonl)
    preview_columns = [
        "epoch",
        "passed_gates",
        "selection_rank_score",
        "alpha",
        "rawtopret_h5_top0p6",
        "rawtopstable_h5_top0p6",
        "selection_metric",
        "selection_score",
        "checkpoint",
    ]
    preview = ranked[[column for column in preview_columns if column in ranked.columns]]
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(output, index=False)
    print(preview.head(args.top).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
