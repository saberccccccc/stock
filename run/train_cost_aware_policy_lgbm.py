"""Train a cost-aware replacement policy that can choose not to trade.

The earlier policy always picked a replacement candidate.  This script trains a
regression score for the *net value of replacing* the baseline fill.  At apply
time, a positive-enough score is required; otherwise the original Alpha row is
left untouched.
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run.train_state_aware_policy_lgbm import build_matrix, infer_feature_columns


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--forward-dataset", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-cost", type=float, default=0.0025)
    parser.add_argument("--rank-cost", type=float, default=0.0040)
    parser.add_argument("--risk-cost", type=float, default=0.0040)
    parser.add_argument("--thresholds", default="0.0,0.0025,0.005,0.0075,0.01")
    parser.add_argument("--num-boost-round", type=int, default=140)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=7)
    parser.add_argument("--min-data-in-leaf", type=int, default=220)
    parser.add_argument("--lambda-l2", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260704)
    return parser.parse_args(argv)


def parse_thresholds(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def load_dataset(path):
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"empty dataset: {path}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


def add_cost_aware_target(frame, args):
    out = frame.copy()
    rank_penalty = args.rank_cost * pd.to_numeric(out["candidate_rank_pct"], errors="coerce").fillna(0.0)
    vol = pd.to_numeric(out.get("specific_vol_60d", 0.0), errors="coerce").fillna(0.0)
    beta = pd.to_numeric(out.get("beta_60d", 1.0), errors="coerce").fillna(1.0)
    crowd = pd.to_numeric(out.get("candidate_industry_top_share", 0.0), errors="coerce").fillna(0.0)
    risk_load = (
        (vol / 0.20).clip(lower=0.0, upper=3.0) * 0.40
        + ((beta - 1.0) / 1.0).clip(lower=0.0, upper=2.0) * 0.30
        + (crowd / 0.50).clip(lower=0.0, upper=2.0) * 0.30
    )
    risk_penalty = args.risk_cost * risk_load
    edge = pd.to_numeric(out["edge_vs_baseline"], errors="coerce")
    out["action_net_edge"] = edge - float(args.base_cost) - rank_penalty - risk_penalty
    out["action_cost_penalty"] = float(args.base_cost) + rank_penalty + risk_penalty
    return out


def eligible_labelled(frame):
    mask = frame["eligible"].eq(1) & frame["label_available"].eq(1) & frame["action_net_edge"].notna()
    out = frame.loc[mask].copy()
    if out.empty:
        raise ValueError("no eligible labelled rows")
    return out


def train_model(train_frame, feature_columns, args):
    x, fill_values = build_matrix(train_frame, feature_columns)
    y = pd.to_numeric(train_frame["action_net_edge"], errors="coerce").to_numpy(dtype=np.float32)
    dataset = lgb.Dataset(x, label=y, feature_name=feature_columns, free_raw_data=False)
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l2": args.lambda_l2,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": args.seed,
        "force_col_wise": True,
        "num_threads": 0,
    }
    model = lgb.train(params, dataset, num_boost_round=args.num_boost_round)
    return model, fill_values, params


def add_scores(frame, model, feature_columns, fill_values):
    scored = frame.copy()
    x, _ = build_matrix(scored, feature_columns, fill_values)
    scored["policy_score"] = model.predict(x)
    return scored


def _mean(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def evaluate_threshold(frame, model, feature_columns, fill_values, split_name, threshold):
    labelled = eligible_labelled(frame)
    scored = add_scores(labelled, model, feature_columns, fill_values)
    rows = []
    for date, group in scored.groupby("date", sort=True):
        baseline = group[group["baseline_fill"].eq(1)].sort_values("candidate_position").head(1)
        if baseline.empty:
            baseline = group.sort_values("candidate_position").head(1)
        chosen = group.sort_values(["policy_score", "candidate_position"], ascending=[False, True]).head(1)
        best_score = float(chosen["policy_score"].iloc[0]) if not chosen.empty else np.nan
        apply = int(np.isfinite(best_score) and best_score >= threshold)
        if apply:
            edge = float(chosen["action_net_edge"].iloc[0])
            raw_edge = float(chosen["edge_vs_baseline"].iloc[0])
            base_edge = float(chosen["exec_base_return"].iloc[0] - baseline["exec_base_return"].iloc[0])
        else:
            edge = 0.0
            raw_edge = 0.0
            base_edge = 0.0
        rows.append(
            {
                "split": split_name,
                "date": date,
                "threshold": threshold,
                "apply": apply,
                "best_score": best_score,
                "realized_net_edge": edge,
                "realized_raw_edge": raw_edge,
                "realized_base_edge": base_edge,
            }
        )
    daily = pd.DataFrame(rows)
    edge = daily["realized_net_edge"]
    std = float(edge.std(ddof=1)) if len(edge) > 1 else np.nan
    t_stat = float(edge.mean() / (std / math.sqrt(len(edge)))) if np.isfinite(std) and std > 1e-12 else np.nan
    summary = {
        "split": split_name,
        "threshold": threshold,
        "days": int(len(daily)),
        "apply_days": int(daily["apply"].sum()),
        "apply_rate": float(daily["apply"].mean()),
        "mean_net_edge": float(edge.mean()),
        "win_rate": float((edge > 0).mean()),
        "t_stat": t_stat,
        "mean_raw_edge": float(daily["realized_raw_edge"].mean()),
        "mean_base_edge": float(daily["realized_base_edge"].mean()),
        "mean_best_score": _mean(daily["best_score"]),
    }
    return daily, summary


def feature_importance_frame(model, feature_columns):
    return (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "gain": model.feature_importance(importance_type="gain"),
                "split": model.feature_importance(importance_type="split"),
            }
        )
        .sort_values(["gain", "split"], ascending=False)
        .reset_index(drop=True)
    )


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = add_cost_aware_target(load_dataset(args.train_dataset), args)
    test = add_cost_aware_target(load_dataset(args.test_dataset), args)
    forward = add_cost_aware_target(load_dataset(args.forward_dataset), args) if args.forward_dataset else None
    feature_columns = [
        column
        for column in infer_feature_columns(train)
        if not column.startswith("action_") and column != "policy_score"
    ]
    train_rows = eligible_labelled(train)
    model, fill_values, params = train_model(train_rows, feature_columns, args)
    thresholds = parse_thresholds(args.thresholds)
    summaries = []
    for split_name, frame in [("train_2024", train), ("test_2025", test), ("forward_2026", forward)]:
        if frame is None:
            continue
        for threshold in thresholds:
            daily, summary = evaluate_threshold(frame, model, feature_columns, fill_values, split_name, threshold)
            daily.to_csv(output / f"{split_name}_threshold_{threshold:g}_daily.csv", index=False)
            summaries.append(summary)
    importance = feature_importance_frame(model, feature_columns)
    importance.to_csv(output / "feature_importance.csv", index=False)
    payload = {
        "model": model,
        "feature_columns": feature_columns,
        "fill_values": fill_values,
        "params": params,
        "target": "action_net_edge",
        "cost_config": {
            "base_cost": args.base_cost,
            "rank_cost": args.rank_cost,
            "risk_cost": args.risk_cost,
        },
    }
    with (output / "model.pkl").open("wb") as fh:
        pickle.dump(payload, fh)
    meta = {
        "train_dataset": str(args.train_dataset),
        "test_dataset": str(args.test_dataset),
        "forward_dataset": str(args.forward_dataset) if args.forward_dataset else None,
        "feature_columns": feature_columns,
        "params": params,
        "summaries": summaries,
        "selection_protocol": "Use 2024 train reference and 2025 test for selection; 2026 forward is observational.",
    }
    (output / "training_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Cost-Aware No-Trade Policy Report",
        "",
        "The score estimates net replacement value. Apply only when score exceeds threshold.",
        "",
        "| split | threshold | days | apply_rate | mean_net_edge | win_rate | t_stat | mean_raw_edge |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['split']} | {item['threshold']:.4f} | {item['days']} | "
            f"{item['apply_rate']:.3f} | {item['mean_net_edge']:.6f} | "
            f"{item['win_rate']:.3f} | {item['t_stat']:.2f} | {item['mean_raw_edge']:.6f} |"
        )
    lines += ["", "## Top Features", "", "| feature | gain | split |", "|---|---:|---:|"]
    for _, row in importance.head(20).iterrows():
        lines.append(f"| {row['feature']} | {row['gain']:.2f} | {int(row['split'])} |")
    (output / "training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
