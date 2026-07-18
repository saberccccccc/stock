"""Train a pairwise replacement policy from candidate-vs-baseline rows."""

import argparse
import json
import math
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


EXCLUDE = {
    "split",
    "date",
    "code",
    "baseline_code",
    "label_available",
    "eligible",
    "pair_net_edge",
    "pair_raw_edge",
    "pair_base_return_edge",
    "pair_path_utility",
    "pair_path_raw_edge",
    "pair_path_return",
    "base_path_return",
    "pair_downside_delta",
    "pair_turnover_delta",
    "pair_quick_fade_delta",
    "pair_rank_delta",
    "ledger_path_utility",
    "ledger_risk_adjusted_utility",
    "ledger_weighted_raw_edge",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--forward-dataset", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-col", default="pair_net_edge")
    parser.add_argument("--raw-edge-col", default=None)
    parser.add_argument("--objective", choices=("regression", "binary", "lambdarank"), default="regression")
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help="For --objective binary, label rows with target-col above this value as positive.",
    )
    parser.add_argument("--thresholds", default="0,0.0025,0.005,0.0075,0.01,0.015,0.02")
    parser.add_argument(
        "--rank-label-bins",
        type=int,
        default=5,
        help="For lambdarank, convert target-col to per-date integer relevance labels in [0, bins-1].",
    )
    parser.add_argument("--num-boost-round", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=7)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--lambda-l2", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260704)
    return parser.parse_args(argv)


def parse_thresholds(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def load_dataset(path):
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


def feature_columns(frame):
    cols = []
    for col in frame.columns:
        if col in EXCLUDE:
            continue
        if col.startswith("ledger_"):
            continue
        if col.startswith("pair_") and col != "pair_risk_delta":
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def build_matrix(frame, cols, fill_values=None):
    x = frame[cols].copy()
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if fill_values is None:
        fill_values = {}
        for col in cols:
            med = x[col].median()
            fill_values[col] = float(med) if np.isfinite(med) else 0.0
    x = x.fillna(fill_values)
    return x.to_numpy(dtype=np.float32), fill_values


def target_values(frame, args):
    if args.target_col not in frame.columns:
        raise ValueError(f"target column not found: {args.target_col}")
    target = pd.to_numeric(frame[args.target_col], errors="coerce").to_numpy(dtype=np.float32)
    if args.objective == "binary":
        return (target > float(args.positive_threshold)).astype(np.float32)
    if args.objective == "lambdarank":
        return rank_relevance_labels(frame, args)
    return target


def rank_relevance_labels(frame, args):
    if args.target_col not in frame.columns:
        raise ValueError(f"target column not found: {args.target_col}")
    bins = int(getattr(args, "rank_label_bins", 5))
    if bins < 2:
        raise ValueError("--rank-label-bins must be >= 2")
    labels = pd.Series(0, index=frame.index, dtype=np.int32)
    target = pd.to_numeric(frame[args.target_col], errors="coerce")
    for _, idx in frame.groupby("date", sort=False).groups.items():
        values = target.loc[idx]
        valid = values.notna()
        if int(valid.sum()) <= 1:
            labels.loc[idx] = 0
            continue
        ranks = values.loc[valid].rank(method="first", pct=True)
        labels.loc[ranks.index] = np.floor(ranks * bins).clip(0, bins - 1).astype(np.int32)
    return labels.to_numpy(dtype=np.int32)


def group_sizes_by_date(frame):
    return frame.groupby("date", sort=False).size().astype(int).tolist()


def train(train_frame, cols, args):
    train_frame = train_frame.sort_values("date").reset_index(drop=True)
    x, fill_values = build_matrix(train_frame, cols)
    y = target_values(train_frame, args)
    group = group_sizes_by_date(train_frame) if args.objective == "lambdarank" else None
    ds = lgb.Dataset(x, label=y, group=group, feature_name=cols, free_raw_data=False)
    metric = "ndcg" if args.objective == "lambdarank" else ("binary_logloss" if args.objective == "binary" else "l2")
    params = {
        "objective": args.objective,
        "metric": metric,
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
    if args.objective == "binary":
        positives = float(y.sum())
        negatives = float(len(y) - positives)
        if positives > 0 and negatives > 0:
            params["scale_pos_weight"] = negatives / positives
    if args.objective == "lambdarank":
        params["label_gain"] = list(range(int(args.rank_label_bins)))
    model = lgb.train(params, ds, num_boost_round=args.num_boost_round)
    return model, fill_values, params


def score(frame, model, cols, fill_values):
    out = frame.copy()
    x, _ = build_matrix(out, cols, fill_values)
    out["policy_score"] = model.predict(x)
    return out


def evaluate(frame, model, cols, fill_values, split, threshold, args):
    scored = score(frame, model, cols, fill_values)
    rows = []
    for date, group in scored.groupby("date", sort=True):
        chosen = group.sort_values(["policy_score", "candidate_position"], ascending=[False, True]).head(1)
        best = float(chosen["policy_score"].iloc[0])
        apply = int(np.isfinite(best) and best >= threshold)
        net = float(chosen[args.target_col].iloc[0]) if apply else 0.0
        positive_label = (
            int(float(chosen[args.target_col].iloc[0]) > float(args.positive_threshold))
            if apply and args.target_col in chosen.columns
            else 0
        )
        raw_col = args.raw_edge_col or ("pair_path_raw_edge" if "pair_path_raw_edge" in chosen.columns else "pair_raw_edge")
        raw = float(chosen[raw_col].iloc[0]) if apply and raw_col in chosen.columns else 0.0
        rows.append(
            {
                "split": split,
                "date": date,
                "threshold": threshold,
                "apply": apply,
                "best_score": best,
                "net_edge": net,
                "raw_edge": raw,
                "positive_label": positive_label,
            }
        )
    daily = pd.DataFrame(rows)
    edge = daily["net_edge"]
    std = float(edge.std(ddof=1)) if len(edge) > 1 else np.nan
    t_stat = float(edge.mean() / (std / math.sqrt(len(edge)))) if np.isfinite(std) and std > 1e-12 else np.nan
    return daily, {
        "split": split,
        "threshold": threshold,
        "days": int(len(daily)),
        "apply_days": int(daily["apply"].sum()),
        "apply_rate": float(daily["apply"].mean()),
        "mean_net_edge": float(daily["net_edge"].mean()),
        "mean_raw_edge": float(daily["raw_edge"].mean()),
        "win_rate": float((daily["net_edge"] > 0).mean()),
        "applied_win_rate": float((daily.loc[daily["apply"].eq(1), "net_edge"] > 0).mean())
        if int(daily["apply"].sum()) > 0
        else 0.0,
        "t_stat": t_stat,
        "mean_best_score": float(daily["best_score"].mean()),
    }


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_frame = load_dataset(args.train_dataset)
    test_frame = load_dataset(args.test_dataset)
    forward_frame = load_dataset(args.forward_dataset) if args.forward_dataset else None
    cols = feature_columns(train_frame)
    model, fill_values, params = train(train_frame, cols, args)
    summaries = []
    for split, frame in [("train_2024", train_frame), ("test_2025", test_frame), ("forward_2026", forward_frame)]:
        if frame is None:
            continue
        for threshold in parse_thresholds(args.thresholds):
            daily, summary = evaluate(frame, model, cols, fill_values, split, threshold, args)
            daily.to_csv(output / f"{split}_threshold_{threshold:g}_daily.csv", index=False)
            summaries.append(summary)
    importance = pd.DataFrame(
        {
            "feature": cols,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    with (output / "model.pkl").open("wb") as fh:
        pickle.dump(
            {
                "model": model,
                "feature_columns": cols,
                "fill_values": fill_values,
                "params": params,
                "target": args.target_col,
                "objective": args.objective,
                "positive_threshold": args.positive_threshold,
                "rank_label_bins": args.rank_label_bins,
            },
            fh,
        )
    meta = {
        "train_dataset": str(args.train_dataset),
        "test_dataset": str(args.test_dataset),
        "forward_dataset": str(args.forward_dataset) if args.forward_dataset else None,
        "feature_columns": cols,
        "params": params,
        "target": args.target_col,
        "objective": args.objective,
        "positive_threshold": args.positive_threshold,
        "rank_label_bins": args.rank_label_bins,
        "summaries": summaries,
    }
    (output / "training_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
