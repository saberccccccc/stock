"""Train the first LightGBM LambdaRank candidate reranker."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from run.build_reranker_dataset import NON_FEATURE_COLS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="reranker_data_20260614",
    )
    parser.add_argument(
        "--output-dir",
        default="reranker_models_20260615/lambdarank_v1",
    )
    parser.add_argument("--validation-year", type=int, default=2023)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=200)
    parser.add_argument("--feature-fraction", type=float, default=0.80)
    parser.add_argument("--bagging-fraction", type=float, default=0.80)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def dataset_paths(data_root):
    paths = sorted(Path(data_root).glob("oof_F[1-6]_*/reranker_dataset.parquet"))
    if len(paths) != 6:
        raise ValueError(f"Expected six OOF datasets under {data_root}, found {len(paths)}")
    return paths


def load_datasets(paths):
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame)
        print(
            f"Loaded {path.parent.name}: rows={len(frame):,}, "
            f"dates={frame['date'].nunique()}",
            flush=True,
        )
    frame = pd.concat(frames, axis=0, ignore_index=True)
    frame = frame.sort_values(["date", "candidate_position"], kind="mergesort")
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate date/code rows found in combined OOF data")
    return frame.reset_index(drop=True)


def feature_columns(frame):
    columns = [column for column in frame.columns if column not in NON_FEATURE_COLS]
    forbidden = [
        column
        for column in columns
        if column.startswith("future_") or column == "relevance"
    ]
    if forbidden:
        raise ValueError(f"Forbidden future columns in features: {forbidden}")
    return columns


def group_sizes(frame):
    return frame.groupby("date", sort=False).size().astype(int).tolist()


def training_params(args):
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [30, 50, 100],
        "label_gain": [0, 1, 3, 7, 15],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_child_samples,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.reg_lambda,
        "seed": args.seed,
        "num_threads": 0,
        "verbosity": -1,
    }


def dcg_at_k(labels, scores, k):
    order = np.argsort(-scores, kind="mergesort")[:k]
    gains = np.power(2.0, labels[order]) - 1.0
    discounts = 1.0 / np.log2(np.arange(len(order), dtype=np.float64) + 2.0)
    return float(np.sum(gains * discounts))


def mean_ndcg(frame, score_col, k):
    values = []
    for _, group in frame.groupby("date", sort=False):
        labels = group["relevance"].to_numpy(dtype=np.int64)
        scores = group[score_col].to_numpy(dtype=np.float64)
        ideal = dcg_at_k(labels, labels.astype(np.float64), k)
        if ideal <= 0:
            continue
        values.append(dcg_at_k(labels, scores, k) / ideal)
    return float(np.mean(values)) if values else np.nan


def score_diagnostics(frame, predictions):
    scored = frame[
        ["date", "code", "candidate_position", "m0_alpha", "future_target", "relevance"]
    ].copy()
    scored["reranker_score"] = np.asarray(predictions, dtype=np.float64)
    scored["m0_score"] = -scored["candidate_position"].astype(float)

    metrics = {}
    for name, column in (("m0", "m0_score"), ("reranker", "reranker_score")):
        metrics[name] = {
            f"ndcg@{k}": mean_ndcg(scored, column, k)
            for k in (30, 50, 100)
        }

    bucketed = scored.copy()
    bucketed["reranker_bucket"] = bucketed.groupby("date")["reranker_score"].transform(
        lambda values: pd.qcut(
            values.rank(method="first"),
            q=10,
            labels=False,
            duplicates="drop",
        )
    )
    bucket_report = (
        bucketed.groupby("reranker_bucket", observed=True)
        .agg(
            rows=("future_target", "size"),
            target_mean=("future_target", "mean"),
            relevance_mean=("relevance", "mean"),
            m0_position_mean=("candidate_position", "mean"),
        )
        .reset_index()
    )
    return metrics, bucket_report, scored


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_datasets(dataset_paths(ROOT / args.data_root))
    features = feature_columns(frame)
    year = frame["date"].dt.year
    train = frame[year < args.validation_year].copy()
    validation = frame[year == args.validation_year].copy()
    if train.empty or validation.empty:
        raise ValueError("Temporal training/validation split is empty")

    print(
        f"Temporal fit: train={len(train):,} rows through "
        f"{train['date'].max().date()}, validation={len(validation):,} rows in "
        f"{args.validation_year}",
        flush=True,
    )
    train_set = lgb.Dataset(
        train[features],
        label=train["relevance"].astype(int),
        group=group_sizes(train),
        feature_name=features,
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation[features],
        label=validation["relevance"].astype(int),
        group=group_sizes(validation),
        feature_name=features,
        reference=train_set,
        free_raw_data=False,
    )
    ranker = lgb.train(
        training_params(args),
        train_set,
        num_boost_round=args.n_estimators,
        valid_sets=[validation_set],
        valid_names=["validation_2023"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=20),
        ],
    )
    best_iteration = int(ranker.best_iteration or args.n_estimators)
    validation_prediction = ranker.predict(validation[features], num_iteration=best_iteration)
    validation_metrics, validation_buckets, validation_scored = score_diagnostics(
        validation,
        validation_prediction,
    )
    validation_buckets.to_csv(output_dir / "validation_2023_buckets.csv", index=False)
    validation_scored.to_parquet(
        output_dir / "validation_2023_scores.parquet",
        index=False,
    )

    print(
        f"Refitting final model on all {len(frame):,} OOF rows with "
        f"{best_iteration} trees",
        flush=True,
    )
    full_set = lgb.Dataset(
        frame[features],
        label=frame["relevance"].astype(int),
        group=group_sizes(frame),
        feature_name=features,
    )
    final_ranker = lgb.train(
        training_params(args),
        full_set,
        num_boost_round=best_iteration,
    )
    with (output_dir / "reranker_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": final_ranker,
                "feature_columns": features,
                "best_iteration": best_iteration,
                "validation_year": args.validation_year,
            },
            handle,
        )

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_gain": final_ranker.feature_importance(importance_type="gain"),
            "importance_split": final_ranker.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    summary = {
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "date_start": str(frame["date"].min().date()),
        "date_end": str(frame["date"].max().date()),
        "features": int(len(features)),
        "validation_year": int(args.validation_year),
        "best_iteration": best_iteration,
        "validation_metrics": validation_metrics,
        "parameters": vars(args),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved reranker to {output_dir / 'reranker_model.pkl'}", flush=True)


if __name__ == "__main__":
    main()
