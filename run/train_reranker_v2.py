"""Train a continuous executable-return boundary reranker."""

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


DATA_ROOT = Path("reranker_v2_data_20260615")
OUTPUT = Path("reranker_models_20260615/regression_v2")
NON_FEATURE_PREFIXES = ("future_", "exec_")
NON_FEATURES = {
    "split",
    "date",
    "code",
    "group_size",
    "candidate_position",
    "relevance",
    "market_regime",
}


def feature_columns(frame):
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURES
        and not column.startswith(NON_FEATURE_PREFIXES)
    ]


def boundary_selection_delta(frame, predictions, replacements=3):
    scored = frame[
        ["date", "candidate_position", "exec_target_raw"]
    ].copy()
    scored["score"] = predictions
    deltas = []
    for _, group in scored.groupby("date", sort=False):
        original = group.nsmallest(replacements, "candidate_position")
        selected = group.nlargest(replacements, "score")
        deltas.append(
            selected["exec_target_raw"].mean() - original["exec_target_raw"].mean()
        )
    return float(np.nanmean(deltas)), float(np.mean(np.asarray(deltas) > 0))


def params():
    return {
        "objective": "huber",
        "metric": "huber",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 150,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "seed": 42,
        "num_threads": 0,
        "verbosity": -1,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in sorted(DATA_ROOT.glob("oof_F[1-6]_*/reranker_v2_dataset.parquet")):
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[frame["exec_target"].notna()].copy()
        frames.append(frame)
        print(f"Loaded {path.parent.name}: {len(frame):,}", flush=True)
    data = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "candidate_position"], kind="mergesort"
    )
    features = feature_columns(data)
    train = data[data["date"].dt.year < 2023]
    validation = data[data["date"].dt.year == 2023]
    train_set = lgb.Dataset(train[features], label=train["exec_target"])
    model = lgb.train(params(), train_set, num_boost_round=400)

    iteration_rows = []
    best = None
    for iteration in range(20, 401, 10):
        predictions = model.predict(validation[features], num_iteration=iteration)
        delta, positive = boundary_selection_delta(validation, predictions)
        row = {
            "iteration": iteration,
            "replacement_delta": delta,
            "positive_days": positive,
        }
        iteration_rows.append(row)
        key = (delta, positive, -iteration)
        if best is None or key > best[0]:
            best = (key, iteration)
    report = pd.DataFrame(iteration_rows)
    report.to_csv(OUTPUT / "validation_2023_iterations.csv", index=False)
    best_iteration = int(best[1])
    print(report.sort_values("replacement_delta", ascending=False).head(10).to_string(index=False))
    print(f"Selected iteration={best_iteration}", flush=True)

    full_set = lgb.Dataset(data[features], label=data["exec_target"])
    final_model = lgb.train(params(), full_set, num_boost_round=best_iteration)
    with (OUTPUT / "reranker_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": final_model,
                "feature_columns": features,
                "best_iteration": best_iteration,
                "boundary_start": 27,
                "boundary_end": 80,
                "max_replacements": 3,
            },
            handle,
        )
    importance = pd.DataFrame(
        {
            "feature": features,
            "gain": final_model.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(OUTPUT / "feature_importance.csv", index=False)
    summary = {
        "rows": int(len(data)),
        "dates": int(data["date"].nunique()),
        "features": len(features),
        "best_iteration": best_iteration,
        "validation_replacement_delta": float(best[0][0]),
        "validation_positive_days": float(best[0][1]),
    }
    (OUTPUT / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
