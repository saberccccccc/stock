"""Train the state-aware marginal fill reranker."""

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


DATA_ROOT = Path("reranker_v3_data_20260615")
OUTPUT = Path("reranker_models_20260615/regression_v3")
NON_FEATURES = {
    "split",
    "date",
    "code",
    "group_size",
    "relevance",
    "market_regime",
    "v3_is_kept",
    "v3_protected_fill",
    "v3_eligible",
    "v3_baseline_fill",
}


def feature_columns(frame):
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURES
        and not column.startswith("future_")
        and not column.startswith("exec_")
    ]


def replacement_delta(frame, predictions):
    scored = frame[
        ["date", "v3_rerank_slots", "v3_baseline_fill", "exec_target_raw"]
    ].copy()
    scored["score"] = predictions
    deltas = []
    for _, group in scored.groupby("date", sort=False):
        slots = int(group["v3_rerank_slots"].iloc[0])
        if slots <= 0:
            continue
        selected = group.nlargest(slots, "score")
        baseline = group[group["v3_baseline_fill"].eq(1)]
        if len(baseline) != slots:
            continue
        deltas.append(
            selected["exec_target_raw"].mean()
            - baseline["exec_target_raw"].mean()
        )
    values = np.asarray(deltas, dtype=np.float64)
    return float(values.mean()), float(np.mean(values > 0)), len(values)


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
    for path in sorted(DATA_ROOT.glob("oof_F[1-6]_*/reranker_v3_dataset.parquet")):
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[
            frame["v3_eligible"].eq(1) & frame["exec_target"].notna()
        ].copy()
        frames.append(frame)
        print(f"Loaded {path.parent.name}: {len(frame):,}", flush=True)
    data = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "candidate_position"], kind="mergesort"
    )
    features = feature_columns(data)
    train = data[data["date"].dt.year < 2023]
    validation = data[data["date"].dt.year == 2023]
    model = lgb.train(
        params(),
        lgb.Dataset(train[features], label=train["exec_target"]),
        num_boost_round=400,
    )

    rows = []
    best = None
    for iteration in range(20, 401, 10):
        predictions = model.predict(validation[features], num_iteration=iteration)
        delta, positive, dates = replacement_delta(validation, predictions)
        rows.append(
            {
                "iteration": iteration,
                "replacement_delta": delta,
                "positive_days": positive,
                "dates": dates,
            }
        )
        key = (delta, positive, -iteration)
        if best is None or key > best[0]:
            best = (key, iteration)
    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT / "validation_2023_iterations.csv", index=False)
    best_iteration = int(best[1])
    print(report.sort_values("replacement_delta", ascending=False).head(10).to_string(index=False))
    print(f"Selected iteration={best_iteration}", flush=True)

    final_model = lgb.train(
        params(),
        lgb.Dataset(data[features], label=data["exec_target"]),
        num_boost_round=best_iteration,
    )
    with (OUTPUT / "reranker_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": final_model,
                "feature_columns": features,
                "best_iteration": best_iteration,
                "candidate_end": 80,
                "max_reranked_fills": 3,
            },
            handle,
        )
    pd.DataFrame(
        {
            "feature": features,
            "gain": final_model.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False).to_csv(
        OUTPUT / "feature_importance.csv", index=False
    )
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
