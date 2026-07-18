"""Train a confidence-gated V4 marginal-fill reranker."""

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


DATA_ROOT = Path("reranker_v3_data_20260615")
OUTPUT = Path("reranker_models_20260615/gated_v4")
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
    "baseline_target",
    "beats_baseline",
    "year",
}
GATE_QUANTILES = (0.20, 0.35, 0.50, 0.65, 0.80)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--validation-year", type=int, default=2023)
    parser.add_argument("--candidate-end", type=int, default=80)
    parser.add_argument("--max-reranked-fills", type=int, default=3)
    return parser.parse_args()


def feature_columns(frame):
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURES
        and not column.startswith("future_")
        and not column.startswith("exec_")
    ]


def add_relative_label(frame):
    baseline = (
        frame[frame["v3_baseline_fill"].eq(1)]
        .groupby("date")["exec_target_raw"]
        .mean()
        .rename("baseline_target")
    )
    frame = frame.join(baseline, on="date")
    frame["beats_baseline"] = (
        frame["exec_target_raw"] > frame["baseline_target"]
    ).astype(np.int8)
    return frame


def model_params(objective):
    params = {
        "objective": objective,
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
    params["metric"] = "binary_logloss" if objective == "binary" else "huber"
    return params


def fit_models(train, features, rounds=200):
    regression = lgb.train(
        model_params("huber"),
        lgb.Dataset(train[features], label=train["exec_target"]),
        num_boost_round=rounds,
    )
    classifier = lgb.train(
        model_params("binary"),
        lgb.Dataset(train[features], label=train["beats_baseline"]),
        num_boost_round=rounds,
    )
    return regression, classifier


def daily_percentile(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64) / max(len(values) - 1, 1)
    return result


def score_frame(frame, regression, classifier, features):
    scored = frame.copy()
    scored["pred_return"] = regression.predict(scored[features])
    scored["pred_win"] = classifier.predict(scored[features])
    scored["return_pct"] = scored.groupby("date")["pred_return"].transform(
        daily_percentile
    )
    scored["win_pct"] = scored.groupby("date")["pred_win"].transform(
        daily_percentile
    )
    scored["v4_score"] = 0.60 * scored["return_pct"] + 0.40 * scored["win_pct"]
    return scored


def evaluate_gate(scored, gate):
    rows = []
    for date, group in scored.groupby("date", sort=False):
        slots = int(group["v3_rerank_slots"].iloc[0])
        baseline = group[group["v3_baseline_fill"].eq(1)]
        if slots <= 0 or len(baseline) != slots:
            continue
        proposed = group.nlargest(slots, "v4_score")
        confidence = (
            proposed["pred_win"].mean() - baseline["pred_win"].mean()
        )
        active = confidence >= gate
        selected = proposed if active else baseline
        rows.append(
            {
                "date": date,
                "active": active,
                "confidence": confidence,
                "delta": (
                    selected["exec_target_raw"].mean()
                    - baseline["exec_target_raw"].mean()
                ),
            }
        )
    report = pd.DataFrame(rows)
    return {
        "gate": gate,
        "dates": len(report),
        "active_share": float(report["active"].mean()),
        "mean_delta": float(report["delta"].mean()),
        "positive_days": float((report["delta"] > 0).mean()),
        "active_delta": float(
            report.loc[report["active"], "delta"].mean()
            if report["active"].any()
            else 0.0
        ),
    }, report


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in sorted(data_root.glob("oof_F[1-6]_*/reranker_v3_dataset.parquet")):
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[
            frame["v3_eligible"].eq(1) & frame["exec_target"].notna()
        ].copy()
        frames.append(frame)
    data = add_relative_label(pd.concat(frames, ignore_index=True))
    data["year"] = data["date"].dt.year
    features = feature_columns(data)

    oof = []
    first_year = int(data["year"].min()) + 1
    for year in range(first_year, args.validation_year + 1):
        train = data[data["year"] < year]
        validation = data[data["year"] == year]
        regression, classifier = fit_models(train, features)
        scored = score_frame(validation, regression, classifier, features)
        oof.append(scored)
        print(f"OOF year {year}: train={len(train):,} validation={len(validation):,}")
    oof_frame = pd.concat(oof, ignore_index=True)

    calibration = oof_frame[oof_frame["year"] < args.validation_year]
    _, calibration_all = evaluate_gate(calibration, -np.inf)
    gate_grid = sorted(
        {
            float(calibration_all["confidence"].quantile(quantile))
            for quantile in GATE_QUANTILES
        }
    )
    gate_rows = []
    for gate in gate_grid:
        result, _ = evaluate_gate(calibration, gate)
        gate_rows.append(result)
    gate_report = pd.DataFrame(gate_rows)
    eligible_gates = gate_report[gate_report["active_share"].between(0.15, 0.80)]
    if eligible_gates.empty:
        raise RuntimeError("No V4 gate satisfies the preregistered coverage range")
    selected_gate = float(
        eligible_gates.sort_values(
            ["mean_delta", "positive_days", "active_share"],
            ascending=False,
        ).iloc[0]["gate"]
    )
    gate_report.to_csv(
        output / f"gate_calibration_{first_year}_{args.validation_year - 1}.csv",
        index=False,
    )

    validation_frame = oof_frame[oof_frame["year"] == args.validation_year]
    validation_result, validation_daily = evaluate_gate(
        validation_frame, selected_gate
    )
    validation_daily.to_csv(
        output / f"validation_{args.validation_year}_daily.csv", index=False
    )
    oof_frame[
        [
            "date",
            "code",
            "candidate_position",
            "v3_baseline_fill",
            "exec_target_raw",
            "pred_return",
            "pred_win",
            "v4_score",
        ]
    ].to_parquet(output / "rolling_oof_scores.parquet", index=False)

    final_regression, final_classifier = fit_models(data, features)
    with (output / "reranker_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "regression": final_regression,
                "classifier": final_classifier,
                "feature_columns": features,
                "gate": selected_gate,
                "score_weights": (0.60, 0.40),
                "candidate_end": args.candidate_end,
                "max_reranked_fills": args.max_reranked_fills,
            },
            handle,
        )
    summary = {
        "rows": int(len(data)),
        "dates": int(data["date"].nunique()),
        "features": len(features),
        "gate": selected_gate,
        "data_root": str(data_root),
        "validation_year": args.validation_year,
        "label_mode": "open_to_open",
        "calibration": gate_report.to_dict(orient="records"),
        "validation_2023": validation_result,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
