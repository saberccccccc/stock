"""Train a date-level rolling-OOF meta gate for V4 candidate proposals."""

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


V4_ROOT = Path("reranker_models_20260615/gated_v4")
DATA_ROOT = Path("reranker_v3_data_20260615")
OUTPUT = Path("reranker_models_20260615/meta_gate_v41")
THRESHOLD_QUANTILES = (0.20, 0.35, 0.50, 0.65, 0.80)


def load_candidate_scores():
    scores = pd.read_parquet(V4_ROOT / "rolling_oof_scores.parquet")
    scores["date"] = pd.to_datetime(scores["date"])
    state_frames = []
    columns = [
        "date",
        "code",
        "v3_vacancies",
        "v3_rerank_slots",
        "market_return_5d",
        "market_return_20d",
        "market_return_60d",
        "market_vol_20d",
        "market_vol_60d",
        "market_drawdown_60d",
        "market_ma20_gap",
        "market_ma60_gap",
    ]
    for path in sorted(DATA_ROOT.glob("oof_F[2-6]_*/reranker_v3_dataset.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame["date"] = pd.to_datetime(frame["date"])
        state_frames.append(frame)
    state = pd.concat(state_frames, ignore_index=True).drop_duplicates(
        ["date", "code"]
    )
    return scores.merge(state, on=["date", "code"], how="left", validate="one_to_one")


def build_daily_meta(scores):
    rows = []
    for date, group in scores.groupby("date", sort=True):
        slots = int(group["v3_rerank_slots"].iloc[0])
        baseline = group[group["v3_baseline_fill"].eq(1)]
        if slots <= 0 or len(baseline) != slots:
            continue
        proposed = group.nlargest(slots, "v4_score")
        proposed_codes = set(proposed["code"])
        baseline_codes = set(baseline["code"])
        overlap = len(proposed_codes & baseline_codes) / slots
        return_top = set(group.nlargest(slots, "pred_return")["code"])
        win_top = set(group.nlargest(slots, "pred_win")["code"])
        agreement = len(return_top & win_top) / slots
        pred_return_margin = (
            proposed["pred_return"].mean() - baseline["pred_return"].mean()
        )
        pred_win_margin = proposed["pred_win"].mean() - baseline["pred_win"].mean()
        score_margin = proposed["v4_score"].mean() - baseline["v4_score"].mean()
        realized_delta = (
            proposed["exec_target_raw"].mean()
            - baseline["exec_target_raw"].mean()
        )
        first = group.iloc[0]
        rows.append(
            {
                "date": date,
                "year": date.year,
                "vacancies": float(first["v3_vacancies"]),
                "slots": float(slots),
                "candidate_count": float(len(group)),
                "proposal_overlap": overlap,
                "head_agreement": agreement,
                "pred_return_margin": pred_return_margin,
                "pred_win_margin": pred_win_margin,
                "score_margin": score_margin,
                "pred_return_std": float(group["pred_return"].std()),
                "pred_win_std": float(group["pred_win"].std()),
                "score_std": float(group["v4_score"].std()),
                "proposed_return_std": float(proposed["pred_return"].std(ddof=0)),
                "proposed_win_std": float(proposed["pred_win"].std(ddof=0)),
                "baseline_return_std": float(baseline["pred_return"].std(ddof=0)),
                "baseline_win_std": float(baseline["pred_win"].std(ddof=0)),
                "market_return_5d": float(first["market_return_5d"]),
                "market_return_20d": float(first["market_return_20d"]),
                "market_return_60d": float(first["market_return_60d"]),
                "market_vol_20d": float(first["market_vol_20d"]),
                "market_vol_60d": float(first["market_vol_60d"]),
                "market_drawdown_60d": float(first["market_drawdown_60d"]),
                "market_ma20_gap": float(first["market_ma20_gap"]),
                "market_ma60_gap": float(first["market_ma60_gap"]),
                "realized_delta": realized_delta,
                "meta_label": int(realized_delta > 0),
            }
        )
    return pd.DataFrame(rows)


def feature_columns(frame):
    return [
        column
        for column in frame.columns
        if column not in {"date", "year", "realized_delta", "meta_label"}
    ]


def params():
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "seed": 42,
        "num_threads": 0,
        "verbosity": -1,
    }


def fit(train, features):
    return lgb.train(
        params(),
        lgb.Dataset(train[features], label=train["meta_label"]),
        num_boost_round=120,
    )


def evaluate(frame, threshold):
    active = frame["meta_probability"] >= threshold
    realized = frame["realized_delta"].where(active, 0.0)
    return {
        "threshold": float(threshold),
        "dates": int(len(frame)),
        "active_share": float(active.mean()),
        "mean_delta": float(realized.mean()),
        "active_delta": float(
            frame.loc[active, "realized_delta"].mean() if active.any() else 0.0
        ),
        "active_win_rate": float(
            (frame.loc[active, "realized_delta"] > 0).mean() if active.any() else 0.0
        ),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily = build_daily_meta(load_candidate_scores())
    features = feature_columns(daily)
    daily.to_parquet(OUTPUT / "daily_meta_dataset.parquet", index=False)

    rolling = []
    for year in (2021, 2022, 2023):
        train = daily[daily["year"] < year]
        validation = daily[daily["year"] == year].copy()
        model = fit(train, features)
        validation["meta_probability"] = model.predict(validation[features])
        rolling.append(validation)
        print(
            f"Meta OOF {year}: train={len(train)} validation={len(validation)}",
            flush=True,
        )
    rolling_frame = pd.concat(rolling, ignore_index=True)
    rolling_frame.to_csv(OUTPUT / "rolling_meta_oof.csv", index=False)

    calibration = rolling_frame[rolling_frame["year"] <= 2022]
    thresholds = sorted(
        {
            float(calibration["meta_probability"].quantile(quantile))
            for quantile in THRESHOLD_QUANTILES
        }
    )
    calibration_rows = [evaluate(calibration, threshold) for threshold in thresholds]
    calibration_report = pd.DataFrame(calibration_rows)
    eligible = calibration_report[
        calibration_report["active_share"].between(0.15, 0.65)
    ]
    selected_threshold = float(
        eligible.sort_values(
            ["mean_delta", "active_win_rate", "active_delta"],
            ascending=False,
        ).iloc[0]["threshold"]
    )
    calibration_report.to_csv(OUTPUT / "threshold_calibration_2021_2022.csv", index=False)

    validation_2023 = rolling_frame[rolling_frame["year"] == 2023]
    validation_result = evaluate(validation_2023, selected_threshold)
    final_model = fit(daily, features)
    with (OUTPUT / "meta_gate_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": final_model,
                "feature_columns": features,
                "threshold": selected_threshold,
            },
            handle,
        )
    summary = {
        "rows": int(len(daily)),
        "date_start": str(daily["date"].min().date()),
        "date_end": str(daily["date"].max().date()),
        "features": features,
        "threshold": selected_threshold,
        "calibration": calibration_rows,
        "validation_2023": validation_result,
    }
    (OUTPUT / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
