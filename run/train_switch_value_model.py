#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train and bucket-validate a switch value model."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pandas.api.types import is_object_dtype, is_string_dtype

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


NON_FEATURE_PREFIXES = (
    "A_ret_fwd_",
    "B_ret_fwd_",
    "switch_edge_raw_",
    "switch_edge_net_",
    "switch_success_",
)
NON_FEATURE_COLS = {
    "split",
    "date",
    "A_code",
    "B_code",
}


def load_dataset(path):
    path = Path(path)
    if path.is_dir():
        train_path = path / "switch_value_dataset_train.parquet"
        val_path = path / "switch_value_dataset_val.parquet"
        if train_path.exists() and val_path.exists():
            return pd.concat([pd.read_parquet(train_path), pd.read_parquet(val_path)], axis=0, ignore_index=True)
        train_csv = path / "switch_value_dataset_train.csv"
        val_csv = path / "switch_value_dataset_val.csv"
        if train_csv.exists() and val_csv.exists():
            return pd.concat([pd.read_csv(train_csv), pd.read_csv(val_csv)], axis=0, ignore_index=True)
        raise FileNotFoundError(f"No split switch value dataset files found under {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def is_feature_col(col):
    if col in NON_FEATURE_COLS:
        return False
    return not any(col.startswith(prefix) for prefix in NON_FEATURE_PREFIXES)


def make_model(args):
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            objective="regression",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            random_state=args.seed,
            n_jobs=-1,
        ), "lightgbm"
    except Exception as exc:
        print(f"LightGBM unavailable, fallback to sklearn HistGradientBoosting: {exc}")
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=args.n_estimators,
            learning_rate=args.learning_rate,
            random_state=args.seed,
        ), "hist_gradient_boosting"


def prepare_xy(train_df, val_df, target_col):
    feature_cols = [c for c in train_df.columns if is_feature_col(c)]
    all_x = pd.concat([train_df[feature_cols], val_df[feature_cols]], axis=0)
    categorical_cols = [
        c for c in all_x.columns
        if (
            is_object_dtype(all_x[c])
            or is_string_dtype(all_x[c])
            or isinstance(all_x[c].dtype, pd.CategoricalDtype)
        )
    ]
    all_x = pd.get_dummies(all_x, columns=categorical_cols, dummy_na=True)
    x_train = all_x.iloc[:len(train_df)].copy()
    x_val = all_x.iloc[len(train_df):].copy()
    y_train = train_df[target_col].astype(float)
    y_val = val_df[target_col].astype(float)
    return x_train, y_train, x_val, y_val, list(all_x.columns)


def horizon_from_target(target_col):
    match = re.search(r"_h(\d+)$", target_col)
    if not match:
        raise ValueError(f"target_col must end with horizon suffix like _h5: {target_col}")
    return int(match.group(1))


def regression_metrics(df, pred, target_col):
    y = df[target_col].astype(float).values
    finite = np.isfinite(y) & np.isfinite(pred)
    y = y[finite]
    p = pred[finite]
    if len(y) == 0:
        return {}
    ic = spearmanr(p, y).correlation if len(y) > 2 else np.nan
    success = y > 0
    pred_success = p > 0
    return {
        "rows": int(len(y)),
        "target_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "spearman_ic": float(ic) if np.isfinite(ic) else np.nan,
        "success_rate": float(np.mean(success)),
        "pred_positive_rate": float(np.mean(pred_success)),
        "pred_positive_true_edge_mean": float(np.mean(y[pred_success])) if np.any(pred_success) else np.nan,
        "pred_positive_success_rate": float(np.mean(success[pred_success])) if np.any(pred_success) else np.nan,
    }


def bucket_report(df, pred, target_col, n_buckets=10):
    horizon = horizon_from_target(target_col)
    success_col = f"switch_success_h{horizon}"
    raw_edge_col = f"switch_edge_raw_h{horizon}"
    tmp = df.copy()
    tmp["pred_switch_value"] = pred
    tmp = tmp[np.isfinite(tmp["pred_switch_value"]) & np.isfinite(tmp[target_col])].copy()
    if tmp.empty:
        return pd.DataFrame()
    try:
        tmp["bucket"] = pd.qcut(tmp["pred_switch_value"], q=n_buckets, duplicates="drop")
    except ValueError:
        tmp["bucket"] = "all"
    report = (
        tmp.groupby("bucket", observed=True)
        .agg(
            rows=(target_col, "size"),
            pred_mean=("pred_switch_value", "mean"),
            true_edge_mean=(target_col, "mean"),
            success_rate=(success_col, "mean"),
            raw_edge_mean=(raw_edge_col, "mean"),
            full_cost_mean=("switch_full_cost", "mean"),
            explicit_cost_mean=("switch_explicit_cost", "mean"),
            impact_cost_mean=("switch_impact_cost", "mean"),
            execution_risk_cost_mean=("switch_execution_risk_cost", "mean"),
            same_industry_rate=("same_industry", "mean"),
            rank_advantage_mean=("rank_advantage", "mean"),
            alpha_diff_mean=("alpha_diff", "mean"),
        )
        .reset_index()
    )
    report["bucket"] = report["bucket"].astype(str)
    return report


def yearly_bucket_report(df, pred, target_col, n_buckets=5):
    horizon = horizon_from_target(target_col)
    success_col = f"switch_success_h{horizon}"
    tmp = df.copy()
    tmp["year"] = pd.to_datetime(tmp["date"]).dt.year
    tmp["pred_switch_value"] = pred
    tmp = tmp[np.isfinite(tmp["pred_switch_value"]) & np.isfinite(tmp[target_col])].copy()
    if tmp.empty:
        return pd.DataFrame()
    try:
        tmp["bucket_id"] = pd.qcut(tmp["pred_switch_value"], q=n_buckets, labels=False, duplicates="drop")
    except ValueError:
        tmp["bucket_id"] = 0
    report = (
        tmp.groupby(["year", "bucket_id"], observed=True)
        .agg(
            rows=(target_col, "size"),
            pred_mean=("pred_switch_value", "mean"),
            true_edge_mean=(target_col, "mean"),
            success_rate=(success_col, "mean"),
            full_cost_mean=("switch_full_cost", "mean"),
        )
        .reset_index()
        .sort_values(["year", "bucket_id"])
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Train switch value model and validate buckets.")
    parser.add_argument("--dataset", default="switch_value_data_20260530/switch_value_dataset.parquet")
    parser.add_argument("--output-dir", default="switch_value_models_20260530/switch_edge_lgb_h5")
    parser.add_argument("--target-col", default="switch_edge_net_h5")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=300)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("dataset must contain both train and val rows")
    train_df = train_df[np.isfinite(train_df[args.target_col])].copy()
    val_df = val_df[np.isfinite(val_df[args.target_col])].copy()

    x_train, y_train, x_val, y_val, feature_cols = prepare_xy(train_df, val_df, args.target_col)
    model, model_type = make_model(args)
    model.fit(x_train, y_train)

    train_pred = np.asarray(model.predict(x_train), dtype=float)
    val_pred = np.asarray(model.predict(x_val), dtype=float)

    metrics = {
        "model_type": model_type,
        "target_col": args.target_col,
        "feature_cols": feature_cols,
        "train": regression_metrics(train_df, train_pred, args.target_col),
        "val": regression_metrics(val_df, val_pred, args.target_col),
    }
    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "metrics": metrics, "target_col": args.target_col},
        out_dir / "switch_value_model.pkl",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bucket_report(train_df, train_pred, args.target_col).to_csv(out_dir / "bucket_report_train.csv", index=False)
    bucket_report(val_df, val_pred, args.target_col).to_csv(out_dir / "bucket_report_val.csv", index=False)
    yearly_bucket_report(val_df, val_pred, args.target_col).to_csv(out_dir / "yearly_bucket_report_val.csv", index=False)

    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        imp.sort_values("importance", ascending=False).to_csv(out_dir / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to: {out_dir / 'switch_value_model.pkl'}")


if __name__ == "__main__":
    main()
