#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the v1 hold/sell trade policy model."""
import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


NON_FEATURE_COLS = {
    "split",
    "date",
    "code",
    "label_hold",
    "hold_ret_fwd",
    "replace_ret_median",
    "replace_ret_mean",
    "edge_vs_replace",
}


def load_dataset(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def make_model(args):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective="binary",
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
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=args.n_estimators,
            learning_rate=args.learning_rate,
            random_state=args.seed,
        ), "hist_gradient_boosting"


def predict_proba_hold(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def metrics_for_split(df, proba):
    y = df["label_hold"].astype(int).values
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "rows": int(len(df)),
        "hold_rate": float(np.mean(y)) if len(y) else np.nan,
        "auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else np.nan,
        "precision_hold": float(precision_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "recall_hold": float(recall_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "edge_mean": float(df["edge_vs_replace"].mean()) if "edge_vs_replace" in df else np.nan,
    }
    return metrics


def classification_metrics_at_threshold(df, proba, threshold):
    y = df["label_hold"].astype(int).values
    pred = (proba >= threshold).astype(int)
    tp = float(np.sum((pred == 1) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    tpr = tp / max(tp + fn, 1.0)
    tnr = tn / max(tn + fp, 1.0)
    precision_hold = tp / max(tp + fp, 1.0)
    recall_hold = tpr
    balanced_accuracy = 0.5 * (tpr + tnr)
    keep_rate = float(np.mean(pred)) if len(pred) else np.nan
    edge_kept = float(df.loc[pred == 1, "edge_vs_replace"].mean()) if np.any(pred == 1) else np.nan
    edge_sold = float(df.loc[pred == 0, "edge_vs_replace"].mean()) if np.any(pred == 0) else np.nan
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else np.nan,
        "balanced_accuracy": float(balanced_accuracy),
        "precision_hold": float(precision_hold),
        "recall_hold": float(recall_hold),
        "true_negative_rate": float(tnr),
        "keep_rate": keep_rate,
        "edge_kept_mean": edge_kept,
        "edge_sold_mean": edge_sold,
    }


def learn_sell_threshold(train_df, train_proba):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        rows.append(classification_metrics_at_threshold(train_df, train_proba, threshold))
    report = pd.DataFrame(rows)
    feasible = report[(report["keep_rate"] >= 0.10) & (report["keep_rate"] <= 0.90)].copy()
    if feasible.empty:
        feasible = report
    best = feasible.sort_values(
        ["balanced_accuracy", "precision_hold", "true_negative_rate"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    return float(best["threshold"]), best, report


def bucket_report(df, proba, n_buckets=10):
    tmp = df.copy()
    tmp["hold_prob"] = proba
    tmp["bucket"] = pd.qcut(tmp["hold_prob"], q=n_buckets, duplicates="drop")
    report = (
        tmp.groupby("bucket", observed=True)
        .agg(
            rows=("label_hold", "size"),
            hold_prob_mean=("hold_prob", "mean"),
            label_hold_rate=("label_hold", "mean"),
            edge_mean=("edge_vs_replace", "mean"),
            hold_ret_mean=("hold_ret_fwd", "mean"),
            replace_ret_median_mean=("replace_ret_median", "mean"),
        )
        .reset_index()
    )
    report["bucket"] = report["bucket"].astype(str)
    return report


def main():
    parser = argparse.ArgumentParser(description="Train hold/sell trade policy model.")
    parser.add_argument("--dataset", default="trade_policy_data/hold_sell_policy_dataset.csv")
    parser.add_argument("--output-dir", default="models_trade_policy/hold_sell_lgb_v1")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=200)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("dataset must contain both train and val split rows")

    X_train = train_df[feature_cols]
    y_train = train_df["label_hold"].astype(int)
    X_val = val_df[feature_cols]

    model, model_type = make_model(args)
    model.fit(X_train, y_train)

    train_proba = predict_proba_hold(model, X_train)
    val_proba = predict_proba_hold(model, X_val)
    sell_threshold, threshold_metrics, threshold_report = learn_sell_threshold(train_df, train_proba)
    metrics = {
        "model_type": model_type,
        "feature_cols": feature_cols,
        "sell_threshold": sell_threshold,
        "threshold_selection": {
            "method": "train_balanced_accuracy_grid",
            "train": threshold_metrics,
            "val_at_threshold": classification_metrics_at_threshold(val_df, val_proba, sell_threshold),
        },
        "train": metrics_for_split(train_df, train_proba),
        "val": metrics_for_split(val_df, val_proba),
    }

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "metrics": metrics,
            "sell_threshold": sell_threshold,
        },
        out_dir / "hold_sell_policy_model.pkl",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    threshold_report.to_csv(out_dir / "threshold_report_train.csv", index=False)
    bucket_report(train_df, train_proba).to_csv(out_dir / "bucket_report_train.csv", index=False)
    bucket_report(val_df, val_proba).to_csv(out_dir / "bucket_report_val.csv", index=False)

    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        imp.sort_values("importance", ascending=False).to_csv(out_dir / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to: {out_dir / 'hold_sell_policy_model.pkl'}")


if __name__ == "__main__":
    main()
