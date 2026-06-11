#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lightweight baseline evaluation for cross-section alpha sanity checks."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from core.config import DataConfig
from data.pipeline import build_cross_section_dataset, samples_from_precomputed_metadata


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 10:
        return np.nan
    a = a[valid]
    b = b[valid]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _top_bottom_spread(score, ret, frac=0.10):
    score = np.asarray(score, dtype=np.float64)
    ret = np.asarray(ret, dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(ret)
    if valid.sum() < 20:
        return np.nan
    score = score[valid]
    ret = ret[valid]
    k = max(5, int(len(score) * frac))
    if len(score) < 2 * k:
        return np.nan
    order = np.argsort(score)
    return float(np.mean(ret[order[-k:]]) - np.mean(ret[order[:k]]))


def _rank01(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(np.argsort(x))
    return order / max(len(order) - 1, 1)


def _sample_target(sample, target_horizon):
    idx = min(target_horizon - 1, sample["y_seq"].shape[1] - 1)
    return sample["y_seq"][:, idx]


def load_samples(cfg):
    result = build_cross_section_dataset(cfg, use_cache=True)
    if isinstance(result, dict):
        cfg.low_feat_dim = result.get("low_agg_dim", getattr(cfg, "low_feat_dim", 14))
        return (
            samples_from_precomputed_metadata(result, "train"),
            samples_from_precomputed_metadata(result, "val"),
        )
    return result


def fit_ridge(train_samples, target_horizon, max_rows=200000, l2=10.0):
    X, y = collect_rows(train_samples, target_horizon, max_rows)
    X1 = np.column_stack([np.ones(len(X), dtype=np.float32), X])
    eye = np.eye(X1.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    beta = np.linalg.solve(X1.T @ X1 + l2 * eye, X1.T @ y)
    return beta


def collect_rows(samples, target_horizon, max_rows=200000):
    xs, ys = [], []
    n_rows = 0
    for sample in samples:
        x = sample["X"].astype(np.float32)
        y = _sample_target(sample, target_horizon).astype(np.float32)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if valid.sum() == 0:
            continue
        xs.append(x[valid])
        ys.append(y[valid])
        n_rows += int(valid.sum())
        if n_rows >= max_rows:
            break
    if not xs:
        raise ValueError("no valid rows for ridge baseline")
    X = np.vstack(xs)[:max_rows]
    y = np.concatenate(ys)[:max_rows]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def score_ridge(sample, beta):
    x = np.nan_to_num(sample["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return beta[0] + x @ beta[1:]


def fit_lgb(train_samples, val_samples, target_horizon, max_train_rows=200000, max_val_rows=50000):
    import lightgbm as lgb

    X_train, y_train = collect_rows(train_samples, target_horizon, max_train_rows)
    X_val, y_val = collect_rows(val_samples, target_horizon, max_val_rows)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "min_data_in_leaf": 50,
        "num_threads": 8,
        "verbose": -1,
        "seed": 42,
    }
    return lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )


def evaluate_scores(val_samples, target_horizon, scorers):
    rows = []
    for name, scorer in scorers.items():
        ics, rank_ics, spreads = [], [], []
        for sample in val_samples:
            y = _sample_target(sample, target_horizon)
            score = scorer(sample)
            ic = _corr(score, y)
            ric = _corr(_rank01(score), _rank01(y))
            spread = _top_bottom_spread(score, y)
            if np.isfinite(ic):
                ics.append(ic)
            if np.isfinite(ric):
                rank_ics.append(ric)
            if np.isfinite(spread):
                spreads.append(spread)
        rows.append({
            "name": name,
            "mean_ic": float(np.mean(ics)) if ics else 0.0,
            "icir": float(np.mean(ics) / (np.std(ics) + 1e-8)) if ics else 0.0,
            "rank_ic": float(np.mean(rank_ics)) if rank_ics else 0.0,
            "topbot": float(np.mean(spreads)) if spreads else 0.0,
            "n_dates": len(ics),
        })
    return pd.DataFrame(rows).sort_values("mean_ic", ascending=False)


def build_config(args):
    cfg = DataConfig()
    cfg.use_technical_features = True
    cfg.use_market_features = True
    cfg.use_macro_features = True
    cfg.use_fundamental_features = True
    cfg.use_shareholder_features = True
    cfg.use_restricted_features = True
    cfg.target_horizon = args.target_horizon
    cfg.seq_len = 40
    cfg.max_horizon = 10
    cfg.min_stocks_per_time = 30
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.test_stocks:
        cfg.test_mode = True
        cfg.test_stocks = args.test_stocks
        cfg.max_stocks = args.test_stocks
        cfg.min_stocks_per_time = max(10, min(cfg.min_stocks_per_time, args.test_stocks // 2))
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate simple alpha baselines on the shared dataset.")
    parser.add_argument("--test-stocks", type=int, default=None)
    parser.add_argument("--target-horizon", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=200000)
    parser.add_argument("--max-lgb-val-rows", type=int, default=50000)
    parser.add_argument("--ridge-l2", type=float, default=10.0)
    parser.add_argument("--include-lgb", action="store_true", help="Train and evaluate a LightGBM baseline.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = build_config(args)
    train, val = load_samples(cfg)
    print(f"train dates: {len(train)}, val dates: {len(val)}")

    ridge_beta = fit_ridge(train, args.target_horizon, args.max_train_rows, args.ridge_l2)
    scorers = {
        "risk_momentum_20d": lambda s: s["risk"][:, 2],
        "risk_reversal_5d": lambda s: -s["risk"][:, 3],
        "risk_low_vol_60d": lambda s: -s["risk"][:, 1],
        "risk_turnover": lambda s: s["risk"][:, 4],
        "x_first_feature": lambda s: s["X"][:, 0],
        "x_mean_score": lambda s: np.nanmean(s["X"], axis=1),
        "ridge_x": lambda s: score_ridge(s, ridge_beta),
    }
    if args.include_lgb:
        print("training LightGBM baseline...")
        lgb_model = fit_lgb(train, val, args.target_horizon, args.max_train_rows, args.max_lgb_val_rows)
        scorers["lightgbm_x"] = lambda s: lgb_model.predict(
            np.nan_to_num(s["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            num_iteration=lgb_model.best_iteration,
        )
    report = evaluate_scores(val, args.target_horizon, scorers)
    print("\n== Baseline validation ==")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
