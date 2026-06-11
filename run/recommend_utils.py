# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import detect_regime
from backtest.predictors import V9GATEnsemblePredictor, V9GATIntersectionPredictor
from backtest.runtime import load_v9_gat_predictors


MAIN_BOARD_PREFIXES = ("000", "001", "002", "003", "302", "600", "601", "603", "605")
CHINEXT_PREFIXES = ("300", "301")
STAR_PREFIXES = ("688", "689")
BSE_PREFIXES = ("8", "9")


def classify_board(code):
    code = str(code)
    if code.startswith(MAIN_BOARD_PREFIXES):
        return "主板"
    if code.startswith(CHINEXT_PREFIXES):
        return "创业板"
    if code.startswith(STAR_PREFIXES):
        return "科创板"
    if code.startswith(BSE_PREFIXES):
        return "北交所"
    return "其他"


def filter_main_board(df, code_col="code"):
    mask = df[code_col].str.startswith(MAIN_BOARD_PREFIXES)
    filtered = df[mask].reset_index(drop=True)
    n_removed = len(df) - len(filtered)
    if n_removed:
        print(f"  已排除 {n_removed} 只非主板股票，剩余 {len(filtered)} 只")
    return filtered


def filter_codes_by_prefix(df, prefixes, code_col="code"):
    if not prefixes:
        return df
    mask = pd.Series(True, index=df.index)
    for prefix in prefixes:
        mask = mask & ~df[code_col].str.startswith(prefix)
    filtered = df[mask].reset_index(drop=True)
    n_removed = len(df) - len(filtered)
    if n_removed:
        prefix_str = "、".join(prefixes)
        print(f"  已排除 {n_removed} 只 {prefix_str} 开头股票，剩余 {len(filtered)} 只")
    return filtered


def build_recommendation_predictor(
    predictor_name,
    train_samples,
    cfg,
    *,
    v9_checkpoint="checkpoints_exp/ultimate_v7_best.pt",
    gat_checkpoint="checkpoints_exp/ultimate_v7_gat_best.pt",
    device="auto",
    signal_top_pct=0.10,
):
    predictors = load_v9_gat_predictors(
        train_samples,
        cfg,
        v9_checkpoint=v9_checkpoint,
        gat_checkpoint=gat_checkpoint,
        device=device,
    )
    if predictor_name == "v9":
        return predictors.v9_predictor
    if predictor_name == "gat":
        return predictors.gat_predictor
    if predictor_name == "avg_score":
        return V9GATEnsemblePredictor(
            predictors.v9_predictor,
            predictors.gat_predictor,
            strategy="avg_score",
            top_pct=signal_top_pct,
        )
    if predictor_name == "union":
        return V9GATEnsemblePredictor(
            predictors.v9_predictor,
            predictors.gat_predictor,
            strategy="union",
            top_pct=signal_top_pct,
        )
    if predictor_name == "intersection":
        return V9GATIntersectionPredictor(
            predictors.v9_predictor,
            predictors.gat_predictor,
            top_frac=signal_top_pct,
        )
    if predictor_name == "top_union_bottom_intersection":
        return V9GATEnsemblePredictor(
            predictors.v9_predictor,
            predictors.gat_predictor,
            strategy="top_union_bottom_intersection",
            top_pct=signal_top_pct,
        )
    raise ValueError(f"Unsupported predictor: {predictor_name}")


def predict_alpha_with_regime(predictor, sample):
    valid = np.ones(len(sample["codes"]), dtype=bool)
    regime = detect_regime(sample)
    alpha = predictor.predict_alpha(sample, valid, regime)
    alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
    if len(alpha) != len(sample["codes"]):
        raise ValueError(f"预测长度不匹配: alpha={len(alpha)}, codes={len(sample['codes'])}")
    return alpha, regime


def rank_alpha(alpha):
    order = np.argsort(-alpha)
    ranks = np.empty(len(alpha), dtype=np.int32)
    ranks[order] = np.arange(1, len(alpha) + 1)
    if len(alpha) > 1:
        percentile = 1.0 - (ranks - 1) / (len(alpha) - 1)
    else:
        percentile = np.ones(len(alpha), dtype=np.float64)
    return ranks, percentile


def score_recommendation_sample(
    predictor,
    sample,
    *,
    include_percentile=True,
    include_predictor=True,
    include_regime=True,
    include_board=True,
):
    alpha, regime = predict_alpha_with_regime(predictor, sample)
    ranks, percentile = rank_alpha(alpha)
    data = {
        "date": pd.Timestamp(sample["date"]).strftime("%Y-%m-%d"),
        "rank": ranks,
        "code": sample["codes"],
        "alpha": alpha,
    }
    if include_percentile:
        data["percentile"] = percentile
    if include_predictor:
        data["predictor"] = getattr(predictor, "name", predictor.__class__.__name__)
    if include_regime:
        data["regime"] = regime
    if include_board:
        data["board"] = [classify_board(c) for c in sample["codes"]]
    df = pd.DataFrame(data).sort_values("rank").reset_index(drop=True)
    return df, regime


def generate_business_dates_inclusive(from_date, to_date):
    start = pd.Timestamp(from_date)
    end = pd.Timestamp(to_date)
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def generate_query_dates(from_date=None, to_date=None, ndates=5):
    if from_date and to_date:
        return generate_business_dates_inclusive(from_date, to_date)[-ndates:]
    if from_date:
        return [from_date]
    if to_date:
        return [to_date]
    today = pd.Timestamp.today()
    date_range = pd.date_range(today - pd.Timedelta(days=ndates * 3), today, freq="B")
    return [d.strftime("%Y-%m-%d") for d in date_range[-ndates:]]


def resolve_output_path(path, project_root):
    path = Path(path)
    return path if path.is_absolute() else Path(project_root) / path
