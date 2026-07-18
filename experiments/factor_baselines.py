"""PIT-compatible price/volume baseline specifications inspired by Alpha158."""

from __future__ import annotations

from copy import deepcopy


BASELINES = {
    "alpha158_compact_price_volume_v1": {
        "label_family": "oo_lag1",
        "source_contract": "v14_transform_contract",
        "description": "Compact A-share price/volume baseline without fundamental, macro, industry, or external features.",
        "features": (
            "ret_5d", "ret_20d", "vol_10d", "vol_60d", "price_momentum",
            "log_volume", "volume_spike", "gap", "amplitude", "sma5_gap",
            "sma20_gap", "rsi_norm", "macd_pct", "atr_pct", "volume_ratio",
        ),
    },
    "alpha158_broad_price_volume_v1": {
        "label_family": "oo_lag1",
        "source_contract": "v14_transform_contract",
        "description": "Broader A-share price/volume baseline, still excluding all non-price/volume feature groups.",
        "features": (
            "ret_5d", "ret_20d", "vol_10d", "vol_60d", "price_momentum",
            "log_volume", "volume_spike", "upper_shadow", "lower_shadow", "body_size",
            "gap", "amplitude", "sma5_gap", "sma10_gap", "sma20_gap", "ema12_gap",
            "ema26_gap", "rsi_norm", "macd_pct", "macd_signal_pct", "macd_diff_pct",
            "atr_pct", "volume_ratio",
        ),
    },
}


def get_factor_baseline(name):
    if name not in BASELINES:
        raise KeyError(f"unknown factor baseline: {name}")
    return deepcopy(BASELINES[name])


def audit_factor_baselines(feature_columns):
    available = set(feature_columns)
    result = {}
    for name, spec in BASELINES.items():
        features = list(spec["features"])
        missing = sorted(set(features) - available)
        forbidden = sorted(
            feature for feature in features
            if feature.startswith(("fund_", "sh_", "restr_", "macro_", "global_"))
        )
        result[name] = {
            **spec,
            "features": features,
            "feature_count": len(features),
            "missing_features": missing,
            "forbidden_non_price_volume_features": forbidden,
            "ready": not missing and not forbidden,
        }
    return result
