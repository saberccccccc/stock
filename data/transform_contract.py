"""Describe v14 transform and point-in-time assumptions without changing data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SCHEMA_VERSION = 1


def build_v14_transform_contract(meta: Mapping[str, Any], *, meta_path: str | Path) -> dict[str, Any]:
    """Build an auditable contract for an existing v14 cache metadata file.

    The cache stores daily cross-sectional normalized values. It does not store
    fitted global scaler parameters, because none are used for the v14 feature
    groups. This contract deliberately records remaining availability gaps
    rather than claiming complete source-level provenance.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(meta["all_dates"])).normalize()
    if dates.empty:
        raise ValueError("cache metadata has no dates")
    label_families = meta.get("label_families", {})
    if not isinstance(label_families, Mapping) or not label_families:
        raise ValueError("cache metadata has no label_families")
    source = Path(meta_path).resolve()
    stat = source.stat()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "v14_transform_contract",
        "cache_metadata": {
            "path": str(source),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "date_start": str(dates.min().date()),
            "date_end": str(dates.max().date()),
            "n_dates": int(len(dates)),
            "n_codes": int(len(meta.get("all_codes", []))),
            "x_dim": int(meta["x_dim"]),
            "risk_full_dim": int(meta["risk_full_dim"]),
            "feature_columns": list(meta.get("feature_cols", [])),
        },
        "feature_transforms": {
            "fit_scope": "daily_cross_section",
            "global_train_fitted_scaler": False,
            "aggregate_features": "per-date 1/99 percentile winsorization, z-score, clip[-4,4]",
            "rank_features": "per-date z-score, clip[-4,4]",
            "industry_relative_features": "per-date z-score, clip[-4,4]",
            "risk_continuous_features": "per-date z-score first six columns, clip[-4,4]",
            "missing_policy": "nan/positive_inf/negative_inf become zero after daily normalization",
            "implication": "reusing the cache does not reuse future-period global scaler statistics",
        },
        "labels": {
            "families": dict(label_families),
            "normalization": "per-date cross-sectional normalization by horizon",
            "execution_note": "oo_lag1 aliases oo with a one-trading-day label shift; task-level label-tail purge remains required",
        },
        "point_in_time_sources": {
            "fundamentals": {
                "availability": "effective_date derived from announcement dates when present; daily values forward-fill only after effective_date",
                "quality_flags": ["has_value", "days_since_effective", "is_fresh_quarter", "notice_is_estimated"],
            },
            "market_and_macro": {
                "availability": "rolling calculations are trailing; macro PMI uses next-month effective date and shifted expanding statistics",
                "required_follow_up": "source publication timestamps and external-market timezone lags must be covered by a source availability audit",
            },
            "execution": {
                "price_contract": "features may use adjusted series; formal execution must use raw OHLC through realistic open ledger",
                "required_follow_up": "corporate-action adjustment lineage is not retained in this cache metadata",
            },
        },
        "coverage_gaps": [
            "cache metadata does not contain raw-source file hashes",
            "cache metadata does not contain per-source publication timestamps",
            "historical universe/ST/listing coverage is owned by the execution-mask audit, not this feature cache",
        ],
        "status": "audited_with_declared_gaps",
    }


def write_transform_contract(contract: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path
