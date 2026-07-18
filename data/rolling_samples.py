"""Streaming v14 memmap samples for rolling research experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.pipeline import _open_memmap


SENTINEL = np.int16(-32768)
SCALE = 1000.0


def resolve_label_view(meta, label_family):
    families = meta.get("label_families", {})
    if label_family not in families:
        raise ValueError(f"unknown label family: {label_family}")
    family = families[label_family]
    if family.get("alias_of"):
        family = families[family["alias_of"]]
    return Path(family["norm_path"]), int(families[label_family].get("date_shift", 0))


def resolve_label_family(meta, label_family):
    families = meta.get("label_families", {})
    if label_family not in families:
        raise ValueError(f"unknown label family: {label_family}")
    declared = families[label_family]
    base = families[declared["alias_of"]] if declared.get("alias_of") else declared
    return (
        Path(base["norm_path"]),
        Path(base["raw_path"]),
        int(declared.get("date_shift", 0)),
    )


def iter_rolling_samples(
    meta,
    time_indices,
    label_family,
    horizon_index,
    feature_indices=None,
    *,
    include_risk=True,
    include_industry=True,
):
    """Yield one label-safe cross section at a time without materializing all dates."""
    n_stocks, n_dates = len(meta["all_codes"]), len(meta["all_dates"])
    horizon_index = int(horizon_index)
    if horizon_index < 0 or horizon_index >= int(meta["max_horizon"]):
        raise ValueError("horizon_index is outside the cached label horizon")
    label_path, label_shift = resolve_label_view(meta, label_family)
    if feature_indices is not None:
        feature_indices = np.asarray(feature_indices, dtype=np.intp)
        if feature_indices.ndim != 1 or len(feature_indices) == 0:
            raise ValueError("feature_indices must be a non-empty one-dimensional sequence")
        if feature_indices.min() < 0 or feature_indices.max() >= int(meta["x_dim"]):
            raise ValueError("feature_indices are outside the cached feature dimension")
    x_mm = _open_memmap(meta["x_norm_path"], np.int16, (n_stocks, n_dates, meta["x_dim"]))
    r_mm = _open_memmap(meta["risk_full_path"], np.int16, (n_stocks, n_dates, meta["risk_full_dim"]))
    y_mm = _open_memmap(label_path, np.int16, (n_stocks, n_dates, meta["max_horizon"]))
    codes = np.asarray(meta["all_codes"], dtype=object)
    industry = meta["industry_array"]
    try:
        for time_index in time_indices:
            label_index = int(time_index) + label_shift
            if label_index >= n_dates:
                continue
            valid = (x_mm[:, time_index, 0] != SENTINEL) & (y_mm[:, label_index, horizon_index] != SENTINEL)
            if int(valid.sum()) < int(meta.get("min_stocks", 30)):
                continue
            selected = np.flatnonzero(valid)
            result = {
                "time_index": int(time_index),
                "date": meta["all_dates"][time_index],
                "codes": codes[selected].tolist(),
                "X": x_mm[selected, time_index, :][:, feature_indices].astype(np.float32) / SCALE
                if feature_indices is not None
                else x_mm[selected, time_index, :].astype(np.float32) / SCALE,
                "y": y_mm[selected, label_index, horizon_index].astype(np.float32) / SCALE,
            }
            if include_risk:
                result["risk"] = r_mm[selected, time_index, :].astype(np.float32) / SCALE
            if include_industry:
                result["industry_ids"] = industry[selected, time_index].astype(np.int64)
            yield result
    finally:
        for matrix in (x_mm, r_mm, y_mm):
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None:
                mmap.close()


def iter_strong_rolling_samples(
    meta,
    time_indices,
    label_family,
    *,
    horizon_indices,
    target_horizon_index,
    include_raw_returns,
    include_lag1_labels,
    lag1_label_family=None,
):
    """Yield the exact multi-label fields consumed by the strong trainer."""

    n_stocks, n_dates = len(meta["all_codes"]), len(meta["all_dates"])
    max_horizon = int(meta["max_horizon"])
    horizons = tuple(int(value) for value in horizon_indices)
    target = int(target_horizon_index)
    required = tuple(sorted(set(horizons) | {target}))
    if not required or required[0] < 0 or required[-1] >= max_horizon:
        raise ValueError("strong label horizons are outside the cached horizon")
    norm_path, raw_path, label_shift = resolve_label_family(meta, label_family)
    if include_lag1_labels:
        if lag1_label_family:
            lag1_norm_path, _, lag1_shift = resolve_label_family(meta, lag1_label_family)
        else:
            lag1_norm_path, lag1_shift = norm_path, label_shift + 1
    else:
        lag1_norm_path = None
        lag1_shift = None
    x_mm = _open_memmap(meta["x_norm_path"], np.int16, (n_stocks, n_dates, meta["x_dim"]))
    r_mm = _open_memmap(
        meta["risk_full_path"], np.int16, (n_stocks, n_dates, meta["risk_full_dim"])
    )
    y_mm = _open_memmap(norm_path, np.int16, (n_stocks, n_dates, max_horizon))
    raw_mm = (
        _open_memmap(raw_path, np.float32, (n_stocks, n_dates, max_horizon))
        if include_raw_returns
        else None
    )
    lag1_mm = (
        _open_memmap(lag1_norm_path, np.int16, (n_stocks, n_dates, max_horizon))
        if lag1_norm_path is not None
        else None
    )
    codes = np.asarray(meta["all_codes"], dtype=object)
    industry = meta["industry_array"]
    matrices = [x_mm, r_mm, y_mm, raw_mm, lag1_mm]
    try:
        for raw_index in time_indices:
            time_index = int(raw_index)
            label_index = time_index + label_shift
            if label_index >= n_dates:
                continue
            valid = np.all(y_mm[:, label_index, :][:, required] != SENTINEL, axis=1)
            if int(valid.sum()) < int(meta.get("min_stocks", 30)):
                continue
            selected = np.flatnonzero(valid)
            y_seq = y_mm[selected, label_index, :].astype(np.float32) / SCALE
            result = {
                "time_index": time_index,
                "date": meta["all_dates"][time_index],
                "codes": codes[selected].tolist(),
                "X": x_mm[selected, time_index, :].astype(np.float32) / SCALE,
                "y": y_seq[:, target].copy(),
                "y_seq": y_seq,
                "risk": r_mm[selected, time_index, :].astype(np.float32) / SCALE,
                "industry_ids": industry[selected, time_index].astype(np.int64),
            }
            if raw_mm is not None:
                result["raw_y_seq"] = raw_mm[selected, label_index, :].astype(np.float32)
            if lag1_mm is not None:
                lag1_values = np.zeros((len(selected), max_horizon), dtype=np.float32)
                lag1_valid = np.zeros(len(selected), dtype=bool)
                lag1_index = time_index + int(lag1_shift)
                if lag1_index < n_dates:
                    lag1_valid = np.all(
                        lag1_mm[selected, lag1_index, :][:, required] != SENTINEL,
                        axis=1,
                    )
                    if lag1_valid.any():
                        lag1_values[lag1_valid] = (
                            lag1_mm[selected[lag1_valid], lag1_index, :].astype(np.float32) / SCALE
                        )
                result["lag1_y_seq"] = lag1_values
                result["lag1_mask"] = lag1_valid
            yield result
    finally:
        seen = set()
        for matrix in matrices:
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None and id(mmap) not in seen:
                seen.add(id(mmap))
                mmap.close()


def iter_v14_inference_samples(meta, time_indices):
    """Yield label-free v14 samples using the established inference universe."""

    n_stocks, n_dates = len(meta["all_codes"]), len(meta["all_dates"])
    x_mm = _open_memmap(meta["x_norm_path"], np.int16, (n_stocks, n_dates, meta["x_dim"]))
    r_mm = _open_memmap(
        meta["risk_full_path"], np.int16, (n_stocks, n_dates, meta["risk_full_dim"])
    )
    codes = np.asarray(meta["all_codes"], dtype=object)
    industry = meta["industry_array"]
    try:
        for raw_index in time_indices:
            time_index = int(raw_index)
            valid = (x_mm[:, time_index, 0] != SENTINEL) & (r_mm[:, time_index, 0] != SENTINEL)
            if int(valid.sum()) < int(meta.get("min_stocks", 30)):
                continue
            selected = np.flatnonzero(valid)
            yield {
                "time_index": time_index,
                "date": meta["all_dates"][time_index],
                "codes": codes[selected].tolist(),
                "X": x_mm[selected, time_index, :].astype(np.float32) / SCALE,
                "risk": r_mm[selected, time_index, :].astype(np.float32) / SCALE,
                "industry_ids": industry[selected, time_index].astype(np.int64),
            }
    finally:
        for matrix in (x_mm, r_mm):
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None:
                mmap.close()
