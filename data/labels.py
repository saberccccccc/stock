"""Forward-return label definitions aligned to the market trading calendar."""

from __future__ import annotations

from typing import Dict

import numpy as np


PHYSICAL_LABEL_FAMILIES = ("cc", "oc", "oo")
LABEL_FAMILIES = PHYSICAL_LABEL_FAMILIES + ("oo_lag1",)


def build_forward_return_labels(open_prices, close_prices, max_horizon: int) -> Dict[str, np.ndarray]:
    """Build close/open forward returns for one calendar-aligned stock series.

    Row ``t`` is the signal date. Horizon ``h`` is one-based:
    cc: close[t+h] / close[t] - 1
    oc: close[t+h] / open[t+1] - 1
    oo: open[t+h+1] / open[t+1] - 1

    ``oo_lag1`` is intentionally not materialized because it is exactly
    ``oo[t+1]``. Consumers expose it through a one-row view.
    """
    if not isinstance(max_horizon, int) or max_horizon < 1:
        raise ValueError("max_horizon must be a positive integer")

    open_arr = np.asarray(open_prices, dtype=np.float32)
    close_arr = np.asarray(close_prices, dtype=np.float32)
    if open_arr.ndim != 1 or close_arr.ndim != 1 or open_arr.shape != close_arr.shape:
        raise ValueError("open_prices and close_prices must be equally sized 1-D arrays")

    n_dates = open_arr.shape[0]
    labels = {
        family: np.full((n_dates, max_horizon), np.nan, dtype=np.float32)
        for family in PHYSICAL_LABEL_FAMILIES
    }

    for horizon in range(1, max_horizon + 1):
        cc_count = n_dates - horizon
        if cc_count > 0:
            cc_base = close_arr[:cc_count]
            cc_end = close_arr[horizon:horizon + cc_count]
            oc_base = open_arr[1:1 + cc_count]
            with np.errstate(divide="ignore", invalid="ignore"):
                labels["cc"][:cc_count, horizon - 1] = cc_end / cc_base - 1.0
                labels["oc"][:cc_count, horizon - 1] = cc_end / oc_base - 1.0

        oo_count = n_dates - horizon - 1
        if oo_count > 0:
            oo_base = open_arr[1:1 + oo_count]
            oo_end = open_arr[horizon + 1:horizon + 1 + oo_count]
            with np.errstate(divide="ignore", invalid="ignore"):
                labels["oo"][:oo_count, horizon - 1] = oo_end / oo_base - 1.0

    for values in labels.values():
        values[~np.isfinite(values)] = np.nan
    return labels


def label_end_offset(label_family: str, horizon_index: int) -> int:
    """Return the last future calendar-row offset consumed by a label."""
    if label_family not in LABEL_FAMILIES:
        raise ValueError(f"unknown label family: {label_family}")
    if not isinstance(horizon_index, int) or horizon_index < 0:
        raise ValueError("horizon_index must be a non-negative integer")
    horizon = horizon_index + 1
    if label_family in ("cc", "oc"):
        return horizon
    if label_family == "oo":
        return horizon + 1
    return horizon + 2
