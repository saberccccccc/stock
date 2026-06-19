"""Reusable diagnostics for Alpha rank stability and chase risk."""

import numpy as np
import pandas as pd


def execution_quality_daily(name, rows, signal_returns, top_n=30):
    top_n = int(top_n)
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    daily = []
    previous = set()
    for row in rows:
        selected = list(row.get("codes", []))[:top_n]
        returns = [
            signal_returns.get(pd.Timestamp(row["date"]), {}).get(code)
            for code in selected
        ]
        returns = np.asarray(
            [value for value in returns if value is not None],
            dtype=float,
        )
        selected_set = set(selected)
        daily.append(
            {
                "name": name,
                "date": pd.Timestamp(row["date"]),
                "mean_signal_return": (
                    float(np.mean(returns)) if len(returns) else np.nan
                ),
                "median_signal_return": (
                    float(np.median(returns)) if len(returns) else np.nan
                ),
                "share_ge_070": (
                    float(np.mean(returns >= 0.07)) if len(returns) else np.nan
                ),
                "share_ge_095": (
                    float(np.mean(returns >= 0.095)) if len(returns) else np.nan
                ),
                "overlap_previous": (
                    len(selected_set & previous) / max(len(selected_set), 1)
                    if previous
                    else np.nan
                ),
            }
        )
        previous = selected_set
    return pd.DataFrame(daily)


def summarize_execution_quality(daily):
    columns = [
        "name",
        "dates",
        "mean_signal_return",
        "median_signal_return",
        "share_ge_070",
        "share_ge_095",
        "overlap_previous",
    ]
    if daily.empty:
        return pd.DataFrame(columns=columns)
    return (
        daily.groupby("name", sort=False)
        .agg(
            dates=("date", "count"),
            mean_signal_return=("mean_signal_return", "mean"),
            median_signal_return=("median_signal_return", "mean"),
            share_ge_070=("share_ge_070", "mean"),
            share_ge_095=("share_ge_095", "mean"),
            overlap_previous=("overlap_previous", "mean"),
        )
        .reset_index()
    )
