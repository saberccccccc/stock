"""Shared close-based market-state classification for research utilities."""

from pathlib import Path

import pandas as pd


def load_index_states(path, ma_window=60, crash_ret=-0.03):
    """Return date, state, and index return using information known at close."""

    ma_window = int(ma_window)
    if ma_window <= 0:
        raise ValueError("ma_window must be positive")

    frame = pd.read_csv(Path(path))
    if "date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "date"})
    close_col = "close" if "close" in frame.columns else frame.columns[-1]
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date")
    close = pd.to_numeric(frame[close_col], errors="coerce")
    min_periods = min(ma_window, max(5, ma_window // 3))
    moving_average = close.rolling(ma_window, min_periods=min_periods).mean()
    daily_return = close.pct_change()

    states = []
    for date, value, average, ret in zip(
        frame["date"], close, moving_average, daily_return
    ):
        state = "normal"
        if pd.notna(value) and pd.notna(average) and value < average:
            state = "bear"
        if pd.notna(ret) and float(ret) <= float(crash_ret):
            state = "crash"
        states.append(
            {
                "date": pd.Timestamp(date).normalize(),
                "market_state": state,
                "index_ret": float(ret) if pd.notna(ret) else 0.0,
            }
        )
    return pd.DataFrame(states)
