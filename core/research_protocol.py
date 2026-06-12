"""Shared research/forward-test boundaries.

Research data is frozen through 2026-05-18. Later observations are reserved
for forward testing and must not enter training, validation, or model selection.
"""

from pathlib import Path

import pandas as pd


RESEARCH_END_DATE = pd.Timestamp("2026-05-18")
FORWARD_START_DATE = pd.Timestamp("2026-05-19")
RESEARCH_DATA_DIR = Path("data/raw")
FORWARD_DATA_DIR = Path("data/forward_raw")
SMALL_ACCOUNT_VALUES = (500_000.0, 1_000_000.0)


def research_end_date_str() -> str:
    return RESEARCH_END_DATE.strftime("%Y-%m-%d")


def assert_research_end_date(value, context: str = "research") -> pd.Timestamp:
    end_date = pd.Timestamp(value) if value is not None else RESEARCH_END_DATE
    if end_date > RESEARCH_END_DATE:
        raise ValueError(
            f"{context} end date {end_date.date()} exceeds frozen research boundary "
            f"{RESEARCH_END_DATE.date()}; later data is reserved for forward testing."
        )
    return end_date


def assert_alpha_rows_within_research(rows, context: str = "backtest") -> None:
    dates = [pd.Timestamp(row["date"]) for row in rows]
    if dates and max(dates) > RESEARCH_END_DATE:
        raise ValueError(
            f"{context} contains signal date {max(dates).date()} after frozen research "
            f"boundary {RESEARCH_END_DATE.date()}."
        )


def assert_alpha_rows_within_forward(rows, context: str = "forward test") -> None:
    dates = [pd.Timestamp(row["date"]) for row in rows]
    if dates and min(dates) < FORWARD_START_DATE:
        raise ValueError(
            f"{context} contains signal date {min(dates).date()} before forward-test "
            f"start {FORWARD_START_DATE.date()}."
        )


def cached_dates_within_research(metadata) -> bool:
    dates = metadata.get("all_dates", [])
    return not dates or pd.Timestamp(max(dates)) <= RESEARCH_END_DATE
