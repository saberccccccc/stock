"""Canonical research, selection, and forward-observation split contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    """Immutable date and governance role for one official evaluation split."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    max_data_date: pd.Timestamp
    role: str
    selection_eligible: bool

    @property
    def is_forward(self) -> bool:
        return self.role == "forward_observation"

    def command_dates(self) -> tuple[str, str, str]:
        return tuple(
            value.strftime("%Y-%m-%d")
            for value in (self.start, self.end, self.max_data_date)
        )


# The selection record ends with calendar year 2025. Full-year 2026 is a
# forward observation period and may extend as new complete market dates arrive.
RESEARCH_END_DATE = pd.Timestamp("2025-12-31")
FORWARD_START_DATE = pd.Timestamp("2026-01-01")
FORWARD_END_DATE = pd.Timestamp("2026-06-30")

# Some historical caches and reports were materialized through this date. It is
# artifact metadata only and must never be used as the forward split boundary.
LEGACY_CACHE_SNAPSHOT_DATE = pd.Timestamp("2026-05-18")

RESEARCH_DATA_DIR = Path("data/raw")
FORWARD_DATA_DIR = Path("data/forward_raw")
SMALL_ACCOUNT_VALUES = (500_000.0, 1_000_000.0)

SPLIT_SPECS = MappingProxyType(
    {
        "val_2024": SplitSpec(
            name="val_2024",
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-12-31"),
            max_data_date=pd.Timestamp("2024-12-31"),
            role="selection",
            selection_eligible=True,
        ),
        "test_2025": SplitSpec(
            name="test_2025",
            start=pd.Timestamp("2025-01-01"),
            end=RESEARCH_END_DATE,
            max_data_date=RESEARCH_END_DATE,
            role="selection",
            selection_eligible=True,
        ),
        "forward_2026": SplitSpec(
            name="forward_2026",
            start=FORWARD_START_DATE,
            end=FORWARD_END_DATE,
            max_data_date=FORWARD_END_DATE,
            role="forward_observation",
            selection_eligible=False,
        ),
    }
)
SELECTION_SPLITS = tuple(name for name, spec in SPLIT_SPECS.items() if spec.selection_eligible)
OBSERVATION_SPLITS = tuple(name for name, spec in SPLIT_SPECS.items() if spec.is_forward)


def get_split_spec(name: str) -> SplitSpec:
    try:
        return SPLIT_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown official split: {name}") from exc


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} must be a boolean, got {value!r}")


def validate_report_role(split: str, *, selection_eligible: Any, is_forward: Any) -> SplitSpec:
    """Reject registry rows whose flags contradict the canonical split role."""

    spec = get_split_spec(split)
    actual_selection = _as_bool(selection_eligible, field="selection_eligible")
    actual_forward = _as_bool(is_forward, field="is_forward")
    if actual_selection != spec.selection_eligible:
        raise ValueError(
            f"split={split} requires selection_eligible={spec.selection_eligible}, "
            f"got {actual_selection}"
        )
    if actual_forward != spec.is_forward:
        raise ValueError(f"split={split} requires is_forward={spec.is_forward}, got {actual_forward}")
    return spec


def validate_result_dates(
    split: str,
    *,
    signal_start: Any,
    signal_end: Any,
    backtest_start: Any,
    backtest_end: Any,
    max_signal_lead_days: int = 10,
) -> SplitSpec:
    """Cross-check actual signal and ledger dates against an official split.

    A signal may precede the evaluation start by a few calendar days because an
    open-price decision can consume the previous trading day's score. Ledger
    dates themselves must remain inside the declared evaluation interval.
    """

    spec = get_split_spec(split)
    raw = {
        "signal_start": signal_start,
        "signal_end": signal_end,
        "backtest_start": backtest_start,
        "backtest_end": backtest_end,
    }
    parsed = {}
    for field, value in raw.items():
        if value is None or str(value).strip() in {"", "nan", "NaT"}:
            raise ValueError(f"split={split} requires result field {field}")
        try:
            parsed[field] = pd.Timestamp(value).normalize()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"split={split} has invalid {field}={value!r}") from exc

    if parsed["signal_start"] > parsed["signal_end"]:
        raise ValueError(f"split={split} signal_start exceeds signal_end")
    if parsed["backtest_start"] > parsed["backtest_end"]:
        raise ValueError(f"split={split} backtest_start exceeds backtest_end")

    earliest_signal = spec.start - pd.Timedelta(days=max_signal_lead_days)
    if parsed["signal_start"] < earliest_signal:
        raise ValueError(
            f"split={split} signal_start {parsed['signal_start'].date()} is earlier than "
            f"the allowed prior-signal window {earliest_signal.date()}"
        )
    if parsed["signal_end"] < spec.start or parsed["signal_end"] > spec.end:
        raise ValueError(
            f"split={split} signal_end {parsed['signal_end'].date()} falls outside "
            f"{spec.start.date()}..{spec.end.date()}"
        )
    if parsed["backtest_start"] < spec.start or parsed["backtest_end"] > spec.end:
        raise ValueError(
            f"split={split} backtest interval "
            f"{parsed['backtest_start'].date()}..{parsed['backtest_end'].date()} falls outside "
            f"{spec.start.date()}..{spec.end.date()}"
        )
    if max(parsed.values()) > spec.max_data_date:
        raise ValueError(f"split={split} result exceeds max_data_date={spec.max_data_date.date()}")
    return spec


def assert_forward_parent_frozen(parent_fit_end: Any, parent_selection_end: Any) -> None:
    """Ensure a full-year 2026 forward artifact was frozen before 2026 began."""

    for field, value in (
        ("parent_fit_end", parent_fit_end),
        ("parent_selection_end", parent_selection_end),
    ):
        if value is None or str(value).strip() == "":
            raise ValueError(f"{field} is required for full-year forward lineage")
        date = pd.Timestamp(value).normalize()
        if date > RESEARCH_END_DATE:
            raise ValueError(
                f"{field} {date.date()} exceeds full-year forward freeze deadline "
                f"{RESEARCH_END_DATE.date()}"
            )


def research_end_date_str() -> str:
    return RESEARCH_END_DATE.strftime("%Y-%m-%d")


def assert_research_end_date(value, context: str = "research") -> pd.Timestamp:
    end_date = pd.Timestamp(value) if value is not None else RESEARCH_END_DATE
    if end_date > RESEARCH_END_DATE:
        raise ValueError(
            f"{context} end date {end_date.date()} exceeds selection boundary "
            f"{RESEARCH_END_DATE.date()}; 2026 data is forward observation only."
        )
    return end_date


def resolve_market_data_end_date(value=None, allow_forward: bool = False):
    """Resolve a market-data ceiling without weakening the selection boundary."""

    if allow_forward:
        return pd.Timestamp(value) if value is not None else None
    return assert_research_end_date(value, context="research market data")


def assert_alpha_rows_within_research(rows, context: str = "backtest") -> None:
    assert_alpha_dates_within_research([row["date"] for row in rows], context=context)


def assert_alpha_dates_within_research(dates, context: str = "backtest") -> None:
    dates = [pd.Timestamp(date) for date in dates]
    if dates and max(dates) > RESEARCH_END_DATE:
        raise ValueError(
            f"{context} contains signal date {max(dates).date()} after selection "
            f"boundary {RESEARCH_END_DATE.date()}."
        )


def assert_alpha_rows_within_forward(rows, context: str = "forward test") -> None:
    assert_alpha_dates_within_forward([row["date"] for row in rows], context=context)


def assert_alpha_dates_within_forward(dates, context: str = "forward test") -> None:
    dates = [pd.Timestamp(date) for date in dates]
    if dates and min(dates) < FORWARD_START_DATE:
        raise ValueError(
            f"{context} contains signal date {min(dates).date()} before full-year "
            f"forward start {FORWARD_START_DATE.date()}."
        )


def cached_dates_within_research(metadata, end_date=None) -> bool:
    dates = metadata.get("all_dates", [])
    cutoff = assert_research_end_date(end_date, context="research cache")
    return not dates or pd.Timestamp(max(dates)) <= cutoff
