"""Qlib-style rolling-window contracts for the frozen research dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha.io import load_alpha_rows, write_alpha_rows
from core.research_protocol import RESEARCH_END_DATE, SPLIT_SPECS


@dataclass(frozen=True)
class RollingWindow:
    name: str
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    predict_start: str
    predict_end: str

    @classmethod
    def from_mapping(cls, value):
        return cls(**{field: str(value[field]) for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class MonthlyRollingSpec:
    """Fixed-length walk-forward schedule over the supplied trading calendar."""

    train_years: int
    valid_months: int
    oos_start: str
    oos_end: str

    def __post_init__(self):
        if int(self.train_years) <= 0:
            raise ValueError("train_years must be positive")
        if int(self.valid_months) <= 0:
            raise ValueError("valid_months must be positive")


@dataclass(frozen=True)
class FixedRollingSpec:
    """Fixed Train/Valid windows with a configurable non-overlapping OOS span."""

    train_years: int
    valid_months: int
    oos_months: int
    oos_start: str
    oos_end: str

    def __post_init__(self):
        if int(self.train_years) <= 0:
            raise ValueError("train_years must be positive")
        if int(self.valid_months) <= 0:
            raise ValueError("valid_months must be positive")
        if int(self.oos_months) <= 0:
            raise ValueError("oos_months must be positive")


def _date(value):
    return pd.Timestamp(value).normalize()


def validate_window(window, research_end=RESEARCH_END_DATE):
    train_start, train_end = _date(window.train_start), _date(window.train_end)
    valid_start, valid_end = _date(window.valid_start), _date(window.valid_end)
    predict_start, predict_end = _date(window.predict_start), _date(window.predict_end)
    if not (train_start <= train_end < valid_start <= valid_end < predict_start <= predict_end):
        raise ValueError(f"rolling window {window.name} has overlapping or unordered segments")
    if predict_end > _date(research_end):
        raise ValueError(f"rolling window {window.name} exceeds frozen research boundary")


def label_safe_indices(dates, start, end, label_end_offset):
    """Select signal dates whose complete future label is available by ``end``."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    start, end = _date(start), _date(end)
    offset = int(label_end_offset)
    if offset < 0:
        raise ValueError("label_end_offset must be non-negative")
    selected = []
    for index, signal_date in enumerate(dates):
        if signal_date < start or signal_date > end:
            continue
        label_index = index + offset
        if label_index < len(dates) and dates[label_index] <= end:
            selected.append(index)
    return selected


def prediction_indices(dates, start, end):
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    start, end = _date(start), _date(end)
    return [index for index, value in enumerate(dates) if start <= value <= end]


def resolve_window_indices(dates, window, label_end_offset):
    validate_window(window)
    return {
        "train": label_safe_indices(dates, window.train_start, window.train_end, label_end_offset),
        "valid": label_safe_indices(dates, window.valid_start, window.valid_end, label_end_offset),
        "predict": prediction_indices(dates, window.predict_start, window.predict_end),
    }


def _calendar_dates(dates):
    calendar = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    if calendar.empty:
        raise ValueError("trading calendar is empty")
    return calendar


def _segment_dates(calendar, start, end):
    return calendar[(calendar >= _date(start)) & (calendar <= _date(end))]


def assert_unique_oos_owners(dates, windows):
    """Reject overlapping OOS windows so every signal date has one model owner."""
    calendar = _calendar_dates(dates)
    owners = {}
    for window in windows:
        for value in _segment_dates(calendar, window.predict_start, window.predict_end):
            key = str(value.date())
            if key in owners:
                raise ValueError(f"duplicate OOS owner for {key}: {owners[key]} and {window.name}")
            owners[key] = window.name
    return owners


def build_split_alpha_files(window_entries, output_dir):
    """Stitch learner-neutral rolling outputs into official split artifacts."""

    rows_by_date = {}
    for entry in window_entries:
        alpha_path = entry.get("alpha_path")
        if not alpha_path:
            raise ValueError(f"window {entry.get('name')} has no alpha_path")
        for row in load_alpha_rows(alpha_path):
            key = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            if key in rows_by_date:
                raise ValueError(f"duplicate stitched OOS alpha owner for {key}")
            rows_by_date[key] = {**row, "date": key, "owner_window": entry.get("name")}
    result = {}
    for split, split_spec in SPLIT_SPECS.items():
        selected = [
            rows_by_date[key]
            for key in sorted(rows_by_date)
            if split_spec.start <= pd.Timestamp(key) <= split_spec.end
        ]
        if not selected:
            continue
        path = Path(output_dir) / "signals" / split / "alpha_policy.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_alpha_rows(path, selected)
        result[split] = {
            "path": str(path.resolve()),
            "rows": len(selected),
            "signal_start": selected[0]["date"],
            "signal_end": selected[-1]["date"],
        }
    return result


def build_monthly_windows(dates, spec, research_end=RESEARCH_END_DATE):
    """Build fixed Train/Valid/one-month-OOS windows from actual trading dates."""
    fixed = FixedRollingSpec(
        train_years=spec.train_years,
        valid_months=spec.valid_months,
        oos_months=1,
        oos_start=spec.oos_start,
        oos_end=spec.oos_end,
    )
    return build_fixed_windows(dates, fixed, research_end=research_end)


def build_fixed_windows(dates, spec, research_end=RESEARCH_END_DATE):
    """Build fixed Train/Valid/OOS windows with exact trading-date ownership."""

    calendar = _calendar_dates(dates)
    start, end = _date(spec.oos_start), _date(spec.oos_end)
    if start > end:
        raise ValueError("oos_start must not exceed oos_end")
    if end > _date(research_end):
        raise ValueError("monthly OOS schedule exceeds frozen research boundary")

    windows = []
    cursor = start.to_period("M").start_time.normalize()
    while cursor <= end:
        period_start = cursor
        period_end = (period_start + pd.DateOffset(months=int(spec.oos_months))) - pd.Timedelta(days=1)
        oos_dates = _segment_dates(calendar, max(start, period_start), min(end, period_end))
        if oos_dates.empty:
            cursor = period_start + pd.DateOffset(months=int(spec.oos_months))
            continue
        valid_start = (period_start - pd.DateOffset(months=int(spec.valid_months))).normalize()
        valid_dates = _segment_dates(calendar, valid_start, period_start - pd.Timedelta(days=1))
        train_start = (valid_start - pd.DateOffset(years=int(spec.train_years))).normalize()
        train_dates = _segment_dates(calendar, train_start, valid_start - pd.Timedelta(days=1))
        if train_dates.empty or valid_dates.empty:
            raise ValueError(
                f"rolling window {period_start:%Y-%m} lacks required train/valid trading dates "
                f"for {spec.train_years}y train and {spec.valid_months}m valid"
            )
        suffix = (
            f"{oos_dates[0]:%Y_%m}"
            if int(spec.oos_months) == 1
            else f"{oos_dates[0]:%Y_%m}_{oos_dates[-1]:%Y_%m}"
        )
        window = RollingWindow(
            name=f"oos_{suffix}",
            train_start=str(train_dates[0].date()),
            train_end=str(train_dates[-1].date()),
            valid_start=str(valid_dates[0].date()),
            valid_end=str(valid_dates[-1].date()),
            predict_start=str(oos_dates[0].date()),
            predict_end=str(oos_dates[-1].date()),
        )
        validate_window(window, research_end=research_end)
        windows.append(window)
        cursor = period_start + pd.DateOffset(months=int(spec.oos_months))
    if not windows:
        raise ValueError("rolling OOS schedule has no trading-date windows")
    assert_unique_oos_owners(calendar, windows)
    return windows
