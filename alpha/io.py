"""JSONL helpers for daily alpha ranking files."""

import json
from pathlib import Path

import pandas as pd


def normalize_date(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def iter_alpha_rows(path, normalize_dates=True):
    """Yield non-empty JSONL rows from an alpha ranking file."""
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if normalize_dates and "date" in row:
                row["date"] = normalize_date(row["date"])
            yield row


def load_alpha_rows(path, timestamp_dates=False):
    rows = list(iter_alpha_rows(path, normalize_dates=not timestamp_dates))
    if timestamp_dates:
        for row in rows:
            row["date"] = pd.Timestamp(row["date"])
        rows.sort(key=lambda row: row["date"])
    return rows


def load_alpha_dates(path):
    dates = [pd.Timestamp(row["date"]).normalize() for row in iter_alpha_rows(path)]
    if not dates:
        raise ValueError("alpha file is empty")
    return sorted(set(dates))


def write_alpha_rows(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def assert_same_date(left, right):
    left_date = normalize_date(left["date"])
    right_date = normalize_date(right["date"])
    if left_date != right_date:
        raise ValueError(f"date mismatch: {left_date} vs {right_date}")
    return left_date
