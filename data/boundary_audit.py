"""Audit physical market-cache dates against an effective dataset role.

The project may keep a physical cache that is a superset of the research
period to avoid duplicating large CSV files.  That is safe only when every
research consumer supplies an effective end date no later than the frozen
cutoff.  This module records both facts instead of treating filesystem
timestamps as a data-boundary guarantee.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from core.research_protocol import (
    FORWARD_DATA_DIR,
    FORWARD_START_DATE,
    RESEARCH_DATA_DIR,
    RESEARCH_END_DATE,
)


DATE_COLUMNS = ("trade_date", "date", "交易日期", "datetime")
STOCK_FILE_PREFIXES = tuple("0123456789")


def resolve_data_dir(data_dir=None, dataset_role="research") -> Path:
    """Resolve the default physical cache root for a dataset role."""
    if dataset_role not in {"research", "forward"}:
        raise ValueError(f"unknown dataset_role: {dataset_role}")
    if data_dir:
        return Path(data_dir)
    return Path(RESEARCH_DATA_DIR if dataset_role == "research" else FORWARD_DATA_DIR)


def _decode_line(raw: bytes) -> str:
    return raw.decode("utf-8-sig", errors="replace")


def _last_nonempty_line(path: Path) -> bytes:
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        if size <= 0:
            return b""
        read_size = min(size, 1024 * 1024)
        handle.seek(-read_size, 2)
        lines = [line for line in handle.read(read_size).splitlines() if line.strip()]
    return lines[-1] if lines else b""


def _parse_row(raw: bytes) -> list[str]:
    if not raw:
        return []
    return next(csv.reader([_decode_line(raw)]), [])


def inspect_market_csv(path: str | Path) -> dict:
    """Read only header, first data row, and tail row from one CSV."""
    path = Path(path)
    with path.open("rb") as handle:
        header = _parse_row(handle.readline())
        first_data = handle.readline()
    date_column = next((column for column in DATE_COLUMNS if column in header), None)
    if date_column is None:
        return {
            "path": str(path),
            "date_column": None,
            "first_date": None,
            "last_date": None,
            "status": "missing_date_column",
        }
    date_index = header.index(date_column)
    first_row = _parse_row(first_data)
    last_row = _parse_row(_last_nonempty_line(path))
    if len(first_row) <= date_index or len(last_row) <= date_index:
        return {
            "path": str(path),
            "date_column": date_column,
            "first_date": None,
            "last_date": None,
            "status": "missing_data_row",
        }
    first = pd.to_datetime(first_row[date_index], errors="coerce")
    last = pd.to_datetime(last_row[date_index], errors="coerce")
    if pd.isna(first) or pd.isna(last):
        return {
            "path": str(path),
            "date_column": date_column,
            "first_date": None if pd.isna(first) else str(pd.Timestamp(first).date()),
            "last_date": None if pd.isna(last) else str(pd.Timestamp(last).date()),
            "status": "invalid_date_row",
        }
    return {
        "path": str(path),
        "date_column": date_column,
        "first_date": str(pd.Timestamp(first).date()),
        "last_date": str(pd.Timestamp(last).date()),
        "status": "ok",
        "boundary_method": "first_and_last_rows_assuming_chronological_rows",
    }


def iter_market_csvs(data_dir: str | Path):
    root = Path(data_dir)
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.glob("*.csv")
        if path.is_file() and path.name and path.name[0] in STOCK_FILE_PREFIXES
    )


def audit_data_root(
    data_dir: str | Path | None = None,
    *,
    dataset_role: str = "research",
    effective_end_date=None,
    research_cutoff=RESEARCH_END_DATE,
) -> dict:
    """Return physical and effective date-boundary evidence for one cache."""
    root = resolve_data_dir(data_dir, dataset_role)
    cutoff = pd.Timestamp(research_cutoff).normalize()
    if effective_end_date is None:
        effective = cutoff if dataset_role == "research" else None
    else:
        effective = pd.Timestamp(effective_end_date).normalize()
    if dataset_role == "research" and effective is not None and effective > cutoff:
        raise ValueError(
            f"research effective end {effective.date()} exceeds cutoff {cutoff.date()}"
        )
    if dataset_role == "forward" and effective is not None and effective < FORWARD_START_DATE:
        raise ValueError(
            f"forward effective end {effective.date()} is before forward start "
            f"{FORWARD_START_DATE.date()}"
        )

    files = iter_market_csvs(root)
    inspected = [inspect_market_csv(path) for path in files]
    valid = [item for item in inspected if item["status"] == "ok"]
    last_dates = [pd.Timestamp(item["last_date"]) for item in valid]
    first_dates = [pd.Timestamp(item["first_date"]) for item in valid]
    future_files = (
        [item for item in valid if pd.Timestamp(item["last_date"]) > cutoff]
        if dataset_role == "research"
        else []
    )
    effective_safe = (
        dataset_role == "research"
        and effective is not None
        and effective <= cutoff
    ) or (
        dataset_role == "forward"
        and (effective is None or effective >= FORWARD_START_DATE)
    )
    physical_clean = not future_files if dataset_role == "research" else True
    if not root.is_dir():
        status = "missing_root"
    elif not valid:
        status = "no_valid_market_csv"
    elif dataset_role == "research" and future_files:
        status = "runtime_cutoff_required"
    elif effective_safe:
        status = "effective_view_safe"
    else:
        status = "effective_view_invalid"

    return {
        "schema_version": 1,
        "dataset_role": dataset_role,
        "data_dir": str(root),
        "research_cutoff": str(cutoff.date()),
        "forward_start": str(FORWARD_START_DATE.date()),
        "effective_end_date": str(effective.date()) if effective is not None else None,
        "effective_view_safe": bool(effective_safe),
        "physical_cache_clean": bool(physical_clean),
        "status": status,
        "boundary_method": "first_and_last_rows_assuming_chronological_rows",
        "file_count": len(files),
        "valid_date_file_count": len(valid),
        "invalid_or_unreadable_file_count": len(inspected) - len(valid),
        "physical_date_coverage": {
            "start": str(min(first_dates).date()) if first_dates else None,
            "end": str(max(last_dates).date()) if last_dates else None,
        },
        "research_cutoff_violation_count": len(future_files),
        "research_cutoff_violation_sample": future_files[:20],
        "note": (
            "A physical research-cache superset is not selection-safe by itself. "
            "Every research consumer must apply effective_end_date <= research_cutoff."
            if dataset_role == "research"
            else "Forward observations are isolated from research selection."
        ),
    }
