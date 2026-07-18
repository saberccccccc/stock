"""Audit the historical evidence behind realistic open-ledger constraints."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.st_status import (
    file_sha256,
    find_st_status_events_path,
    read_st_manifest,
    validate_st_event_frame,
)


def _read_csv(path: Path):
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError):
            continue
    return None


def _date_range(values):
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return {"count": 0, "start": None, "end": None}
    dates = pd.DatetimeIndex(dates).normalize().unique().sort_values()
    return {"count": int(len(dates)), "start": str(dates[0].date()), "end": str(dates[-1].date())}


def _manifest_date(manifest, key):
    if not manifest or not manifest.get(key):
        return None
    parsed = pd.to_datetime(manifest.get(key), errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _audit_historical_st_source(data_root, start, end, *, dataset_role="research"):
    event_path = find_st_status_events_path(data_root)
    if event_path is not None:
        event_frame = _read_csv(event_path)
        facts = validate_st_event_frame(event_frame) if event_frame is not None else {
            "required_columns_present": False,
            "missing_columns": ["unreadable_file"],
            "row_count": 0,
            "code_count": 0,
            "malformed_date_rows": 0,
            "malformed_event_date_rows": 0,
            "malformed_code_rows": 0,
            "malformed_status_rows": 0,
            "valid": False,
        }
        event_col = "event_date" if event_frame is not None and "event_date" in event_frame.columns else "imp_date"
        event_dates = _date_range(event_frame[event_col]) if event_frame is not None and event_col in event_frame.columns else _date_range([])
        manifest = read_st_manifest(data_root)
        manifest_role = (manifest or {}).get("dataset_role", "research")
        role_matches = manifest_role == dataset_role
        manifest_start = _manifest_date(manifest, "coverage_start")
        manifest_end = _manifest_date(manifest, "coverage_end")
        hash_matches = bool(
            manifest
            and manifest.get("output_sha256")
            and manifest.get("output_sha256") == file_sha256(event_path)
        )
        covers = bool(
            facts["valid"]
            and int((manifest or {}).get("invalid_row_count", 0)) == 0
            and manifest_start is not None
            and manifest_end is not None
            and manifest_start <= start
            and manifest_end >= end
            and hash_matches
            and role_matches
        )
        return {
            "path": str(event_path.resolve()),
            "source_type": (manifest or {}).get("source_kind", "event_history"),
            "source_endpoint": (manifest or {}).get("source_endpoint"),
            "source_label": (manifest or {}).get("source_label"),
            "dataset_role": manifest_role,
            "requested_dataset_role": dataset_role,
            "dataset_role_matches": role_matches,
            "required_columns_present": facts["required_columns_present"],
            "required_columns_missing": facts["missing_columns"],
            "event_date_coverage": event_dates,
            "source_coverage": {
                "start": manifest.get("coverage_start") if manifest else None,
                "end": manifest.get("coverage_end") if manifest else None,
            },
            "row_count": facts["row_count"],
            "code_count": facts["code_count"],
            "malformed_date_rows": facts["malformed_date_rows"],
            "malformed_event_date_rows": facts["malformed_event_date_rows"],
            "malformed_code_rows": facts["malformed_code_rows"],
            "malformed_status_rows": facts["malformed_status_rows"],
            "manifest_present": manifest is not None,
            "manifest_hash_matches": hash_matches,
            "covers_requested_interval": covers,
            "note": (
                "Historical ST status is reconstructed from imp_date events. "
                "The manifest coverage range is required so an event file ending "
                "before the requested interval cannot look complete."
            ),
        }

    snapshot_path = next(
        (path for path in (data_root / "stock_industry.csv", data_root.parent / "stock_industry.csv") if path.exists()),
        None,
    )
    snapshot = _read_csv(snapshot_path) if snapshot_path else None
    snapshot_dates = _date_range(snapshot["updateDate"]) if snapshot is not None and "updateDate" in snapshot else _date_range([])
    required = bool(snapshot is not None and {"updateDate", "code", "code_name"}.issubset(snapshot.columns))
    return {
        "path": str(snapshot_path.resolve()) if snapshot_path else None,
        "source_type": "current_snapshot" if snapshot_path else None,
        "required_columns_present": required,
        "snapshot_date_coverage": snapshot_dates,
        "covers_requested_interval": False,
        "note": (
            "A current-name/single-snapshot fallback is not historical ST evidence. "
            "Do not label a report as historical-ST-complete until dated status events cover the interval."
        ),
    }


def audit_execution_coverage(
    data_dir, matrix_cache_dir, *, start_date, end_date, dataset_role="research"
):
    """Return coverage facts and explicit blockers for a research interval."""
    data_root = Path(data_dir)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    stable_path = data_root / "stable_stocks.csv"
    stable = _read_csv(stable_path) if stable_path.exists() else None
    list_dates = stable.get("list_date", pd.Series(dtype=object)) if stable is not None else pd.Series(dtype=object)
    matrix_path = Path(matrix_cache_dir) / "ohlc_matrix_meta.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.is_file() else {}
    matrix_dates = _date_range(matrix.get("dates", []))

    historical_st = _audit_historical_st_source(
        data_root, start, end, dataset_role=dataset_role
    )
    st_history_covers_interval = bool(historical_st["covers_requested_interval"])

    gaps = []
    if stable is None or "ts_code" not in stable.columns:
        gaps.append("missing_stock_master")
    elif "list_date" not in stable.columns or int(pd.to_datetime(list_dates, errors="coerce").notna().sum()) < len(stable):
        gaps.append("incomplete_listing_dates")
    if not matrix or matrix_dates["start"] is None or pd.Timestamp(matrix_dates["start"]) > start or pd.Timestamp(matrix_dates["end"]) < end:
        gaps.append("ohlc_matrix_does_not_cover_requested_interval")
    if not st_history_covers_interval:
        gaps.append("historical_st_status_not_covered")

    return {
        "schema_version": 1,
        "dataset_role": dataset_role,
        "requested_interval": {"start": str(start.date()), "end": str(end.date())},
        "stock_master": {
            "path": str(stable_path.resolve()),
            "rows": int(len(stable)) if stable is not None else 0,
            "listing_date_count": int(pd.to_datetime(list_dates, errors="coerce").notna().sum()),
        },
        "ohlc_matrix": {
            "path": str(matrix_path.resolve()),
            "source_count": int(matrix.get("source_count", 0)),
            "fields": list(matrix.get("fields", [])),
            "date_coverage": matrix_dates,
        },
        "historical_st": {
            **historical_st,
            "covers_requested_interval": st_history_covers_interval,
        },
        "gaps": gaps,
        "status": "audited_with_declared_gaps" if gaps else "coverage_complete",
    }
