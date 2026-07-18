"""Point-in-time historical ST status event contract.

The execution layer needs the status that was effective on a trading date,
not the current name of a stock. This module owns the small, auditable data
contract used by the downloader and by ``open_ledger``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


EVENT_COLUMNS = [
    "ts_code",
    "name",
    "pub_date",
    "imp_date",
    "event_date",
    "st_type",
    "st_reason",
    "st_explain",
    "is_st",
    "source_endpoint",
    "source_fetched_at",
]
REQUIRED_EVENT_COLUMNS = {"ts_code", "imp_date", "event_date", "is_st"}


def normalize_ts_code(code):
    """Normalize common Tushare and local stock-code representations."""
    if code is None or pd.isna(code):
        return None
    text = str(code).strip()
    if not text or text.lower() == "nan":
        return None
    lower = text.lower()
    if lower.startswith(("sh.", "sz.", "bj.")):
        return f"{text[3:9]}.{lower[:2].upper()}"
    if "." in text:
        left, right = text.split(".", 1)
        if left.isdigit():
            return f"{left.zfill(6)}.{right.upper()}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        digits = digits[-6:]
        suffix = (
            "BJ"
            if digits[:2] in {"43", "83", "87", "88", "92"}
            else ("SH" if digits[:1] in {"5", "6", "9"} else "SZ")
        )
        return f"{digits}.{suffix}"
    return text


def parse_event_date(value):
    """Parse YYYYMMDD values without pandas' integer nanosecond ambiguity."""
    if value is None or pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) == 8:
        return pd.Timestamp(pd.to_datetime(text, format="%Y%m%d", errors="coerce")).normalize()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).normalize()


def _text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def derive_is_st(st_type="", st_reason="", st_explain=""):
    """Derive whether an event leaves the stock under an ST-like warning.

    Tushare's event feed contains both activation and removal events. The
    transition wording takes precedence over a raw ``st_type`` value so that
    phrases such as "撤销*ST并实行ST" remain active while "撤销*ST" becomes
    inactive. ``None`` means that the row is not safe to interpret.
    """
    type_text = _text(st_type)
    reason_text = _text(st_reason)
    explain_text = _text(st_explain)
    combined = " ".join(part for part in (type_text, reason_text, explain_text) if part)
    if not combined:
        return None

    activation_terms = (
        "实行ST",
        "实施ST",
        "实行*ST",
        "实施*ST",
        "叠加ST",
        "叠加*ST",
        "加入ST",
        "标记为ST",
        "高风险警示",
        "退市风险警示",
    )
    removal_terms = ("撤销", "撤消", "取消", "解除", "摘帽", "恢复")
    if any(term in combined for term in activation_terms):
        return True
    if any(term in combined for term in removal_terms):
        return False

    upper = combined.upper()
    if "*ST" in upper or "ST" in upper or "PT" in upper:
        return True
    if any(term in combined for term in ("风险警示", "退市整理", "退市风险")):
        return True
    return None


def derive_is_st_from_name(name):
    """Derive the status of a historical name interval from its displayed name."""
    text = _text(name).upper()
    if not text:
        return None
    return bool(
        "*ST" in text
        or "ST" in text
        or "PT" in text
        or "风险警示" in text
        or "退市整理" in text
    )


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def find_st_status_events_path(data_dir):
    """Find the nearest event file without crossing into another data root."""
    data_root = Path(data_dir)
    for path in (data_root / "st_status_events.csv", data_root.parent / "st_status_events.csv"):
        if path.is_file():
            return path
    return None


def find_st_status_manifest_path(data_dir):
    data_root = Path(data_dir)
    for path in (
        data_root / "st_status_events_manifest.json",
        data_root.parent / "st_status_events_manifest.json",
    ):
        if path.is_file():
            return path
    return None


def _empty_events():
    return pd.DataFrame(columns=EVENT_COLUMNS)


def normalize_st_events(
    frame,
    *,
    as_of_date=None,
    source_endpoint="st",
    source_fetched_at=None,
    strict=True,
):
    """Normalize raw ``st`` rows into the immutable event schema.

    Rows whose effective date is after ``as_of_date`` are excluded. Missing
    effective dates or undecidable status transitions are never guessed; in
    strict mode they raise, and in non-strict mode they are skipped and
    counted in ``DataFrame.attrs['invalid_row_count']``.
    """
    if frame is None:
        frame = pd.DataFrame()
    frame = pd.DataFrame(frame).copy()
    if frame.empty:
        result = _empty_events()
        result.attrs["source_row_count"] = 0
        result.attrs["invalid_row_count"] = 0
        return result
    if "ts_code" not in frame.columns:
        raise ValueError("ST event feed is missing required column: ts_code")
    if "imp_date" not in frame.columns:
        raise ValueError("ST event feed is missing required column: imp_date")

    cutoff = parse_event_date(as_of_date) if as_of_date is not None else None
    fetched_at = source_fetched_at or pd.Timestamp.now(tz="UTC").isoformat()
    rows = []
    invalid = []
    for source_row, row in frame.iterrows():
        code = normalize_ts_code(row.get("ts_code"))
        imp_date = parse_event_date(row.get("imp_date"))
        pub_date = parse_event_date(row.get("pub_date"))
        derived = derive_is_st(
            row.get("st_type"), row.get("st_reason"), row.get("st_explain")
        )
        supplied = _parse_bool(row.get("is_st")) if "is_st" in row else None
        if derived is None:
            derived = supplied
        elif supplied is not None and supplied != derived:
            invalid.append((source_row, "is_st_conflicts_with_event_text"))
            continue
        if not code or pd.isna(imp_date) or derived is None:
            invalid.append((source_row, "missing_code_effective_date_or_status"))
            continue
        if cutoff is not None and imp_date > cutoff:
            continue
        rows.append(
            {
                "ts_code": code,
                "name": _text(row.get("name")),
                "pub_date": pub_date.strftime("%Y-%m-%d") if not pd.isna(pub_date) else "",
                "imp_date": imp_date.strftime("%Y-%m-%d"),
                "event_date": imp_date.strftime("%Y-%m-%d"),
                "st_type": _text(row.get("st_type")),
                "st_reason": _text(row.get("st_reason")),
                "st_explain": _text(row.get("st_explain")),
                "is_st": bool(derived),
                "source_endpoint": source_endpoint,
                "source_fetched_at": str(fetched_at),
            }
        )

    if invalid and strict:
        preview = ", ".join(f"{row}:{reason}" for row, reason in invalid[:5])
        raise ValueError(f"invalid ST event rows ({len(invalid)}): {preview}")

    result = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if result.empty:
        result = _empty_events()
    else:
        result["_pub_sort"] = pd.to_datetime(result["pub_date"], errors="coerce")
        result["_event_sort"] = pd.to_datetime(result["event_date"], errors="coerce")
        result = (
            result.sort_values(["ts_code", "_event_sort", "_pub_sort"], kind="stable")
            .drop_duplicates(["ts_code", "event_date"], keep="last")
            .drop(columns=["_pub_sort", "_event_sort"])
            .reset_index(drop=True)
        )
    result.attrs["source_row_count"] = int(len(frame))
    result.attrs["invalid_row_count"] = int(len(invalid))
    result.attrs["invalid_rows"] = invalid[:20]
    return result


def normalize_namechange_events(
    frame,
    *,
    as_of_date=None,
    source_endpoint="tushare.namechange",
    source_fetched_at=None,
    strict=True,
):
    """Turn historical name intervals into dated ST state transitions.

    ``namechange`` is a fallback source, not an ST-event equivalent. Each
    interval contributes a transition on ``start_date``. An active interval
    also contributes an inactive transition on the day after ``end_date``
    when the next interval does not already define that state. Missing
    interval dates or names are rejected instead of guessed.
    """
    frame = pd.DataFrame(frame).copy() if frame is not None else pd.DataFrame()
    if frame.empty:
        result = _empty_events()
        result.attrs["source_row_count"] = 0
        result.attrs["invalid_row_count"] = 0
        return result
    required = {"ts_code", "name", "start_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"namechange feed is missing required columns: {missing}")

    cutoff = parse_event_date(as_of_date) if as_of_date is not None else None
    fetched_at = source_fetched_at or pd.Timestamp.now(tz="UTC").isoformat()
    intervals = []
    invalid = []
    for source_row, row in frame.iterrows():
        code = normalize_ts_code(row.get("ts_code"))
        name = _text(row.get("name"))
        start_date = parse_event_date(row.get("start_date"))
        end_date = parse_event_date(row.get("end_date"))
        status = derive_is_st_from_name(name)
        if not code or not name or pd.isna(start_date) or status is None:
            invalid.append((source_row, "missing_code_name_start_date_or_status"))
            continue
        if not pd.isna(end_date) and end_date < start_date:
            invalid.append((source_row, "end_date_before_start_date"))
            continue
        if cutoff is not None and start_date > cutoff:
            continue
        intervals.append(
            {
                "ts_code": code,
                "name": name,
                "pub_date": parse_event_date(row.get("ann_date")),
                "start_date": start_date,
                "end_date": end_date,
                "st_reason": _text(row.get("change_reason")),
                "is_st": bool(status),
            }
        )

    if invalid and strict:
        preview = ", ".join(f"{row}:{reason}" for row, reason in invalid[:5])
        raise ValueError(f"invalid namechange rows ({len(invalid)}): {preview}")

    intervals.sort(key=lambda item: (item["ts_code"], item["start_date"]))
    rows = []
    for position, interval in enumerate(intervals):
        rows.append(
            {
                "ts_code": interval["ts_code"],
                "name": interval["name"],
                "pub_date": interval["pub_date"].strftime("%Y-%m-%d") if not pd.isna(interval["pub_date"]) else "",
                "imp_date": interval["start_date"].strftime("%Y-%m-%d"),
                "event_date": interval["start_date"].strftime("%Y-%m-%d"),
                "st_type": "NAMECHANGE_INTERVAL",
                "st_reason": interval["st_reason"],
                "st_explain": "status derived from historical displayed name",
                "is_st": interval["is_st"],
                "source_endpoint": source_endpoint,
                "source_fetched_at": str(fetched_at),
            }
        )
        next_interval = intervals[position + 1] if position + 1 < len(intervals) else None
        if not interval["is_st"] or pd.isna(interval["end_date"]):
            continue
        clear_date = interval["end_date"] + pd.Timedelta(days=1)
        if next_interval is not None and next_interval["ts_code"] == interval["ts_code"] and next_interval["start_date"] <= clear_date:
            continue
        if cutoff is not None and clear_date > cutoff:
            continue
        rows.append(
            {
                "ts_code": interval["ts_code"],
                "name": interval["name"],
                "pub_date": interval["pub_date"].strftime("%Y-%m-%d") if not pd.isna(interval["pub_date"]) else "",
                "imp_date": clear_date.strftime("%Y-%m-%d"),
                "event_date": clear_date.strftime("%Y-%m-%d"),
                "st_type": "NAMECHANGE_INTERVAL_END",
                "st_reason": "historical name interval ended",
                "st_explain": "status reset after historical displayed name interval",
                "is_st": False,
                "source_endpoint": source_endpoint,
                "source_fetched_at": str(fetched_at),
            }
        )

    result = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if result.empty:
        result = _empty_events()
    else:
        result["_event_sort"] = pd.to_datetime(result["event_date"], errors="coerce")
        result = (
            result.sort_values(["ts_code", "_event_sort"], kind="stable")
            .drop_duplicates(["ts_code", "event_date"], keep="last")
            .drop(columns=["_event_sort"])
            .reset_index(drop=True)
        )
    result.attrs["source_row_count"] = int(len(frame))
    result.attrs["invalid_row_count"] = int(len(invalid))
    result.attrs["invalid_rows"] = invalid[:20]
    return result


def validate_st_event_frame(frame):
    """Return contract facts; malformed rows are never treated as false."""
    frame = pd.DataFrame(frame)
    missing = sorted(REQUIRED_EVENT_COLUMNS - set(frame.columns))
    malformed_dates = 0
    malformed_event_dates = 0
    malformed_codes = 0
    malformed_status = 0
    if not missing:
        malformed_dates = int(frame["imp_date"].map(parse_event_date).isna().sum())
        malformed_event_dates = int(frame["event_date"].map(parse_event_date).isna().sum())
        malformed_codes = int(frame["ts_code"].map(normalize_ts_code).isna().sum())
        malformed_status = int(frame["is_st"].map(_parse_bool).isna().sum())
    return {
        "required_columns_present": not missing,
        "missing_columns": missing,
        "row_count": int(len(frame)),
        "code_count": int(frame["ts_code"].nunique()) if "ts_code" in frame else 0,
        "malformed_date_rows": malformed_dates,
        "malformed_event_date_rows": malformed_event_dates,
        "malformed_code_rows": malformed_codes,
        "malformed_status_rows": malformed_status,
        "valid": (
            not missing
            and malformed_dates == 0
            and malformed_event_dates == 0
            and malformed_codes == 0
            and malformed_status == 0
        ),
    }


def load_st_status_events(data_dir):
    """Load normalized events as ``(effective_date, code, is_st)`` tuples."""
    path = find_st_status_events_path(data_dir)
    if path is None:
        return []
    frame = None
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            break
        except (OSError, UnicodeDecodeError, pd.errors.ParserError):
            continue
    if frame is None:
        raise ValueError(f"unable to read ST event file: {path}")
    facts = validate_st_event_frame(frame)
    if not facts["valid"]:
        raise ValueError(f"invalid ST event file {path}: {facts}")
    event_col = "event_date" if "event_date" in frame.columns else "imp_date"
    events = []
    for _, row in frame.iterrows():
        date = parse_event_date(row[event_col])
        code = normalize_ts_code(row["ts_code"])
        status = _parse_bool(row["is_st"])
        if code and not pd.isna(date) and status is not None:
            events.append((date, code, bool(status)))
    return events


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_st_manifest(data_dir):
    path = find_st_status_manifest_path(data_dir)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
