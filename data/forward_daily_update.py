"""Efficient, resumable writes for forward A-share daily bars."""

from __future__ import annotations

import csv
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


DAILY_COLUMNS = (
    "trade_date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "money",
    "factor",
)


@dataclass(frozen=True)
class AppendResult:
    status: str
    path: str


@dataclass
class _FileState:
    status: str
    last_date: pd.Timestamp | None


@dataclass(frozen=True)
class TailRepairResult:
    changed: bool
    removed_rows: int
    reordered: bool


def _last_nonempty_line(path: Path, read_size: int = 8192) -> str:
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        if size == 0:
            return ""
        handle.seek(-min(size, read_size), 2)
        lines = [line for line in handle.read().splitlines() if line.strip()]
    return lines[-1].decode("utf-8-sig", errors="strict") if lines else ""


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def _has_minimum_rows(path: Path, minimum: int) -> bool:
    if minimum <= 0:
        return True
    with path.open("rb") as handle:
        line_count = 0
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            line_count += chunk.count(b"\n")
            if line_count >= minimum + 1:
                return True
    return line_count >= minimum + 1


def last_trade_date(path: str | Path) -> pd.Timestamp | None:
    path = Path(path)
    if not path.is_file():
        return None
    line = _last_nonempty_line(path)
    if not line:
        return None
    row = next(csv.reader([line]), [])
    if not row:
        return None
    parsed = pd.to_datetime(row[0], errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _tail_contains_trade_date(
    path: Path, target: pd.Timestamp, read_size: int = 64 * 1024
) -> bool:
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        if size == 0:
            return False
        handle.seek(-min(size, read_size), 2)
        lines = handle.read().splitlines()
    target = pd.Timestamp(target).normalize()
    for raw in lines:
        if not raw.strip():
            continue
        row = next(csv.reader([raw.decode("utf-8-sig", errors="strict")]), [])
        if not row:
            continue
        value = pd.to_datetime(row[0], errors="coerce")
        if pd.notna(value) and pd.Timestamp(value).normalize() == target:
            return True
    return False


def _ordered_values(row: Mapping) -> list:
    values = []
    for column in DAILY_COLUMNS:
        value = row[column]
        if column == "trade_date":
            value = str(pd.Timestamp(value).normalize())
        values.append(value)
    return values


def _restore_tail_backup(path: Path, backup: Path) -> None:
    payload = backup.read_bytes()
    if len(payload) < 8:
        raise ValueError(f"invalid tail repair backup: {backup}")
    offset = struct.unpack(">Q", payload[:8])[0]
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(payload[8:])
        handle.truncate()
        handle.flush()
    backup.unlink()


def _trade_date_key(value: str) -> int | None:
    compact = value.strip()[:10].replace("-", "")
    if len(compact) >= 8 and compact[:8].isdigit():
        return int(compact[:8])
    return None


def repair_recent_tail(
    path: str | Path,
    cutoff: str | pd.Timestamp,
    *,
    read_size: int = 64 * 1024,
    durable: bool = False,
) -> TailRepairResult:
    """Deduplicate and order a recent CSV tail without rewriting old history."""

    path = Path(path)
    backup = path.with_suffix(path.suffix + ".tail-repair.bak")
    if backup.exists():
        _restore_tail_backup(path, backup)

    target = int(pd.Timestamp(cutoff).strftime("%Y%m%d"))
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        start = max(0, size - read_size)
        handle.seek(start)
        chunk = handle.read()
    if start:
        first_newline = chunk.find(b"\n")
        if first_newline < 0:
            raise ValueError(f"tail read window contains no complete row: {path}")
        start += first_newline + 1
        chunk = chunk[first_newline + 1 :]

    records: list[tuple[int, bytes]] = []
    first_offset = None
    cursor = start
    for raw_line in chunk.splitlines(keepends=True):
        line = raw_line.rstrip(b"\r\n")
        row = next(csv.reader([line.decode("utf-8-sig", errors="strict")]), [])
        date = _trade_date_key(row[0]) if row else None
        if date is not None and date >= target:
            if first_offset is None:
                first_offset = cursor
            records.append((date, raw_line))
        cursor += len(raw_line)

    if first_offset is None or not records:
        return TailRepairResult(False, 0, False)

    latest_by_date: dict[int, bytes] = {}
    observed_dates = []
    for date, raw_line in records:
        observed_dates.append(date)
        latest_by_date[date] = raw_line
    sorted_dates = sorted(latest_by_date)
    canonical = b"".join(latest_by_date[date] for date in sorted_dates)
    original = chunk[first_offset - start :]
    if canonical == original:
        return TailRepairResult(False, 0, False)

    backup.write_bytes(struct.pack(">Q", first_offset) + original)
    try:
        with path.open("r+b") as handle:
            handle.seek(first_offset)
            handle.write(canonical)
            handle.truncate()
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
    except BaseException:
        _restore_tail_backup(path, backup)
        raise
    backup.unlink()
    return TailRepairResult(
        True,
        len(records) - len(latest_by_date),
        observed_dates != sorted(observed_dates),
    )


def _backfill(path: Path, row: Mapping) -> None:
    existing = pd.read_csv(path)
    existing["trade_date"] = pd.to_datetime(
        existing["trade_date"], format="mixed", errors="coerce"
    )
    if existing["trade_date"].isna().any():
        raise ValueError(f"invalid trade_date values in {path}")
    incoming = pd.DataFrame([{column: row[column] for column in DAILY_COLUMNS}])
    incoming["trade_date"] = pd.to_datetime(incoming["trade_date"])
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    temp = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temp, index=False)
    temp.replace(path)


def append_daily_row(
    path: str | Path, row: Mapping, *, min_existing_rows: int = 200
) -> AppendResult:
    """Append an ordered new date, while retaining a safe historical backfill."""

    path = Path(path)
    if not path.is_file():
        return AppendResult("missing_local", str(path))
    if not _has_minimum_rows(path, min_existing_rows):
        return AppendResult("short_local", str(path))

    columns = _header(path)
    if columns != list(DAILY_COLUMNS):
        raise ValueError(f"unexpected daily CSV columns for {path}: {columns}")
    target = pd.Timestamp(row["trade_date"]).normalize()
    current = last_trade_date(path)
    if current is not None and target == current:
        return AppendResult("already_present", str(path))
    if current is not None and target < current:
        if _tail_contains_trade_date(path, target):
            return AppendResult("already_present", str(path))
        _backfill(path, row)
        return AppendResult("backfilled", str(path))

    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(_ordered_values(row))
    return AppendResult("appended", str(path))


class DailyFileWriter:
    """Cache per-file validation and tail dates across a multi-day update."""

    def __init__(self, data_dir: str | Path, *, min_existing_rows: int = 1):
        self.data_dir = Path(data_dir)
        self.min_existing_rows = int(min_existing_rows)
        self._states: dict[str, _FileState] = {}

    def _load_state(self, code: str) -> _FileState:
        path = self.data_dir / f"{code}.csv"
        if not path.is_file():
            return _FileState("missing_local", None)
        if not _has_minimum_rows(path, self.min_existing_rows):
            return _FileState("short_local", last_trade_date(path))
        columns = _header(path)
        if columns != list(DAILY_COLUMNS):
            raise ValueError(f"unexpected daily CSV columns for {path}: {columns}")
        return _FileState("ready", last_trade_date(path))

    def write(self, row: Mapping) -> AppendResult:
        code = str(row["code"])
        path = self.data_dir / f"{code}.csv"
        state = self._states.get(code)
        if state is None:
            state = self._load_state(code)
            self._states[code] = state
        if state.status != "ready":
            return AppendResult(state.status, str(path))

        target = pd.Timestamp(row["trade_date"]).normalize()
        if state.last_date is not None and target == state.last_date:
            return AppendResult("already_present", str(path))
        if state.last_date is not None and target < state.last_date:
            if _tail_contains_trade_date(path, target):
                return AppendResult("already_present", str(path))
            _backfill(path, row)
            return AppendResult("backfilled", str(path))

        with path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerow(_ordered_values(row))
        state.last_date = target
        return AppendResult("appended", str(path))


def load_progress(path: str | Path, *, start_date: str, end_date: str) -> dict:
    path = Path(path)
    if not path.is_file():
        return {
            "schema_version": 1,
            "start_date": start_date,
            "end_date": end_date,
            "status": "running",
            "completed_dates": [],
            "dates": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("start_date") != start_date or payload.get("end_date") != end_date:
        raise ValueError("progress range does not match the requested update range")
    return payload


def write_progress(path: str | Path, payload: Mapping) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)
    return path
