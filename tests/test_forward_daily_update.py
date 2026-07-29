import json

import pandas as pd

from data.forward_daily_update import (
    DAILY_COLUMNS,
    DailyFileWriter,
    append_daily_row,
    last_trade_date,
    load_progress,
    repair_recent_tail,
)
from scripts.update_forward_stock_daily_one_date import (
    codes_with_at_most_rows,
    parse_args,
    read_code_filter,
    resolve_requested_dates,
)
from scripts.update_forward_market_data import parse_args as parse_market_args


def _write_daily(path, dates):
    pd.DataFrame(
        [
            {
                "trade_date": date,
                "code": "000001.SZ",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100.0,
                "money": 1000.0,
                "factor": 1.0,
            }
            for date in dates
        ],
        columns=DAILY_COLUMNS,
    ).to_csv(path, index=False)


def _row(date, close=12.5):
    return {
        "trade_date": pd.Timestamp(date),
        "code": "000001.SZ",
        "open": 12.0,
        "high": 13.0,
        "low": 11.0,
        "close": close,
        "volume": 120.0,
        "money": 1200.0,
        "factor": 1.0,
    }


def test_ordered_date_uses_fast_append_and_is_idempotent(tmp_path):
    path = tmp_path / "000001.SZ.csv"
    _write_daily(path, ["2026-06-29", "2026-06-30"])
    before = path.stat().st_size

    result = append_daily_row(path, _row("2026-07-01"), min_existing_rows=0)

    assert result.status == "appended"
    assert path.stat().st_size > before
    assert last_trade_date(path) == pd.Timestamp("2026-07-01")
    duplicate = append_daily_row(path, _row("2026-07-01"), min_existing_rows=0)
    assert duplicate.status == "already_present"
    assert len(pd.read_csv(path)) == 3


def test_historical_date_uses_atomic_backfill(tmp_path):
    path = tmp_path / "000001.SZ.csv"
    _write_daily(path, ["2026-06-29", "2026-07-01"])

    result = append_daily_row(path, _row("2026-06-30"), min_existing_rows=0)

    frame = pd.read_csv(path)
    assert result.status == "backfilled"
    assert frame["trade_date"].tolist() == ["2026-06-29", "2026-06-30", "2026-07-01"]


def test_historical_backfill_accepts_mixed_existing_date_formats(tmp_path):
    path = tmp_path / "000001.SZ.csv"
    _write_daily(path, ["20260629", "2026-07-01"])

    result = append_daily_row(path, _row("2026-06-30"), min_existing_rows=0)

    frame = pd.read_csv(path)
    assert result.status == "backfilled"
    assert frame["trade_date"].tolist() == ["2026-06-29", "2026-06-30", "2026-07-01"]


def test_recent_tail_repair_removes_duplicates_and_restores_order(tmp_path):
    path = tmp_path / "000001.SZ.csv"
    _write_daily(
        path,
        [
            "2026-06-30",
            "2026-07-01",
            "2026-07-02",
            "2026-07-02",
            "2026-07-03",
            "2026-07-01",
        ],
    )

    result = repair_recent_tail(path, "2026-07-01")

    frame = pd.read_csv(path)
    assert result.changed
    assert result.removed_rows == 2
    assert result.reordered
    assert frame["trade_date"].tolist() == [
        "2026-06-30",
        "2026-07-01",
        "2026-07-02",
        "2026-07-03",
    ]


def test_recent_tail_repair_does_not_touch_clean_file(tmp_path):
    path = tmp_path / "000001.SZ.csv"
    _write_daily(path, ["2026-06-30", "2026-07-01", "2026-07-02"])
    before = path.read_bytes()

    result = repair_recent_tail(path, "2026-07-01")

    assert not result.changed
    assert path.read_bytes() == before


def test_existing_historical_date_is_found_in_tail_without_rewrite(tmp_path):
    path = tmp_path / "000001.SZ.csv"
    _write_daily(path, ["2026-06-29", "2026-06-30", "2026-07-01"])
    before = path.read_bytes()

    result = append_daily_row(path, _row("2026-06-30", close=99.0), min_existing_rows=0)

    assert result.status == "already_present"
    assert path.read_bytes() == before


def test_multi_day_writer_keeps_tail_state_in_memory(tmp_path):
    path = tmp_path / "000001.SZ.csv"
    _write_daily(path, ["2026-06-30"])
    writer = DailyFileWriter(tmp_path)

    first = writer.write(_row("2026-07-01"))
    second = writer.write(_row("2026-07-02"))

    assert first.status == second.status == "appended"
    assert last_trade_date(path) == pd.Timestamp("2026-07-02")


def test_multi_day_writer_loads_existing_state_only_once(tmp_path, monkeypatch):
    writer = DailyFileWriter(tmp_path)
    original = writer._load_state
    calls = 0

    def counted_load_state(code):
        nonlocal calls
        calls += 1
        return original(code)

    monkeypatch.setattr(writer, "_load_state", counted_load_state)
    writer.write(_row("2026-07-01"))
    writer.write(_row("2026-07-02"))

    assert calls == 1


def test_progress_range_is_immutable(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(
        json.dumps({"start_date": "20260701", "end_date": "20260702"}),
        encoding="utf-8",
    )

    try:
        load_progress(path, start_date="20260701", end_date="20260703")
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected mismatched progress range to fail")


def test_range_uses_exchange_open_dates():
    args = parse_args(["--start-date", "2026-07-01", "--end-date", "2026-07-05"])

    class Client:
        def trade_dates(self, start, end):
            assert (start, end) == ("20260701", "20260705")
            return ["20260701", "20260702", "20260703"]

    start, end, dates = resolve_requested_dates(args, Client())
    assert (start, end) == ("20260701", "20260705")
    assert dates == ["20260701", "20260702", "20260703"]


def test_explicit_open_dates_bypass_exchange_calendar():
    args = parse_args(["--open-dates", "2026-07-02,20260701,2026-07-02"])

    class Client:
        def trade_dates(self, start, end):
            raise AssertionError("trade calendar must not be called")

    start, end, dates = resolve_requested_dates(args, Client())
    assert (start, end) == ("20260701", "20260702")
    assert dates == ["20260701", "20260702"]


def test_code_filter_normalizes_csv_suffix(tmp_path):
    path = tmp_path / "codes.txt"
    path.write_text("000001.SZ\n600000.SH.csv\n", encoding="utf-8")

    assert read_code_filter(path) == {"000001.SZ", "600000.SH"}


def test_short_history_filter_counts_data_rows_only(tmp_path):
    _write_daily(tmp_path / "000001.SZ.csv", ["2026-07-01"])
    _write_daily(
        tmp_path / "600000.SH.csv",
        ["2026-07-01", "2026-07-02", "2026-07-03"],
    )

    assert codes_with_at_most_rows(tmp_path, 2) == {"000001.SZ"}


def test_market_update_scope_can_select_broad_indices_only():
    assert parse_market_args(["--scope", "broad"]).scope == "broad"
