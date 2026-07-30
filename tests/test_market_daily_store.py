import json
from pathlib import Path

import pandas as pd
import pytest

from data.market_daily_store import (
    MarketDailyStore,
    _writer_lock,
    validate_market_daily_frame,
)


def _frame(date="2026-07-29", close=10.5):
    return pd.DataFrame(
        [
            {
                "trade_date": date,
                "code": "600000.SH",
                "open": 10.1,
                "high": 10.8,
                "low": 10.0,
                "close": close,
                "volume": 100.0,
                "money": 1000.0,
                "factor": 1.0,
            },
            {
                "trade_date": date,
                "code": "000001.SZ",
                "open": 11.1,
                "high": 11.8,
                "low": 11.0,
                "close": 11.5,
                "volume": 200.0,
                "money": 2000.0,
                "factor": 1.0,
            },
        ]
    )


def test_commit_is_content_addressed_and_idempotent(tmp_path):
    store = MarketDailyStore(tmp_path)

    first = store.commit_partition(
        _frame(), instrument_type="equity", source="tushare_daily"
    )
    second = store.commit_partition(
        _frame().iloc[::-1], instrument_type="equity", source="tushare_daily"
    )

    assert first.status == "committed"
    assert second.status == "already_present"
    assert first.logical_sha256 == second.logical_sha256
    assert len(list(tmp_path.rglob("part-*.parquet"))) == 1
    assert len((tmp_path / "update_events.jsonl").read_text().splitlines()) == 1
    manifest, _ = store.load_manifest()
    assert manifest["coverage"]["equity"] == {
        "date_start": "2026-07-29",
        "date_end": "2026-07-29",
        "months": 1,
        "partitions": 1,
        "rows": 2,
    }
    assert list(manifest["monthly_indexes"]) == ["equity:202607"]


def test_revision_requires_explicit_permission_and_preserves_old_file(tmp_path):
    store = MarketDailyStore(tmp_path)
    first = store.commit_partition(
        _frame(), instrument_type="equity", source="tushare_daily"
    )

    with pytest.raises(ValueError, match="different content"):
        store.commit_partition(
            _frame(close=10.6),
            instrument_type="equity",
            source="tushare_daily",
        )
    revised = store.commit_partition(
        _frame(close=10.6),
        instrument_type="equity",
        source="tushare_daily",
        allow_revision=True,
    )

    assert revised.status == "revised"
    assert len(list(tmp_path.rglob("part-*.parquet"))) == 2
    assert first.partition_path != revised.partition_path
    loaded = store.load(
        instrument_type="equity",
        start_date="2026-07-29",
        end_date="2026-07-29",
        fields=["close"],
    )
    assert loaded.loc[loaded["code"] == "600000.SH", "close"].iloc[0] == 10.6


def test_failed_current_switch_leaves_previous_manifest_active(tmp_path, monkeypatch):
    store = MarketDailyStore(tmp_path)
    first = store.commit_partition(
        _frame(), instrument_type="equity", source="tushare_daily"
    )
    pointer_before = (tmp_path / "CURRENT").read_bytes()

    def fail_switch(*args, **kwargs):
        raise RuntimeError("simulated pointer failure")

    monkeypatch.setattr(store, "_write_current", fail_switch)
    with pytest.raises(RuntimeError, match="simulated"):
        store.commit_partition(
            _frame(close=10.6),
            instrument_type="equity",
            source="tushare_daily",
            allow_revision=True,
        )

    assert (tmp_path / "CURRENT").read_bytes() == pointer_before
    manifest, path = store.load_manifest()
    assert str(path) == first.manifest_path
    assert manifest["generation"] == 1
    assert not list((tmp_path / "staging").glob("*.parquet"))


def test_validation_rejects_duplicates_and_invalid_ohlc():
    duplicate = pd.concat([_frame().iloc[[0]], _frame().iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_market_daily_frame(
            duplicate,
            instrument_type="equity",
            source="tushare_daily",
        )

    invalid = _frame()
    invalid.loc[0, "high"] = 9.0
    with pytest.raises(ValueError, match="high"):
        validate_market_daily_frame(
            invalid,
            instrument_type="equity",
            source="tushare_daily",
        )


def test_current_pointer_and_manifest_hash_are_verified(tmp_path):
    store = MarketDailyStore(tmp_path)
    store.commit_partition(
        _frame(), instrument_type="equity", source="tushare_daily"
    )
    pointer = json.loads((tmp_path / "CURRENT").read_text())
    manifest_path = tmp_path / pointer["manifest"]
    manifest_path.write_text("{}")

    with pytest.raises(ValueError, match="hash mismatch"):
        store.load_manifest()


def test_load_deduplicates_key_fields(tmp_path):
    store = MarketDailyStore(tmp_path)
    store.commit_partition(
        _frame(), instrument_type="equity", source="tushare_daily"
    )

    loaded = store.load(
        instrument_type="equity",
        start_date="2026-07-29",
        end_date="2026-07-29",
        fields=["code", "trade_date", "close"],
    )

    assert loaded.columns.tolist() == ["trade_date", "code", "close"]


def test_root_manifest_stays_month_sharded(tmp_path):
    store = MarketDailyStore(tmp_path)
    store.commit_partition(
        _frame("2026-07-29"), instrument_type="equity", source="tushare_daily"
    )
    store.commit_partition(
        _frame("2026-07-30"), instrument_type="equity", source="tushare_daily"
    )

    manifest, _ = store.load_manifest()
    assert list(manifest["monthly_indexes"]) == ["equity:202607"]
    month_record = manifest["monthly_indexes"]["equity:202607"]
    assert month_record["partitions"] == 2
    assert month_record["row_count"] == 4
    assert len(list((tmp_path / "indexes").rglob("index-*.json"))) == 2


def test_writer_lock_rejects_a_second_writer(tmp_path):
    lock_path = tmp_path / "locks" / "writer.lock"

    with _writer_lock(lock_path):
        with pytest.raises(RuntimeError, match="already active"):
            with _writer_lock(lock_path):
                raise AssertionError("second writer unexpectedly acquired the lock")


def test_audit_validates_active_manifest_graph(tmp_path):
    store = MarketDailyStore(tmp_path)
    store.commit_partition(
        _frame("2026-07-29"), instrument_type="equity", source="tushare_daily"
    )
    store.commit_partition(
        _frame("2026-07-30"), instrument_type="equity", source="tushare_daily"
    )

    result = store.audit()

    assert result["status"] == "passed"
    assert result["active_months"] == 1
    assert result["active_partitions"] == 2
    assert result["active_rows"] == 4
    assert result["physical_hashes_verified"] is True


def test_audit_rejects_corrupt_active_partition(tmp_path):
    store = MarketDailyStore(tmp_path)
    committed = store.commit_partition(
        _frame(), instrument_type="equity", source="tushare_daily"
    )
    path = Path(committed.partition_path)
    path.write_bytes(path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="partition hash mismatch"):
        store.audit()


def test_audit_rejects_corrupt_month_index(tmp_path):
    store = MarketDailyStore(tmp_path)
    store.commit_partition(
        _frame(), instrument_type="equity", source="tushare_daily"
    )
    manifest, _ = store.load_manifest()
    record = manifest["monthly_indexes"]["equity:202607"]
    path = tmp_path / record["path"]
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="month index hash mismatch"):
        store.audit()
