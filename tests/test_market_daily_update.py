import pandas as pd
import pytest

from data.market_daily_store import MarketDailyStore
from data.market_daily_update import (
    AKSHARE_BROAD_INDEX_SYMBOLS,
    FORMAL_BROAD_INDEX_CODES,
    run_incremental_update,
    update_progress_lock,
)


def _bars(codes, date="20260730", close=10.5):
    rows = []
    for index, code in enumerate(codes):
        row_close = close + index
        rows.append(
            {
                "ts_code": code,
                "trade_date": date,
                "open": row_close - 0.5,
                "high": row_close + 0.5,
                "low": row_close - 1.0,
                "close": row_close,
                "vol": 100.0 + index,
                "amount": 1000.0 + index,
            }
        )
    return pd.DataFrame(rows)


class FakeClient:
    equity_source = "fake.equity"
    index_source = "fake.index"
    index_money_semantics = "fake_amount"
    def __init__(self, *, close=10.5, fail_index=False):
        self.close = close
        self.fail_index = fail_index
        self.equity_calls = 0
        self.index_calls = 0

    def equity_daily(self, trade_date):
        self.equity_calls += 1
        return _bars(["000001.SZ", "600000.SH"], trade_date, self.close)

    def index_daily(self, trade_date, codes):
        self.index_calls += 1
        if self.fail_index:
            raise RuntimeError("simulated index outage")
        return _bars(codes, trade_date, self.close + 100.0)


def test_incremental_update_commits_equity_and_indices_and_resumes(tmp_path):
    store_root = tmp_path / "store"
    progress_path = tmp_path / "progress.json"
    client = FakeClient()

    first = run_incremental_update(
        client=client,
        store_root=store_root,
        dates=["2026-07-30"],
        progress_path=progress_path,
        min_equity_rows=2,
    )
    second = run_incremental_update(
        client=client,
        store_root=store_root,
        dates=["20260730"],
        progress_path=progress_path,
        min_equity_rows=2,
    )

    assert first["status"] == second["status"] == "completed"
    assert first["completed_dates"] == ["20260730"]
    assert first["coverage_alignment"]["latest_dates_aligned"] is True
    assert client.equity_calls == client.index_calls == 1
    audit = MarketDailyStore(store_root).audit()
    assert audit["coverage"]["equity"]["rows"] == 2
    assert audit["coverage"]["index"]["rows"] == 4
    assert audit["coverage"]["equity"]["date_end"] == "2026-07-30"
    assert audit["coverage"]["index"]["date_end"] == "2026-07-30"


def test_new_progress_replays_as_store_noop(tmp_path):
    store_root = tmp_path / "store"
    first = run_incremental_update(
        client=FakeClient(),
        store_root=store_root,
        dates=["20260730"],
        progress_path=tmp_path / "first.json",
        min_equity_rows=2,
    )
    second = run_incremental_update(
        client=FakeClient(),
        store_root=store_root,
        dates=["20260730"],
        progress_path=tmp_path / "second.json",
        min_equity_rows=2,
    )

    assert first["dates"]["20260730"]["equity"]["status"] == "committed"
    assert first["dates"]["20260730"]["index"]["status"] == "committed"
    assert second["dates"]["20260730"]["equity"]["status"] == "already_present"
    assert second["dates"]["20260730"]["index"]["status"] == "already_present"


def test_partial_failure_is_recorded_and_recovered(tmp_path):
    store_root = tmp_path / "store"
    progress_path = tmp_path / "progress.json"
    with pytest.raises(RuntimeError, match="simulated index outage"):
        run_incremental_update(
            client=FakeClient(fail_index=True),
            store_root=store_root,
            dates=["20260730"],
            progress_path=progress_path,
            min_equity_rows=2,
        )

    failed = pd.read_json(progress_path, typ="series")
    assert failed["status"] == "failed"
    assert MarketDailyStore(store_root).get_partition("equity", "20260730") is None

    recovered = run_incremental_update(
        client=FakeClient(),
        store_root=store_root,
        dates=["20260730"],
        progress_path=progress_path,
        min_equity_rows=2,
    )
    assert recovered["status"] == "completed"


def test_failure_after_equity_commit_recovers_missing_index(tmp_path, monkeypatch):
    store_root = tmp_path / "store"
    progress_path = tmp_path / "progress.json"
    original = MarketDailyStore.commit_partition
    failed_once = False

    def fail_index_once(self, frame, *, instrument_type, source, allow_revision=False):
        nonlocal failed_once
        if instrument_type == "index" and not failed_once:
            failed_once = True
            raise RuntimeError("simulated interruption after equity commit")
        return original(
            self,
            frame,
            instrument_type=instrument_type,
            source=source,
            allow_revision=allow_revision,
        )

    monkeypatch.setattr(MarketDailyStore, "commit_partition", fail_index_once)
    with pytest.raises(RuntimeError, match="after equity commit"):
        run_incremental_update(
            client=FakeClient(),
            store_root=store_root,
            dates=["20260730"],
            progress_path=progress_path,
            min_equity_rows=2,
        )
    assert MarketDailyStore(store_root).get_partition("equity", "20260730") is not None
    assert MarketDailyStore(store_root).get_partition("index", "20260730") is None

    monkeypatch.setattr(MarketDailyStore, "commit_partition", original)
    recovered = run_incremental_update(
        client=FakeClient(),
        store_root=store_root,
        dates=["20260730"],
        progress_path=progress_path,
        min_equity_rows=2,
    )
    assert recovered["dates"]["20260730"]["equity"]["status"] == "already_present"
    assert recovered["dates"]["20260730"]["index"]["status"] == "committed"


def test_revision_requires_explicit_permission(tmp_path):
    store_root = tmp_path / "store"
    run_incremental_update(
        client=FakeClient(close=10.5),
        store_root=store_root,
        dates=["20260730"],
        progress_path=tmp_path / "first.json",
        min_equity_rows=2,
    )

    with pytest.raises(ValueError, match="different content"):
        run_incremental_update(
            client=FakeClient(close=10.6),
            store_root=store_root,
            dates=["20260730"],
            progress_path=tmp_path / "second.json",
            min_equity_rows=2,
        )
    revised = run_incremental_update(
        client=FakeClient(close=10.6),
        store_root=store_root,
        dates=["20260730"],
        progress_path=tmp_path / "third.json",
        min_equity_rows=2,
        allow_revision=True,
    )
    assert revised["dates"]["20260730"]["equity"]["status"] == "revised"
    assert revised["dates"]["20260730"]["index"]["status"] == "revised"


def test_missing_broad_index_is_rejected_before_any_commit(tmp_path):
    class MissingIndexClient(FakeClient):
        def index_daily(self, trade_date, codes):
            return _bars(codes[:-1], trade_date, self.close + 100.0)

    store_root = tmp_path / "store"
    with pytest.raises(ValueError, match="coverage mismatch"):
        run_incremental_update(
            client=MissingIndexClient(),
            store_root=store_root,
            dates=["20260730"],
            progress_path=tmp_path / "progress.json",
            min_equity_rows=2,
        )
    assert not (store_root / "CURRENT").exists()


def test_progress_lock_rejects_concurrent_updater(tmp_path):
    progress = tmp_path / "progress.json"
    with update_progress_lock(progress):
        with pytest.raises(RuntimeError, match="another market-daily update"):
            with update_progress_lock(progress):
                raise AssertionError("second updater unexpectedly acquired the lock")


def test_formal_broad_index_set_is_frozen():
    assert FORMAL_BROAD_INDEX_CODES == (
        "000016.SH",
        "000300.SH",
        "000905.SH",
        "399006.SZ",
    )
    assert set(AKSHARE_BROAD_INDEX_SYMBOLS) == set(FORMAL_BROAD_INDEX_CODES)
