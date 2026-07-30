import json
from types import SimpleNamespace

import pandas as pd
import pytest

from backtest.monthly_ohlcv_cache import MonthlyOhlcvCache, _cache_lock
from backtest.open_ledger import (
    load_execution_market_frames,
    prepare_execution_constraint_masks,
)
from data.market_daily_store import MarketDailyStore


def _frame(date, close=10.5):
    return pd.DataFrame(
        [
            {
                "trade_date": date,
                "code": code,
                "open": close - 0.5 + index,
                "high": close + 0.5 + index,
                "low": close - 1.0 + index,
                "close": close + index,
                "volume": 100.0 + index,
                "money": 1000.0 + index,
                "factor": 1.0,
            }
            for index, code in enumerate(("000001.SZ", "600000.SH"))
        ]
    )


def test_month_cache_build_load_and_idempotency(tmp_path):
    store_root = tmp_path / "store"
    cache_root = tmp_path / "cache"
    store = MarketDailyStore(store_root)
    store.commit_partition(_frame("2026-01-30"), instrument_type="equity", source="csv")
    store.commit_partition(_frame("2026-02-02", 11.5), instrument_type="equity", source="csv")
    cache = MonthlyOhlcvCache(store_root=store_root, cache_root=cache_root)

    first = cache.ensure_month("2026-01")
    second = cache.ensure_month("2026-01")
    frames = cache.load(
        codes=["600000.SH", "000001.SZ"],
        fields=[
            "open",
            "close",
            "money",
            "pre_close",
            "pct_chg",
            "valid_ohlc_mask",
            "zero_volume_mask",
            "basic_open_tradable_mask",
        ],
        start_date="2026-01-30",
        end_date="2026-02-02",
        money_scale=0.001,
    )

    assert first["status"] == "rebuilt"
    assert second["status"] == "already_current"
    assert frames["close"].index.tolist() == [
        pd.Timestamp("2026-01-30"),
        pd.Timestamp("2026-02-02"),
    ]
    assert frames["close"].columns.tolist() == ["600000.SH", "000001.SZ"]
    assert pd.isna(frames["pre_close"].iloc[0, 0])
    assert frames["pre_close"].iloc[1, 0] == frames["close"].iloc[0, 0]
    assert frames["money"].iloc[0, 0] == pytest.approx(1.001)
    assert frames["valid_ohlc_mask"].all().all()
    assert not frames["zero_volume_mask"].any().any()
    assert frames["basic_open_tradable_mask"].all().all()

    identity = cache.active_identity(
        start_date="2026-01-30",
        end_date="2026-02-02",
    )
    assert [item["month"] for item in identity["months"]] == [
        "2026-01",
        "2026-02",
    ]
    assert all(
        item["generation"].startswith("generation-")
        for item in identity["months"]
    )


def test_revision_only_invalidates_affected_month(tmp_path):
    store_root = tmp_path / "store"
    cache_root = tmp_path / "cache"
    store = MarketDailyStore(store_root)
    store.commit_partition(_frame("2026-01-30"), instrument_type="equity", source="csv")
    store.commit_partition(_frame("2026-02-02"), instrument_type="equity", source="csv")
    cache = MonthlyOhlcvCache(store_root=store_root, cache_root=cache_root)
    january = cache.ensure_month("2026-01")
    february = cache.ensure_month("2026-02")
    february_current = (cache_root / "year=2026" / "month=02" / "CURRENT").read_bytes()

    store.commit_partition(
        _frame("2026-01-30", 10.6),
        instrument_type="equity",
        source="csv",
        allow_revision=True,
    )
    revised = cache.ensure_month("2026-01")

    assert revised["status"] == "rebuilt"
    assert revised["source_month_index_sha256"] != january["source_month_index_sha256"]
    assert cache.ensure_month("2026-02")["status"] == "already_current"
    assert (cache_root / "year=2026" / "month=02" / "CURRENT").read_bytes() == february_current
    assert february["source_month_index_sha256"] != revised["source_month_index_sha256"]


def test_current_metadata_hash_is_verified(tmp_path):
    store = MarketDailyStore(tmp_path / "store")
    store.commit_partition(_frame("2026-01-30"), instrument_type="equity", source="csv")
    cache = MonthlyOhlcvCache(store_root=tmp_path / "store", cache_root=tmp_path / "cache")
    cache.ensure_month("2026-01")
    month_root = tmp_path / "cache" / "year=2026" / "month=01"
    pointer = json.loads((month_root / "CURRENT").read_text())
    (month_root / pointer["generation"] / "meta.json").write_text("{}")

    with pytest.raises(ValueError, match="metadata hash mismatch"):
        cache.ensure_month("2026-01")


def test_month_audit_rejects_corrupt_field(tmp_path):
    store = MarketDailyStore(tmp_path / "store")
    store.commit_partition(_frame("2026-01-30"), instrument_type="equity", source="csv")
    cache = MonthlyOhlcvCache(store_root=tmp_path / "store", cache_root=tmp_path / "cache")
    cache.ensure_month("2026-01")
    month_root = tmp_path / "cache" / "year=2026" / "month=01"
    pointer = json.loads((month_root / "CURRENT").read_text())
    generation = month_root / pointer["generation"]
    meta = json.loads((generation / "meta.json").read_text())
    field = generation / meta["files"]["close"]["path"]
    payload = bytearray(field.read_bytes())
    payload[0] ^= 1
    field.write_bytes(payload)

    with pytest.raises(ValueError, match="field hash mismatch"):
        cache.audit_month("2026-01")


def test_month_cache_lock_rejects_second_writer(tmp_path):
    lock = tmp_path / "writer.lock"
    with _cache_lock(lock):
        with pytest.raises(RuntimeError, match="already active"):
            with _cache_lock(lock):
                raise AssertionError("second writer unexpectedly acquired cache lock")


def test_open_ledger_uses_monthly_frames_for_realistic_masks(tmp_path):
    store_root = tmp_path / "store"
    cache_root = tmp_path / "cache"
    data_dir = tmp_path / "legacy_metadata"
    data_dir.mkdir()
    (data_dir / "stable_stocks.csv").write_text(
        "ts_code,name,list_date\n000001.SZ,Demo,20100101\n"
        "600000.SH,Demo2,20100101\n",
        encoding="utf-8",
    )
    store = MarketDailyStore(store_root)
    first = _frame("2026-01-29")
    second = _frame("2026-01-30", 10.6)
    store.commit_partition(
        first,
        instrument_type="equity",
        source="csv",
    )
    store.commit_partition(
        second,
        instrument_type="equity",
        source="csv",
    )
    for code, group in pd.concat([first, second]).groupby("code"):
        group.drop(columns="code").to_csv(data_dir / f"{code}.csv", index=False)
    args = SimpleNamespace(
        ohlc_backend="monthly",
        market_daily_store_root=str(store_root),
        ohlc_monthly_cache_dir=str(cache_root),
        data_dir=str(data_dir),
        money_scale=1000.0,
        progress_every=0,
        ohlc_cache_dir=str(tmp_path / "window"),
        no_ohlc_cache=True,
        ohlc_matrix_cache_dir=str(tmp_path / "global"),
        no_ohlc_matrix_cache=True,
        rebuild_ohlc_matrix_cache=False,
        execution_constraint_mode="realistic",
        no_execution_mask_cache=True,
        execution_mask_cache_dir=str(tmp_path / "masks"),
        block_intraday_limit_touch=False,
        limit_price_tolerance=1e-4,
        no_limit_first_trading_days=5,
        min_buy_listing_days=60,
    )

    frames = load_execution_market_frames(
        args,
        ["000001.SZ", "600000.SH"],
        start_date="2026-01-29",
        end_date="2026-01-30",
    )
    csv_args = SimpleNamespace(**vars(args))
    csv_args.ohlc_backend = "csv"
    csv_frames = load_execution_market_frames(
        csv_args,
        ["000001.SZ", "600000.SH"],
        start_date="2026-01-29",
        end_date="2026-01-30",
    )
    masks = prepare_execution_constraint_masks(
        args,
        ["000001.SZ", "600000.SH"],
        frames["open"],
        frames["close"],
        frames["money"],
        load_start="2026-01-29",
        load_end="2026-01-30",
        ohlcv_frames=frames,
    )

    assert set(frames) == {"open", "high", "low", "close", "volume", "money"}
    for field in frames:
        pd.testing.assert_frame_equal(
            csv_frames[field],
            frames[field],
            check_exact=True,
        )
    assert frames["money"].iloc[0, 0] == pytest.approx(1_000_000.0)
    assert set(masks) == {
        "buy_block",
        "sell_block",
        "no_trade",
        "limit_up_open",
        "limit_down_open",
        "limit_up_touch",
        "limit_down_touch",
        "new_stock_buy_block",
    }
