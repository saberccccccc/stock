import json

import pandas as pd
import pytest

from data.market_daily_migration import migrate_csv_month
from data.market_daily_store import MarketDailyStore


def _write_stock(path, code, dates):
    pd.DataFrame(
        [
            {
                "trade_date": date,
                "code": code,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100.0,
                "money": 1000.0,
                "factor": 1.0,
            }
            for date in dates
        ]
    ).to_csv(path, index=False)


def test_month_migration_is_exact_and_resumable(tmp_path):
    source = tmp_path / "csv"
    store = tmp_path / "store"
    source.mkdir()
    _write_stock(
        source / "000001.SZ.csv",
        "000001.SZ",
        ["2026-06-30", "2026-07-01", "2026-07-02"],
    )
    _write_stock(
        source / "600000.SH.csv",
        "600000.SH",
        ["2026-07-01", "2026-07-02", "2026-08-03"],
    )

    first = migrate_csv_month(
        source_root=source,
        store_root=store,
        month="2026-07",
        progress_every=0,
    )
    second = migrate_csv_month(
        source_root=source,
        store_root=store,
        month="2026-07",
        progress_every=0,
    )

    assert first["audit"]["status"] == second["audit"]["status"] == "passed"
    assert first["audit"]["rows"] == 4
    assert first["completed_dates"] == ["20260701", "20260702"]
    assert len(list(store.rglob("part-*.parquet"))) == 2
    loaded = MarketDailyStore(store).load(
        instrument_type="equity",
        start_date="2026-07-01",
        end_date="2026-07-31",
        fields=["close"],
    )
    assert len(loaded) == 4


def test_resume_rejects_changed_source_signature(tmp_path):
    source = tmp_path / "csv"
    store = tmp_path / "store"
    source.mkdir()
    path = source / "000001.SZ.csv"
    _write_stock(path, "000001.SZ", ["2026-07-01"])
    migrate_csv_month(
        source_root=source,
        store_root=store,
        month="2026-07",
        progress_every=0,
    )
    frame = pd.read_csv(path)
    frame.loc[0, "close"] = 10.6
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="source_signature"):
        migrate_csv_month(
            source_root=source,
            store_root=store,
            month="2026-07",
            progress_every=0,
        )


def test_progress_claiming_missing_partition_is_rejected(tmp_path):
    source = tmp_path / "csv"
    store = tmp_path / "store"
    source.mkdir()
    _write_stock(source / "000001.SZ.csv", "000001.SZ", ["2026-07-01"])
    payload = migrate_csv_month(
        source_root=source,
        store_root=store,
        month="2026-07",
        progress_every=0,
    )
    current = json.loads((store / "CURRENT").read_text())
    manifest = store / current["manifest"]
    manifest.unlink()

    with pytest.raises(FileNotFoundError):
        migrate_csv_month(
            source_root=source,
            store_root=store,
            month="2026-07",
            progress_every=0,
        )


def test_year_period_migrates_multiple_months_in_one_source_scan(tmp_path):
    source = tmp_path / "csv"
    store = tmp_path / "store"
    source.mkdir()
    _write_stock(
        source / "000001.SZ.csv",
        "000001.SZ",
        ["2025-12-31", "2026-01-05", "2026-07-01"],
    )

    payload = migrate_csv_month(
        source_root=source,
        store_root=store,
        month="2026",
        progress_every=0,
    )

    assert payload["completed_dates"] == ["20260105", "20260701"]
    manifest, _ = MarketDailyStore(store).load_manifest()
    assert sorted(manifest["monthly_indexes"]) == [
        "equity:202601",
        "equity:202607",
    ]
