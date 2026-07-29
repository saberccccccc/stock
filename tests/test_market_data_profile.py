import pandas as pd

from backtest.ohlc_matrix_cache import RAW_FIELDS
from data.market_data_profile import inventory_csv_root, profile_csv_range


def _write_stock(path, dates):
    rows = []
    for date in dates:
        rows.append(
            {
                "trade_date": date,
                "code": path.stem,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100.0,
                "money": 1000.0,
                "factor": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_inventory_uses_fast_tail_dates(tmp_path):
    _write_stock(tmp_path / "000001.SZ.csv", ["2026-07-28", "2026-07-29"])
    _write_stock(tmp_path / "600000.SH.csv", ["2026-07-28"])

    result = inventory_csv_root(tmp_path)

    assert result["stock_files"] == 2
    assert result["signature"]["source_count"] == 2
    assert dict(result["latest_dates_top20"]) == {
        "2026-07-28": 1,
        "2026-07-29": 1,
    }


def test_profile_csv_range_streams_requested_interval(tmp_path):
    _write_stock(
        tmp_path / "000001.SZ.csv",
        ["2025-12-31", "2026-01-02", "2026-01-05"],
    )

    result = profile_csv_range(
        tmp_path,
        start_date="2026-01-01",
        end_date="2026-01-03",
    )

    assert result["files_scanned"] == 1
    assert result["rows_in_range"] == 1
    assert not result["failures"]
    assert set(RAW_FIELDS) == {"open", "high", "low", "close", "volume", "money"}
