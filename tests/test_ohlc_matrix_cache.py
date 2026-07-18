import pandas as pd

from backtest.ohlc_matrix_cache import (
    build_ohlc_matrix_cache,
    load_ohlcv_fields_from_matrix_cache,
    load_ohlc_money_from_matrix_cache,
    matrix_cache_is_current,
)


def test_ohlc_matrix_cache_loads_requested_window_and_codes(tmp_path):
    data_dir = tmp_path / "raw"
    cache_dir = tmp_path / "matrix"
    data_dir.mkdir()
    (data_dir / "000001.SZ.csv").write_text(
        "trade_date,open,high,low,close,volume,money\n"
        "2025-01-02,10,11.5,9.5,11,1000,100\n"
        "2025-01-03,12,13.5,11.5,13,2000,200\n",
        encoding="utf-8",
    )
    (data_dir / "600000.SH.csv").write_text(
        "trade_date,open,high,low,close,volume,money\n"
        "2025-01-03,20,21.5,19.5,21,3000,300\n"
        "2025-01-06,22,23.5,21.5,23,4000,400\n",
        encoding="utf-8",
    )
    (data_dir / "hs300_index.csv").write_text(
        "trade_date,close\n2025-01-03,100\n",
        encoding="utf-8",
    )

    meta = build_ohlc_matrix_cache(data_dir, cache_dir, progress_every=0)
    open_df, close_df, money_df = load_ohlc_money_from_matrix_cache(
        data_dir,
        cache_dir,
        ["600000.SH", "000001.SZ", "MISSING"],
        money_scale=1000.0,
        start_date="2025-01-03",
        end_date="2025-01-06",
        progress_every=0,
    )

    assert meta["codes"] == ["000001.SZ", "600000.SH"]
    assert open_df.index.tolist() == [
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-06"),
    ]
    assert open_df.columns.tolist() == ["600000.SH", "000001.SZ"]
    assert close_df.loc[pd.Timestamp("2025-01-03"), "000001.SZ"] == 13.0
    assert money_df.loc[pd.Timestamp("2025-01-03"), "600000.SH"] == 300_000.0

    fields = load_ohlcv_fields_from_matrix_cache(
        data_dir,
        cache_dir,
        ["000001.SZ"],
        fields=("high", "low", "volume", "pre_close", "pct_chg"),
        start_date="2025-01-02",
        end_date="2025-01-03",
        progress_every=0,
    )
    assert fields["high"].loc[pd.Timestamp("2025-01-03"), "000001.SZ"] == 13.5
    assert fields["low"].loc[pd.Timestamp("2025-01-02"), "000001.SZ"] == 9.5
    assert fields["volume"].loc[pd.Timestamp("2025-01-03"), "000001.SZ"] == 2000
    assert fields["pre_close"].loc[pd.Timestamp("2025-01-03"), "000001.SZ"] == 11
    assert fields["pct_chg"].loc[pd.Timestamp("2025-01-03"), "000001.SZ"] == 13 / 11 - 1


def test_ohlc_matrix_cache_invalidates_when_source_changes(tmp_path):
    data_dir = tmp_path / "raw"
    cache_dir = tmp_path / "matrix"
    data_dir.mkdir()
    source = data_dir / "000001.SZ.csv"
    source.write_text(
        "trade_date,open,high,low,close,volume,money\n2025-01-02,10,11,9,11,1000,100\n",
        encoding="utf-8",
    )

    build_ohlc_matrix_cache(data_dir, cache_dir, progress_every=0)
    assert matrix_cache_is_current(data_dir, cache_dir)

    source.write_text(
        "trade_date,open,high,low,close,volume,money\n"
        "2025-01-02,10,11,9,11,1000,100\n"
        "2025-01-03,12,13,11,13,2000,200\n",
        encoding="utf-8",
    )

    assert not matrix_cache_is_current(data_dir, cache_dir)
    open_df, _, _ = load_ohlc_money_from_matrix_cache(
        data_dir,
        cache_dir,
        ["000001.SZ"],
        money_scale=1.0,
        progress_every=0,
    )
    assert open_df.index.tolist() == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
    ]
