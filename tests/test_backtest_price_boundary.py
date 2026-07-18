from types import SimpleNamespace

import pandas as pd


def test_load_price_volume_hides_rows_after_configured_research_end(tmp_path):
    from backtest.engine import load_price_volume

    pd.DataFrame(
        {
            "trade_date": ["2024-12-31", "2025-01-02"],
            "close": [10.0, 99.0],
            "volume": [1000.0, 2000.0],
        }
    ).to_csv(tmp_path / "000001.SZ.csv", index=False)
    config = SimpleNamespace(
        data_dir=str(tmp_path),
        max_stocks=None,
        research_end_date="2024-12-31",
    )

    prices, volumes = load_price_volume(config)

    assert prices["000001.SZ"].index.max() == pd.Timestamp("2024-12-31")
    assert volumes["000001.SZ"].index.max() == pd.Timestamp("2024-12-31")
