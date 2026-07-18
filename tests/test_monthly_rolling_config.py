import pandas as pd

from run.build_monthly_rolling_config import build_config


def test_monthly_config_freezes_24_unique_oos_windows():
    dates = pd.date_range("2018-01-01", "2025-12-31", freq="B")
    base = {
        "name": "compact",
        "data": {"label_family": "oo_lag1", "horizon_index": 4},
        "model": {"seed": 7},
        "windows": [],
    }

    config = build_config(
        base,
        {"all_dates": dates},
        train_years=4,
        valid_months=6,
        oos_start="2024-01-01",
        oos_end="2025-12-31",
    )

    schedule = config["monthly_schedule"]
    assert schedule["window_count"] == 24
    assert schedule["unique_oos_dates"] == len(pd.date_range("2024-01-01", "2025-12-31", freq="B"))
    assert config["windows"][0]["predict_start"] == "2024-01-01"
    assert config["windows"][-1]["predict_end"] == "2025-12-31"
    assert all(item["train"] > 0 and item["valid"] > 0 for item in schedule["window_counts"])


def test_quarterly_and_halfyear_schedules_keep_the_same_unique_oos_dates():
    dates = pd.date_range("2018-01-01", "2025-12-31", freq="B")
    base = {
        "name": "compact",
        "data": {"label_family": "oo_lag1", "horizon_index": 4},
        "model": {"seed": 7},
        "windows": [],
    }

    quarterly = build_config(
        base,
        {"all_dates": dates},
        train_years=4,
        valid_months=6,
        oos_months=3,
        oos_start="2024-01-01",
        oos_end="2025-12-31",
    )
    halfyear = build_config(
        base,
        {"all_dates": dates},
        train_years=4,
        valid_months=6,
        oos_months=6,
        oos_start="2024-01-01",
        oos_end="2025-12-31",
    )

    expected = len(pd.date_range("2024-01-01", "2025-12-31", freq="B"))
    assert quarterly["monthly_schedule"]["window_count"] == 8
    assert halfyear["monthly_schedule"]["window_count"] == 4
    assert quarterly["monthly_schedule"]["unique_oos_dates"] == expected
    assert halfyear["monthly_schedule"]["unique_oos_dates"] == expected
    assert quarterly["windows"][0]["name"] == "oos_2024_01_2024_03"
    assert halfyear["windows"][0]["name"] == "oos_2024_01_2024_06"
