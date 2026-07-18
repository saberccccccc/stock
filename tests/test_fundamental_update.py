import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scripts.download_fundamentals_akshare import _expected_latest_report_period
from data.fundamental_factors import merge_to_daily_akshare


def test_expected_latest_report_period_respects_disclosure_calendar():
    assert _expected_latest_report_period("2026-04-29") == pd.Timestamp("2025-09-30")
    assert _expected_latest_report_period("2026-04-30") == pd.Timestamp("2025-12-31")
    assert _expected_latest_report_period("2026-05-01") == pd.Timestamp("2026-03-31")
    assert _expected_latest_report_period("2026-09-01") == pd.Timestamp("2026-06-30")
    assert _expected_latest_report_period("2026-11-01") == pd.Timestamp("2026-09-30")
    print("test_fundamental_update.py: ALL PASSED")


def test_merge_to_daily_akshare_adds_quality_features():
    funda = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "effective_date": pd.Timestamp("2025-04-28"),
                "end_date": pd.Timestamp("2025-03-31"),
                "roe": 0.10,
                "revenue_yoy": 12.0,
                "notice_is_estimated": True,
            }
        ]
    )
    dates = pd.to_datetime(["2025-04-25", "2025-04-28", "2025-05-08", "2025-06-01"])

    daily = merge_to_daily_akshare(
        funda,
        ["000001.SZ", "000002.SZ"],
        dates,
        include_quality=True,
    )

    assert daily.loc[pd.Timestamp("2025-04-25"), "000001.SZ_roe"] == 0.0
    assert daily.loc[pd.Timestamp("2025-04-28"), "000001.SZ_roe"] == 0.10
    assert daily.loc[pd.Timestamp("2025-04-25"), "000001.SZ_has_value"] == 0.0
    assert daily.loc[pd.Timestamp("2025-04-28"), "000001.SZ_has_value"] == 1.0
    assert daily.loc[pd.Timestamp("2025-05-08"), "000001.SZ_days_since_effective"] == 10.0
    assert daily.loc[pd.Timestamp("2025-05-08"), "000001.SZ_is_fresh_quarter"] == 1.0
    assert daily.loc[pd.Timestamp("2025-06-01"), "000001.SZ_is_fresh_quarter"] == 0.0
    assert daily.loc[pd.Timestamp("2025-05-08"), "000001.SZ_notice_is_estimated"] == 1.0
    assert daily["000002.SZ_has_value"].sum() == 0.0


if __name__ == "__main__":
    test_expected_latest_report_period_respects_disclosure_calendar()
    test_merge_to_daily_akshare_adds_quality_features()
