import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scripts.download_fundamentals_akshare import _expected_latest_report_period


def main():
    assert _expected_latest_report_period("2026-04-29") == pd.Timestamp("2025-09-30")
    assert _expected_latest_report_period("2026-04-30") == pd.Timestamp("2025-12-31")
    assert _expected_latest_report_period("2026-05-01") == pd.Timestamp("2026-03-31")
    assert _expected_latest_report_period("2026-09-01") == pd.Timestamp("2026-06-30")
    assert _expected_latest_report_period("2026-11-01") == pd.Timestamp("2026-09-30")
    print("test_fundamental_update.py: ALL PASSED")


if __name__ == "__main__":
    main()
