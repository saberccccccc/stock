# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def test_data_config_defaults():
    from core.config import DataConfig
    cfg = DataConfig()
    assert cfg.data_dir == "data/raw"
    assert cfg.seq_len == 40
    assert cfg.target_horizon == 5
    assert cfg.max_horizon == 10
    assert cfg.min_stocks_per_time == 30
    assert cfg.test_mode is False
    assert cfg.research_end_date == "2025-12-31"
    print("  DataConfig defaults: OK")


def test_research_protocol():
    from core.research_protocol import (
        FORWARD_START_DATE,
        FORWARD_END_DATE,
        RESEARCH_END_DATE,
        SPLIT_SPECS,
        SMALL_ACCOUNT_VALUES,
        assert_forward_parent_frozen,
        assert_alpha_dates_within_forward,
        assert_alpha_dates_within_research,
        assert_research_end_date,
        cached_dates_within_research,
        resolve_market_data_end_date,
    )
    assert str(RESEARCH_END_DATE.date()) == "2025-12-31"
    assert str(FORWARD_START_DATE.date()) == "2026-01-01"
    assert str(FORWARD_END_DATE.date()) == "2026-06-30"
    assert tuple(SPLIT_SPECS) == ("val_2024", "test_2025", "forward_2026")
    assert SPLIT_SPECS["forward_2026"].selection_eligible is False
    assert SMALL_ACCOUNT_VALUES == (500_000.0, 1_000_000.0)
    assert assert_research_end_date(None) == RESEARCH_END_DATE
    assert resolve_market_data_end_date(None) == RESEARCH_END_DATE
    assert resolve_market_data_end_date(None, allow_forward=True) is None
    metadata = {"all_dates": ["2024-01-01", "2025-01-01"]}
    assert cached_dates_within_research(metadata)
    assert not cached_dates_within_research(metadata, "2024-12-31")
    assert_alpha_dates_within_research(["2025-12-31"])
    assert_alpha_dates_within_forward(["2026-01-01"])
    assert_forward_parent_frozen("2025-12-31", "2025-12-31")
    try:
        assert_alpha_dates_within_research(["2026-01-01"])
    except ValueError:
        pass
    else:
        raise AssertionError("research Alpha boundary must reject forward dates")
    try:
        assert_alpha_dates_within_forward(["2025-12-31"])
    except ValueError:
        pass
    else:
        raise AssertionError("forward Alpha boundary must reject research dates")
    try:
        assert_research_end_date("2026-01-01")
    except ValueError:
        pass
    else:
        raise AssertionError("research boundary must reject dates after 2025-12-31")
    try:
        assert_forward_parent_frozen("2026-01-01", "2025-12-31")
    except ValueError:
        pass
    else:
        raise AssertionError("full-year forward must reject a parent fitted in 2026")
    print("  research protocol: OK")


def test_build_v9_config():
    from backtest.runtime import build_v9_backtest_config
    cfg = build_v9_backtest_config()
    assert cfg.use_technical_features is True
    assert cfg.use_market_features is True
    assert cfg.use_macro_features is True
    assert cfg.min_stocks_per_time == 30
    assert cfg.target_horizon == 5
    assert cfg.seq_len == 40
    assert cfg.max_horizon == 10
    print("  build_v9_backtest_config: OK")


def test_data_update_respects_research_boundary():
    from data.update import resolve_update_end_date

    assert resolve_update_end_date("data/raw") == "20251231"
    assert resolve_update_end_date("data/forward_raw", "2026-06-19") == "20260619"
    try:
        resolve_update_end_date("data/raw", "2026-01-01")
    except ValueError:
        pass
    else:
        raise AssertionError("data/raw update must reject dates after research cutoff")


def test_daily_update_uses_same_research_boundary():
    from data.update_daily import resolve_update_end_date

    assert resolve_update_end_date("data/raw") == "20251231"
    assert resolve_update_end_date("data/tracking_raw", "2026-06-19") == "20260619"


def test_daily_update_truncates_rows_at_resolved_end_date(tmp_path):
    import data.update_daily as update_daily

    original_end = update_daily.UPDATE_END_DATE
    try:
        update_daily.UPDATE_END_DATE = pd.Timestamp("2025-12-31")
        frame = pd.DataFrame(
            {"close": [10.0, 11.0]},
            index=pd.to_datetime(["2025-12-31", "2026-01-01"]),
        )
        path = tmp_path / "000001.SZ.csv"

        update_daily.safe_to_csv(frame, path)

        saved = pd.read_csv(path, index_col=0, parse_dates=True)
        assert saved.index.max() == pd.Timestamp("2025-12-31")
    finally:
        update_daily.UPDATE_END_DATE = original_end


def test_daily_update_skips_file_already_at_frozen_end_date(tmp_path):
    import data.update_daily as update_daily

    original_dir = update_daily.DATA_DIR
    original_end = update_daily.UPDATE_END_DATE
    try:
        update_daily.DATA_DIR = str(tmp_path)
        update_daily.UPDATE_END_DATE = pd.Timestamp("2025-12-31")
        pd.DataFrame(
            {"close": [10.0]},
            index=pd.to_datetime(["2025-12-31"]),
        ).to_csv(tmp_path / "000001.SZ.csv")

        needs_update, status = update_daily.needs_update("000001.SZ")

        assert needs_update is False
        assert status == "skip (behind 0d)"
    finally:
        update_daily.DATA_DIR = original_dir
        update_daily.UPDATE_END_DATE = original_end


def test_full_update_removes_existing_rows_after_end_date(tmp_path):
    from data.update import TushareProLite

    path = tmp_path / "000001.SZ.csv"
    pd.DataFrame(
        {"close": [10.0, 11.0]},
        index=pd.to_datetime(["2025-12-31", "2026-01-01"]),
    ).to_csv(path)
    updater = object.__new__(TushareProLite)

    result = updater.update_single_stock(
        "000001.SZ",
        str(tmp_path),
        start_date="20251231",
        end_date="20251231",
    )

    saved = pd.read_csv(path, index_col=0, parse_dates=True)
    assert result.index.max() == pd.Timestamp("2025-12-31")
    assert saved.index.max() == pd.Timestamp("2025-12-31")


def test_global_constants():
    from core.config import TRADING_DAYS, ADV_LIMIT_RATIO, TARGET_VOL, EPS
    assert TRADING_DAYS == 252
    assert ADV_LIMIT_RATIO == 0.02
    assert TARGET_VOL == 0.15
    assert EPS == 1e-8
    print("  global constants: OK")


if __name__ == "__main__":
    test_data_config_defaults()
    test_research_protocol()
    test_build_v9_config()
    test_global_constants()
    print("test_config.py: ALL PASSED")
