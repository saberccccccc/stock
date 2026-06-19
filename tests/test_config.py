# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

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
    assert cfg.research_end_date == "2026-05-18"
    print("  DataConfig defaults: OK")


def test_research_protocol():
    from core.research_protocol import (
        FORWARD_START_DATE,
        RESEARCH_END_DATE,
        SMALL_ACCOUNT_VALUES,
        assert_alpha_dates_within_forward,
        assert_alpha_dates_within_research,
        assert_research_end_date,
    )
    assert str(RESEARCH_END_DATE.date()) == "2026-05-18"
    assert str(FORWARD_START_DATE.date()) == "2026-05-19"
    assert SMALL_ACCOUNT_VALUES == (500_000.0, 1_000_000.0)
    assert assert_research_end_date(None) == RESEARCH_END_DATE
    assert_alpha_dates_within_research(["2026-05-18"])
    assert_alpha_dates_within_forward(["2026-05-19"])
    try:
        assert_alpha_dates_within_research(["2026-05-19"])
    except ValueError:
        pass
    else:
        raise AssertionError("research Alpha boundary must reject forward dates")
    try:
        assert_alpha_dates_within_forward(["2026-05-18"])
    except ValueError:
        pass
    else:
        raise AssertionError("forward Alpha boundary must reject research dates")
    try:
        assert_research_end_date("2026-05-19")
    except ValueError:
        pass
    else:
        raise AssertionError("research boundary must reject dates after 2026-05-18")
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

    assert resolve_update_end_date("data/raw") == "20260518"
    assert resolve_update_end_date("data/forward_raw", "2026-06-19") == "20260619"
    try:
        resolve_update_end_date("data/raw", "2026-05-19")
    except ValueError:
        pass
    else:
        raise AssertionError("data/raw update must reject dates after research cutoff")


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
