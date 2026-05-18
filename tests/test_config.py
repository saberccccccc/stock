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
    print("  DataConfig defaults: OK")


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


def test_global_constants():
    from core.config import TRADING_DAYS, ADV_LIMIT_RATIO, TARGET_VOL, EPS
    assert TRADING_DAYS == 252
    assert ADV_LIMIT_RATIO == 0.02
    assert TARGET_VOL == 0.15
    assert EPS == 1e-8
    print("  global constants: OK")


if __name__ == "__main__":
    test_data_config_defaults()
    test_build_v9_config()
    test_global_constants()
    print("test_config.py: ALL PASSED")
