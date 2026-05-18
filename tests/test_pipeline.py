# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def test_normalize_ts_code():
    from data.pipeline import _normalize_ts_code
    assert _normalize_ts_code("000001.SZ") == "000001.SZ"
    assert _normalize_ts_code("000001") == "000001.SZ"
    assert _normalize_ts_code("600000.sh") == "600000.SH"
    assert _normalize_ts_code("600000") == "600000.SH"
    print("  _normalize_ts_code: OK")


def test_load_industry_map():
    from data.pipeline import _load_industry_map
    ind_dict, all_inds, ind_to_idx, n_inds = _load_industry_map("data/raw")
    assert isinstance(ind_dict, dict)
    assert isinstance(ind_to_idx, dict)
    assert n_inds == len(all_inds)
    assert n_inds > 0, "should have industries"
    print(f"  _load_industry_map: {n_inds} industries")


def test_compute_base_features():
    import numpy as np
    import pandas as pd
    from data.pipeline import _compute_base_features

    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    df = pd.DataFrame({
        "open": np.random.randn(100).cumsum() + 100,
        "high": np.random.randn(100).cumsum() + 102,
        "low": np.random.randn(100).cumsum() + 98,
        "close": np.random.randn(100).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 100),
    }, index=dates)
    df["close"] = df["close"].clip(lower=1)
    df_dict = {"TEST.SZ": df}
    base_features = _compute_base_features(df_dict)
    assert len(base_features) == 12
    for col in base_features:
        assert col in df.columns, f"missing column: {col}"
    print(f"  _compute_base_features: {len(base_features)} features computed")


if __name__ == "__main__":
    test_normalize_ts_code()
    test_load_industry_map()
    test_compute_base_features()
    print("test_pipeline.py: ALL PASSED")
