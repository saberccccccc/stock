import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.cache_upgrade import SENTINEL, derive_v14_meta_path, upgrade_v13_cache


def _write_memmap(path, dtype, shape, values=0):
    mm = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    mm[:] = values
    mm.flush()
    del mm


def test_derive_v14_meta_path_adds_research_end():
    source = Path("cache/cross_section_v13_config_key_tech_all_meta.pkl")
    result = derive_v14_meta_path(source, "2026-05-18")
    assert result.name == (
        "cross_section_v14_multilabel_open_tech_all_end20260518_meta.pkl"
    )


def test_upgrade_reuses_features_and_builds_all_label_families(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    raw_dir = tmp_path / "data" / "raw"
    cache.mkdir()
    raw_dir.mkdir(parents=True)

    codes = ["000001.SZ", "000002.SZ"]
    dates = pd.bdate_range("2024-01-02", periods=6)
    n_stocks, n_dates, horizon = 2, len(dates), 2
    x_dim, risk_dim = 2, 1

    feat = cache / "source_feat.dat"
    risk = cache / "source_risk.dat"
    x_norm = cache / "source_X_norm.dat"
    risk_full = cache / "source_risk_full.dat"
    _write_memmap(feat, np.float32, (n_stocks, n_dates, 1))
    _write_memmap(risk, np.float32, (n_stocks, n_dates, 1))
    _write_memmap(x_norm, np.int16, (n_stocks, n_dates, x_dim))
    _write_memmap(risk_full, np.int16, (n_stocks, n_dates, risk_dim))
    x_mm = np.memmap(x_norm, dtype=np.int16, mode="r+", shape=(n_stocks, n_dates, x_dim))
    x_mm[1, 1, :] = SENTINEL
    x_mm.flush()
    del x_mm

    for stock_idx, code in enumerate(codes):
        base = 10.0 + stock_idx
        pd.DataFrame({
            "trade_date": dates,
            "open": base + np.arange(n_dates),
            "close": base + 0.5 + np.arange(n_dates),
        }).to_csv(raw_dir / f"{code}.csv", index=False)

    source_meta = cache / "cross_section_v13_config_key_tech_test_meta.pkl"
    source = {
        "feat_path": str(feat),
        "risk_path": str(risk),
        "x_norm_path": str(x_norm),
        "risk_full_path": str(risk_full),
        "all_codes": codes,
        "all_dates": list(dates),
        "industry_array": np.zeros((n_stocks, n_dates), dtype=np.int16),
        "train_indices": [0, 1, 2],
        "val_indices": [3],
        "x_dim": x_dim,
        "risk_full_dim": risk_dim,
        "max_horizon": horizon,
        "target_horizon": 1,
        "min_stocks": 1,
        "residualize": False,
    }
    with source_meta.open("wb") as handle:
        pickle.dump(source, handle)

    upgraded, output_meta = upgrade_v13_cache(
        source_meta, data_dir=raw_dir, stock_chunk=1
    )

    assert output_meta.exists()
    assert "end20240109" in output_meta.name
    assert sorted(upgraded["label_families"]) == ["cc", "oc", "oo", "oo_lag1"]
    assert upgraded["label_families"]["oo_lag1"] == {
        "alias_of": "oo", "date_shift": 1
    }
    assert Path(upgraded["x_norm_path"]).resolve() == x_norm.resolve()

    oo_meta = upgraded["label_families"]["oo"]
    oo_raw = np.memmap(
        oo_meta["raw_path"], dtype=np.float32, mode="r",
        shape=(n_stocks, n_dates, horizon),
    )
    assert oo_raw[0, 0, 0] == pytest.approx(12.0 / 11.0 - 1.0)

    oo_norm = np.memmap(
        oo_meta["norm_path"], dtype=np.int16, mode="r",
        shape=(n_stocks, n_dates, horizon),
    )
    assert oo_norm[1, 1, 0] == SENTINEL
    assert not list(cache.glob("*.building"))
    assert not list(cache.glob("*.tmp"))
