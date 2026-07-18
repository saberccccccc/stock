import pickle
from pathlib import Path

import numpy as np
import pytest

from data.cache_metadata import load_explicit_cross_section_meta
from data.pipeline import samples_from_precomputed_metadata


def _bundle(tmp_path: Path):
    codes = ["000001", "000002"]
    dates = ["2024-01-02", "2024-01-03"]
    shape = (2, 2)
    x = np.memmap(tmp_path / "x.dat", dtype=np.int16, mode="w+", shape=shape + (2,))
    risk = np.memmap(tmp_path / "risk.dat", dtype=np.int16, mode="w+", shape=shape + (1,))
    y = np.memmap(tmp_path / "y.dat", dtype=np.int16, mode="w+", shape=shape)
    ys = np.memmap(tmp_path / "ys.dat", dtype=np.int16, mode="w+", shape=shape + (2,))
    x[:] = 100
    risk[:] = 100
    y[:] = 100
    ys[:] = 100
    y[1, 1] = -32768
    ys[1, 1, :] = -32768
    for array in (x, risk, y, ys):
        array.flush()
    meta = {
        "all_codes": codes,
        "all_dates": dates,
        "industry_array": np.zeros(shape, dtype=np.int16),
        "train_indices": [0],
        "val_indices": [1],
        "x_dim": 2,
        "risk_full_dim": 1,
        "max_horizon": 2,
        "min_stocks": 1,
        "x_norm_path": str(tmp_path / "x.dat"),
        "risk_full_path": str(tmp_path / "risk.dat"),
        "y_norm_path": str(tmp_path / "y.dat"),
        "y_seq_norm_path": str(tmp_path / "ys.dat"),
        "label_families": {
            "oo": {"norm_path": str(tmp_path / "ys.dat"), "date_shift": 0},
            "oo_lag1": {"alias_of": "oo", "date_shift": 1},
        },
    }
    path = tmp_path / "meta.pkl"
    with path.open("wb") as handle:
        pickle.dump(meta, handle)
    return path


def test_explicit_meta_validates_dimension_labels_and_logical_view(tmp_path):
    path = _bundle(tmp_path)
    meta = load_explicit_cross_section_meta(
        path,
        project_root=tmp_path,
        expected_input_dim=2,
        required_label_families=("oo", "oo_lag1"),
        logical_end="2024-01-02",
    )
    assert meta["physical_data_end"] == "2024-01-03"
    assert meta["effective_data_end"] == "2024-01-02"
    assert meta["cache_view_kind"] == "physical_superset_logical_cutoff"


def test_explicit_meta_rejects_wrong_architecture(tmp_path):
    with pytest.raises(ValueError, match="input dimension mismatch"):
        load_explicit_cross_section_meta(
            _bundle(tmp_path), project_root=tmp_path, expected_input_dim=250
        )


def test_label_free_samples_do_not_filter_on_future_label_availability(tmp_path):
    meta = load_explicit_cross_section_meta(_bundle(tmp_path), project_root=tmp_path)
    labeled = samples_from_precomputed_metadata(meta, time_indices=[1], require_labels=True)
    inference = samples_from_precomputed_metadata(meta, time_indices=[1], require_labels=False)
    assert labeled[0]["codes"] == ["000001"]
    assert inference[0]["codes"] == ["000001", "000002"]
    assert np.all(inference[0]["y"] == 0)
