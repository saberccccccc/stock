import numpy as np

from data.rolling_samples import iter_rolling_samples, iter_v14_inference_samples


def write_memmap(path, values, dtype):
    out = np.memmap(path, dtype=dtype, mode="w+", shape=values.shape)
    out[:] = values
    out.flush()
    out._mmap.close()


def test_rolling_samples_use_selected_family_and_lag_shift(tmp_path):
    sentinel = np.int16(-32768)
    x = np.full((2, 3, 1), sentinel, dtype=np.int16)
    x[:, :, 0] = 100
    risk = np.zeros((2, 3, 1), dtype=np.int16)
    oo = np.full((2, 3, 1), sentinel, dtype=np.int16)
    oo[:, 1, 0] = [120, 220]
    x_path, r_path, oo_path = tmp_path / "x.dat", tmp_path / "r.dat", tmp_path / "oo.dat"
    write_memmap(x_path, x, np.int16)
    write_memmap(r_path, risk, np.int16)
    write_memmap(oo_path, oo, np.int16)
    meta = {
        "all_codes": ["A", "B"], "all_dates": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "x_dim": 1, "risk_full_dim": 1, "max_horizon": 1, "min_stocks": 1,
        "x_norm_path": x_path, "risk_full_path": r_path,
        "industry_array": np.zeros((2, 3), dtype=np.int16),
        "label_families": {"oo": {"norm_path": oo_path, "date_shift": 0}, "oo_lag1": {"alias_of": "oo", "date_shift": 1}},
    }

    rows = list(iter_rolling_samples(meta, [0, 1], "oo_lag1", 0))

    assert len(rows) == 1
    assert rows[0]["time_index"] == 0
    assert rows[0]["codes"] == ["A", "B"]
    assert np.allclose(rows[0]["y"], [0.12, 0.22])


def test_rolling_samples_can_select_a_feature_subset(tmp_path):
    sentinel = np.int16(-32768)
    x = np.full((2, 2, 3), sentinel, dtype=np.int16)
    x[:, :, 0] = 100
    x[:, :, 1] = 200
    x[:, :, 2] = 300
    risk = np.zeros((2, 2, 1), dtype=np.int16)
    y = np.full((2, 2, 1), sentinel, dtype=np.int16)
    y[:, 0, 0] = [10, 20]
    x_path, r_path, y_path = tmp_path / "x.dat", tmp_path / "r.dat", tmp_path / "y.dat"
    write_memmap(x_path, x, np.int16); write_memmap(r_path, risk, np.int16); write_memmap(y_path, y, np.int16)
    meta = {"all_codes": ["A", "B"], "all_dates": ["2024-01-02", "2024-01-03"], "x_dim": 3, "risk_full_dim": 1, "max_horizon": 1, "min_stocks": 1, "x_norm_path": x_path, "risk_full_path": r_path, "industry_array": np.zeros((2, 2), dtype=np.int16), "label_families": {"oo": {"norm_path": y_path, "date_shift": 0}}}
    row = next(iter_rolling_samples(meta, [0], "oo", 0, feature_indices=[2, 0]))
    assert np.allclose(row["X"], [[0.3, 0.1], [0.3, 0.1]])


def test_v14_inference_samples_are_label_free_and_require_x_and_risk(tmp_path):
    sentinel = np.int16(-32768)
    x = np.array(
        [
            [[100, 200]],
            [[300, 400]],
            [[sentinel, 500]],
        ],
        dtype=np.int16,
    )
    risk = np.array([[[10, 20]], [[sentinel, 30]], [[40, 50]]], dtype=np.int16)
    x_path, risk_path = tmp_path / "x.dat", tmp_path / "risk.dat"
    write_memmap(x_path, x, np.int16)
    write_memmap(risk_path, risk, np.int16)
    meta = {
        "all_codes": ["A", "B", "C"],
        "all_dates": ["2024-01-02"],
        "x_dim": 2,
        "risk_full_dim": 2,
        "min_stocks": 1,
        "x_norm_path": x_path,
        "risk_full_path": risk_path,
        "industry_array": np.array([[1], [2], [3]], dtype=np.int16),
    }

    row = next(iter_v14_inference_samples(meta, [0]))

    assert row["codes"] == ["A"]
    assert "y" not in row and "y_seq" not in row
    assert np.allclose(row["X"], [[0.1, 0.2]])
    assert np.allclose(row["risk"], [[0.01, 0.02]])
    assert row["industry_ids"].tolist() == [1]
