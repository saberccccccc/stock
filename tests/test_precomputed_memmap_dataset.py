import numpy as np

from core.train_utils import PrecomputedMemmapDataset, SCALE, SENTINEL


def test_precomputed_dataset_uses_shared_label_validity_mask():
    n_stocks, n_dates = 4, 2
    x = np.full((n_stocks, n_dates, 2), SENTINEL, dtype=np.int16)
    risk = np.full((n_stocks, n_dates, 1), SENTINEL, dtype=np.int16)
    y = np.full((n_stocks, n_dates), SENTINEL, dtype=np.int16)
    y_seq = np.full((n_stocks, n_dates, 2), SENTINEL, dtype=np.int16)
    industry = np.full((n_stocks, n_dates), -1, dtype=np.int16)

    valid_idx = np.array([0, 2, 3])
    x[valid_idx, 1, :] = np.array([[100, 200], [300, 400], [500, 600]])
    risk[valid_idx, 1, :] = 700
    y[valid_idx, 1] = np.array([800, 900, 1000])
    y_seq[valid_idx, 1, :] = 1100

    ds = PrecomputedMemmapDataset(
        x,
        risk,
        y,
        y_seq,
        industry,
        ["a", "b", "c", "d"],
        [0, 1],
        [0, 1],
        n_industries=1,
        max_horizon=2,
        min_stocks=3,
    )

    assert ds.time_indices == [1]
    sample = ds[0]
    assert sample["X"].shape == (3, 2)
    assert np.isclose(sample["X"][0, 0].item(), 100 / SCALE)
    assert np.isclose(sample["y"][-1].item(), 1000 / SCALE)
