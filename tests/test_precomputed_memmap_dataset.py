import numpy as np
import torch

from core.train_utils import PrecomputedMemmapDataset, SCALE, SENTINEL, collate_fn


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
    assert "lag1_y_seq" not in sample
    assert "lag1_mask" not in sample


def test_precomputed_dataset_exposes_raw_returns():
    x = np.zeros((2, 1, 1), dtype=np.int16)
    risk = np.zeros((2, 1, 1), dtype=np.int16)
    y = np.zeros((2, 1), dtype=np.int16)
    y_seq = np.zeros((2, 1, 2), dtype=np.int16)
    raw = np.asarray([[[0.01, 0.02]], [[0.03, 0.04]]], dtype=np.float32)
    industry = np.zeros((2, 1), dtype=np.int16)

    ds = PrecomputedMemmapDataset(
        x,
        risk,
        y,
        y_seq,
        industry,
        ["a", "b"],
        [0],
        [0],
        n_industries=1,
        max_horizon=2,
        min_stocks=1,
        raw_ret_mm=raw,
    )

    sample = ds[0]
    assert np.allclose(sample["raw_y_seq"].numpy(), raw[:, 0, :])


def test_precomputed_dataset_filters_only_selected_horizons():
    n_stocks, n_dates, horizon = 3, 2, 3
    x = np.zeros((n_stocks, n_dates, 1), dtype=np.int16)
    risk = np.zeros((n_stocks, n_dates, 1), dtype=np.int16)
    y = np.zeros((n_stocks, n_dates), dtype=np.int16)
    y_seq = np.zeros((n_stocks, n_dates, horizon), dtype=np.int16)
    industry = np.zeros((n_stocks, n_dates), dtype=np.int16)
    y_seq[0, 0, 1] = SENTINEL  # Unused h2 must not remove this stock.
    y_seq[1, 0, 2] = SENTINEL  # Selected h3 must remove this stock.

    ds = PrecomputedMemmapDataset(
        x, risk, y, y_seq, industry, ["a", "b", "c"], [0, 1], [0],
        n_industries=1, max_horizon=horizon, min_stocks=2,
        horizon_indices=(0, 2), target_horizon_index=0,
    )

    sample = ds[0]
    assert sample["X"].shape[0] == 2
    assert sample["y_seq"].shape == (2, horizon)


def test_precomputed_dataset_supports_shifted_primary_family_view():
    x = np.zeros((2, 3, 1), dtype=np.int16)
    risk = np.zeros((2, 3, 1), dtype=np.int16)
    y = np.zeros((2, 3), dtype=np.int16)
    y_seq = np.zeros((2, 3, 1), dtype=np.int16)
    y_seq[:, 1, 0] = np.asarray([100, 200], dtype=np.int16)
    raw = np.zeros((2, 3, 1), dtype=np.float32)
    raw[:, 1, 0] = np.asarray([0.01, 0.02], dtype=np.float32)
    industry = np.zeros((2, 3), dtype=np.int16)

    ds = PrecomputedMemmapDataset(
        x, risk, y, y_seq, industry, ["a", "b"], [0, 1, 2], [0],
        n_industries=1, max_horizon=1, min_stocks=1, raw_ret_mm=raw,
        horizon_indices=(0,), target_horizon_index=0, label_date_shift=1,
    )

    sample = ds[0]
    assert np.allclose(sample["y"].numpy(), [0.1, 0.2])
    assert np.allclose(sample["raw_y_seq"].numpy()[:, 0], [0.01, 0.02])


def test_precomputed_dataset_exposes_lag1_labels_for_same_stock_set():
    n_stocks, n_dates, horizon = 3, 3, 2
    x = np.zeros((n_stocks, n_dates, 1), dtype=np.int16)
    risk = np.zeros((n_stocks, n_dates, 1), dtype=np.int16)
    y = np.zeros((n_stocks, n_dates), dtype=np.int16)
    y_seq = np.zeros((n_stocks, n_dates, horizon), dtype=np.int16)
    industry = np.zeros((n_stocks, n_dates), dtype=np.int16)
    y[1, 2] = SENTINEL
    y_seq[:, 2, :] = np.asarray([[100, 200], [300, 400], [500, 600]], dtype=np.int16)

    ds = PrecomputedMemmapDataset(
        x,
        risk,
        y,
        y_seq,
        industry,
        ["a", "b", "c"],
        [0, 1, 2],
        [1],
        n_industries=1,
        max_horizon=horizon,
        min_stocks=1,
        include_lag1_labels=True,
    )

    sample = ds[0]
    assert np.allclose(
        sample["lag1_y_seq"].numpy(),
        np.asarray([[0.1, 0.2], [0.0, 0.0], [0.5, 0.6]], dtype=np.float32),
    )
    assert sample["lag1_mask"].numpy().tolist() == [True, False, True]


def test_precomputed_dataset_can_read_lag1_from_separate_label_family():
    n_stocks, n_dates, horizon = 2, 3, 1
    x = np.zeros((n_stocks, n_dates, 1), dtype=np.int16)
    risk = np.zeros((n_stocks, n_dates, 1), dtype=np.int16)
    y = np.zeros((n_stocks, n_dates), dtype=np.int16)
    cc = np.zeros((n_stocks, n_dates, horizon), dtype=np.int16)
    oo = np.zeros((n_stocks, n_dates, horizon), dtype=np.int16)
    industry = np.zeros((n_stocks, n_dates), dtype=np.int16)
    cc[:, 0, 0] = [10, 20]
    oo[:, 1, 0] = [300, 500]

    ds = PrecomputedMemmapDataset(
        x,
        risk,
        y,
        cc,
        industry,
        ["a", "b"],
        [0, 1, 2],
        [0],
        n_industries=1,
        max_horizon=horizon,
        min_stocks=1,
        include_lag1_labels=True,
        horizon_indices=(0,),
        lag1_y_seq_norm_mm=oo,
        lag1_label_date_shift=1,
    )

    sample = ds[0]
    assert np.allclose(sample["y_seq"].numpy()[:, 0], [0.01, 0.02])
    assert np.allclose(sample["lag1_y_seq"].numpy()[:, 0], [0.3, 0.5])


def test_subsample_collate_preserves_optional_training_labels():
    item = {
        "X": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "y": torch.arange(4, dtype=torch.float32),
        "y_seq": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "risk": torch.zeros(4, 1),
        "industry_ids": torch.arange(4),
        "raw_y_seq": torch.arange(8, dtype=torch.float32).reshape(4, 2) + 10,
        "lag1_y_seq": torch.arange(8, dtype=torch.float32).reshape(4, 2) + 20,
        "lag1_mask": torch.tensor([True, False, True, False]),
    }

    output = collate_fn([item], keep_ratio=1.0, min_keep=1)
    valid = output["mask"][0]

    assert torch.allclose(
        output["raw_y_seq"][0, valid] - output["y_seq"][0, valid],
        torch.full((4, 2), 10.0),
    )
    assert torch.allclose(
        output["lag1_y_seq"][0, valid] - output["y_seq"][0, valid],
        torch.full((4, 2), 20.0),
    )
    assert output["lag1_mask"][0, valid].sum().item() == 2
