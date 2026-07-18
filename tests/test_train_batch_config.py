import pandas as pd
import pytest
import torch

from run.train import MODEL_CONFIGS, build_config, resolve_batch, resolve_time_split


def test_resolve_batch_accepts_memory_safe_overrides():
    batch_size, accum_steps, val_batch_size, _ = resolve_batch(
        "v9",
        torch.device("cpu"),
        batch_size=1,
        val_batch_size=1,
        accum_steps=16,
    )

    assert batch_size == 1
    assert val_batch_size == 1
    assert accum_steps == 16


def test_resolve_batch_uses_training_override_for_validation_by_default():
    batch_size, accum_steps, val_batch_size, _ = resolve_batch(
        "v9",
        torch.device("cpu"),
        batch_size=2,
    )

    assert batch_size == 2
    assert val_batch_size == 2
    assert accum_steps == 4


def test_build_config_accepts_memmap_trim_interval():
    args = type(
        "Args",
        (),
        {
            "top_focus_loss_weight": None,
            "top_focus_temperature": None,
            "top_focus_delay_epochs": None,
            "pairwise_top_loss_weight": None,
            "pairwise_top_frac": None,
            "pairwise_num_pairs": None,
            "pairwise_model_top_weight": None,
            "pairwise_delay_epochs": None,
            "best_val_metric": None,
            "eval_top_fracs": None,
            "horizon_weights": None,
            "save_every_epoch": False,
            "early_stop_patience": None,
            "memmap_trim_interval": 32,
        },
    )()

    cfg = build_config(MODEL_CONFIGS["v9"], args=args)

    assert cfg.memmap_trim_interval == 32


def test_build_config_accepts_horizon_weights_and_epoch_saves():
    args = type(
        "Args",
        (),
        {
            "top_focus_loss_weight": None,
            "top_focus_temperature": None,
            "top_focus_delay_epochs": None,
            "pairwise_top_loss_weight": None,
            "pairwise_top_frac": None,
            "pairwise_num_pairs": None,
            "pairwise_model_top_weight": None,
            "pairwise_delay_epochs": None,
            "best_val_metric": None,
            "eval_top_fracs": None,
            "horizon_weights": "0.10,0.25,0.40,0.25",
            "save_every_epoch": True,
            "early_stop_patience": 12,
            "memmap_trim_interval": None,
        },
    )()

    cfg = build_config(MODEL_CONFIGS["v9"], args=args)

    assert cfg.horizon_weights == (0.10, 0.25, 0.40, 0.25)
    assert cfg.save_every_epoch is True
    assert cfg.early_stop_patience == 12


def test_build_config_accepts_loss_ablation_weights():
    args = type(
        "Args",
        (),
        {
            "top_focus_loss_weight": 0.005,
            "top_focus_temperature": None,
            "top_focus_delay_epochs": None,
            "downside_loss_weight": 0.003,
            "downside_temperature": 0.6,
            "downside_delay_epochs": 2,
            "pairwise_top_loss_weight": None,
            "pairwise_top_frac": None,
            "pairwise_num_pairs": None,
            "pairwise_model_top_weight": None,
            "pairwise_delay_epochs": None,
            "best_val_metric": None,
            "eval_top_fracs": None,
            "horizon_weights": None,
            "save_every_epoch": False,
            "early_stop_patience": None,
            "memmap_trim_interval": None,
            "industry_loss_weight": 0.0,
            "multi_loss_weight": 0.2,
            "diversity_loss_weight": 0.0,
            "spread_loss_weight": 0.0,
            "spread_delay_epochs": 2,
        },
    )()

    cfg = build_config(MODEL_CONFIGS["v9"], args=args)

    assert cfg.industry_loss_weight == 0.0
    assert cfg.multi_loss_weight == 0.2
    assert cfg.diversity_loss_weight == 0.0
    assert cfg.spread_loss_weight == 0.0
    assert cfg.spread_delay_epochs == 2
    assert cfg.downside_loss_weight == 0.003
    assert cfg.downside_temperature == 0.6
    assert cfg.downside_delay_epochs == 2


def test_resolve_time_split_purges_labels_at_boundaries():
    dates = pd.bdate_range("2023-12-01", "2025-01-31")
    meta = {
        "all_dates": list(dates),
        "train_indices": list(range(20)),
        "val_indices": list(range(20, len(dates) - 10)),
        "max_horizon": 10,
    }

    train, val, heldout = resolve_time_split(
        meta,
        train_label_end="2023-12-31",
        val_label_end="2024-12-31",
    )

    assert dates[train[-1] + 10] <= pd.Timestamp("2023-12-31")
    assert dates[val[0]] > pd.Timestamp("2023-12-31")
    assert dates[val[-1] + 10] <= pd.Timestamp("2024-12-31")
    assert dates[heldout[0]] > pd.Timestamp("2024-12-31")
    assert not set(train) & set(val)
    assert not set(val) & set(heldout)

    shifted_train, shifted_val, _ = resolve_time_split(
        meta,
        train_label_end="2023-12-31",
        val_label_end="2024-12-31",
        label_shift=1,
    )
    assert dates[shifted_train[-1] + 11] <= pd.Timestamp("2023-12-31")
    assert dates[shifted_val[-1] + 11] <= pd.Timestamp("2024-12-31")
    assert len(shifted_train) < len(train)
    assert len(shifted_val) < len(val)

    oo_train, oo_val, _ = resolve_time_split(
        meta,
        train_label_end="2023-12-31",
        val_label_end="2024-12-31",
        label_family="oo",
        horizon_indices=(6,),
    )
    assert dates[oo_train[-1] + 8] <= pd.Timestamp("2023-12-31")
    assert dates[oo_val[-1] + 8] <= pd.Timestamp("2024-12-31")

    cc_oo_lag1_train, cc_oo_lag1_val, _ = resolve_time_split(
        meta,
        train_label_end="2023-12-31",
        val_label_end="2024-12-31",
        label_family="cc",
        auxiliary_label_family="oo_lag1",
        horizon_indices=(6,),
    )
    assert dates[cc_oo_lag1_train[-1] + 9] <= pd.Timestamp("2023-12-31")
    assert dates[cc_oo_lag1_val[-1] + 9] <= pd.Timestamp("2024-12-31")


def test_resolve_time_split_requires_both_boundaries():
    meta = {
        "all_dates": list(pd.bdate_range("2023-01-01", periods=100)),
        "train_indices": list(range(50)),
        "val_indices": list(range(50, 90)),
        "max_horizon": 5,
    }

    try:
        resolve_time_split(meta, train_label_end="2023-03-01")
    except ValueError as exc:
        assert "provided together" in str(exc)
    else:
        raise AssertionError("one-sided time split must be rejected")


def test_resolve_time_split_supports_fixed_length_feature_windows():
    dates = pd.bdate_range("2019-01-01", "2025-03-31")
    meta = {
        "all_dates": list(dates),
        "train_indices": list(range(len(dates) // 2)),
        "val_indices": list(range(len(dates) // 2, len(dates) - 10)),
        "max_horizon": 10,
    }

    train, val, _ = resolve_time_split(
        meta,
        train_start="2020-01-01",
        train_label_end="2023-06-30",
        val_start="2023-07-03",
        val_label_end="2023-12-29",
        label_family="oo_lag1",
        horizon_indices=(0, 2, 4, 6),
    )

    assert dates[train[0]] >= pd.Timestamp("2020-01-01")
    assert dates[train[-1] + 9] <= pd.Timestamp("2023-06-30")
    assert dates[val[0]] >= pd.Timestamp("2023-07-03")
    assert dates[val[-1] + 9] <= pd.Timestamp("2023-12-29")


def test_resolve_time_split_rejects_partial_or_overlapping_starts():
    dates = pd.bdate_range("2022-01-01", "2024-12-31")
    meta = {
        "all_dates": list(dates),
        "train_indices": list(range(len(dates) // 2)),
        "val_indices": list(range(len(dates) // 2, len(dates) - 10)),
        "max_horizon": 5,
    }

    with pytest.raises(ValueError, match="provided together"):
        resolve_time_split(
            meta,
            train_start="2022-01-01",
            train_label_end="2023-06-30",
            val_label_end="2023-12-29",
        )
    with pytest.raises(ValueError, match="val_start must be after"):
        resolve_time_split(
            meta,
            train_start="2022-01-01",
            train_label_end="2023-06-30",
            val_start="2023-06-01",
            val_label_end="2023-12-29",
        )


def test_shifted_labels_require_explicit_boundaries():
    meta = {
        "all_dates": list(pd.bdate_range("2023-01-01", periods=100)),
        "train_indices": list(range(50)),
        "val_indices": list(range(50, 90)),
        "max_horizon": 5,
    }

    try:
        resolve_time_split(meta, label_shift=1)
    except ValueError as exc:
        assert "requires explicit" in str(exc)
    else:
        raise AssertionError("shifted labels without explicit boundaries must be rejected")
