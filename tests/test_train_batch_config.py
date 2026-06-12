import torch

from run.train import MODEL_CONFIGS, build_config, resolve_batch


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
            "memmap_trim_interval": 32,
        },
    )()

    cfg = build_config(MODEL_CONFIGS["v9"], args=args)

    assert cfg.memmap_trim_interval == 32
