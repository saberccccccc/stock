from pathlib import Path

from experiments.rolling import RollingWindow
from experiments.strong_rolling import (
    build_raw_inference_command,
    build_reconstructed_training_stages,
    build_smoke_contract,
    build_smoke_train_command,
    load_json,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "reconstructed_multi_downside_e19_profile_v1.json"
PROFILE_V2 = ROOT / "configs" / "reconstructed_multi_downside_e19_profile_v2.json"
SCHEDULE = ROOT / "configs" / "monthly_rolling_compact_4y6m1m_2024_2025.json"


def _window():
    return RollingWindow.from_mapping(load_json(SCHEDULE)["windows"][0])


def test_smoke_command_has_fixed_window_purge_and_reconstructed_losses(tmp_path):
    command = build_smoke_train_command(
        python="python",
        profile=load_json(PROFILE),
        window=_window(),
        output_dir=tmp_path / "training",
        cache_meta=ROOT / "cache" / "dummy.pkl",
        epochs=1,
    )
    text = " ".join(map(str, command))

    assert "--train-start 2019-07-01" in text
    assert "--train-label-end 2023-06-30" in text
    assert "--val-start 2023-07-03" in text
    assert "--val-label-end 2023-12-29" in text
    assert "--downside-loss-weight 0.2" in text
    assert "--multi-loss-weight 0.1" in text
    assert "--lag1-label-family oo_lag1" in text
    assert "--epochs 1" in text
    assert "--resume-from" not in text
    assert "--cache-meta" in text
    assert "--expected-input-dim 250" in text


def test_smoke_contract_is_raw_nonselecting_and_hashed(tmp_path):
    window = _window()
    cache = tmp_path / "cache.pkl"
    cache.write_bytes(b"test-cache-contract")
    train = build_smoke_train_command(
        python="python", profile=load_json(PROFILE), window=window, output_dir=tmp_path / "train",
        cache_meta=cache,
    )
    infer = build_raw_inference_command(
        python="python", window=window, checkpoint=tmp_path / "model.pt", output=tmp_path / "alpha.jsonl",
        cache_meta=cache,
    )
    contract = build_smoke_contract(
        profile_path=PROFILE,
        schedule_path=SCHEDULE,
        window=window,
        train_command=train,
        inference_command=infer,
        cache_meta=cache,
    )

    assert contract["selection_allowed"] is False
    assert contract["promotion_allowed"] is False
    assert "--predictor-mode" in infer and infer[infer.index("--predictor-mode") + 1] == "none"
    assert len(contract["contract_sha256"]) == 64


def test_reconstructed_training_stages_match_historical_lineage(tmp_path):
    stages = build_reconstructed_training_stages(
        python="python",
        profile=load_json(PROFILE),
        window=_window(),
        output_dir=tmp_path / "window",
        cache_meta=tmp_path / "cache.pkl",
    )
    assert [item["target_epoch"] for item in stages] == [6, 15, 19]
    first, second, third = (" ".join(item["command"]) for item in stages)
    assert "--lr 0.0001" in first and "--multi-loss-weight 0" in first
    assert "--lr 1e-05" in second and "epoch_006.pt --reset-optimizer" in second
    assert "--lr 5e-06" in third and "epoch_015.pt --reset-optimizer" in third
    assert "--multi-loss-weight 0.1" in third
    assert "--downside-loss-weight 0.2" in third
    assert all("canonical_stage_id" not in item for item in stages)


def test_selected_transition_uses_best_checkpoint_with_logical_stage_boundary(tmp_path):
    stages = build_reconstructed_training_stages(
        python="python",
        profile=load_json(PROFILE),
        window=_window(),
        output_dir=tmp_path / "window",
        cache_meta=tmp_path / "cache.pkl",
        transition_checkpoint="selected",
    )
    second, third = stages[1]["command"], stages[2]["command"]
    assert Path(second[second.index("--resume-from") + 1]).name == "ultimate_v7_best.pt"
    assert second[second.index("--resume-start-epoch") + 1] == "6"
    assert Path(third[third.index("--resume-from") + 1]).name == "ultimate_v7_best.pt"
    assert third[third.index("--resume-start-epoch") + 1] == "15"


def test_hardened_profile_adds_canonical_stages_without_changing_legacy_paths(tmp_path):
    stages = build_reconstructed_training_stages(
        python="python",
        profile=load_json(PROFILE_V2),
        window=_window(),
        output_dir=tmp_path / "window",
        cache_meta=tmp_path / "cache.pkl",
    )

    assert [stage["name"] for stage in stages] == [
        "base_e6",
        "lag1_low_lr_e15",
        "multi_downside_e19",
    ]
    assert [stage["canonical_stage_id"] for stage in stages] == [
        "oo_lag1_e1_e6",
        "oo_lag1_low_lr_e7_e15",
        "multi_downside_e16_e19",
    ]
    assert Path(stages[0]["exact_checkpoint"]).parent.parent.name == "base_e6"
