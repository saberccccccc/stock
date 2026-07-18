import json
from pathlib import Path

import pytest

from run.rolling_strong_staged_pilot import (
    PILOT_WINDOWS,
    _load_progress,
    _resume_command,
    _select_windows,
    build_contract,
    parse_args,
)
from experiments.strong_rolling import (
    build_standard_rolling_manifest,
    finalize_full_rolling_contract,
    load_json,
)
from core.train_utils import resolve_resume_start_epoch


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "reconstructed_multi_downside_e19_profile_v1.json"
SCHEDULE = ROOT / "configs" / "monthly_rolling_compact_4y6m1m_2024_2025.json"
CACHE = ROOT / "cache" / "cross_section_v14_multilabel_open_tech_market_funda_shareh_restr_macro_all_s40_t5_h10_min30_mad_rawlab_end20260518_meta.pkl"


def test_staged_contract_freezes_three_windows_and_three_stage_lineage(tmp_path):
    schedule = load_json(SCHEDULE)
    windows = _select_windows(schedule)
    contract = build_contract(PROFILE, SCHEDULE, CACHE, windows, tmp_path, "cuda")
    assert tuple(item["window"]["name"] for item in contract["windows"]) == PILOT_WINDOWS
    assert [item["target_epoch"] for item in contract["windows"][0]["stages"]] == [6, 15, 19]
    assert contract["selection_allowed"] is False
    assert contract["promotion_allowed"] is False
    assert len(contract["contract_sha256"]) == 64


def test_progress_rejects_contract_drift(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"schema": "strong_rolling_staged_progress_v1", "contract_sha256": "old", "windows": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen contract"):
        _load_progress(path, "new")


def test_interrupted_stage_resumes_latest_without_resetting_optimizer(tmp_path):
    epoch_dir = tmp_path / "epochs"
    epoch_dir.mkdir()
    (epoch_dir / "epoch_007.pt").write_bytes(b"seven")
    (epoch_dir / "epoch_008.pt").write_bytes(b"eight")
    stage = {
        "command": [
            "python", "train.py", "--resume-from", "parent.pt", "--reset-optimizer",
            "--resume-start-epoch", "6",
        ],
        "exact_checkpoint": str(epoch_dir / "epoch_015.pt"),
    }
    resumed = _resume_command(stage)
    assert resumed[resumed.index("--resume-from") + 1].endswith("epoch_008.pt")
    assert "--reset-optimizer" not in resumed
    assert "--resume-start-epoch" not in resumed


def test_resume_epoch_override_requires_optimizer_reset():
    assert resolve_resume_start_epoch(3, 6, True) == 6
    assert resolve_resume_start_epoch(3, None, False) == 3
    with pytest.raises(ValueError, match="requires reset_optimizer"):
        resolve_resume_start_epoch(3, 6, False)
    with pytest.raises(ValueError, match="cannot precede"):
        resolve_resume_start_epoch(7, 6, True)


def test_controlled_stage_pause_is_explicit_and_does_not_change_contract(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--pause-after-stage",
            "base_e6",
        ]
    )
    assert args.pause_after_stage == "base_e6"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                str(tmp_path / "run"),
                "--pause-after-stage",
                "unknown",
            ]
        )


def test_staged_contract_freezes_selected_transition_and_window_subset(tmp_path):
    schedule = load_json(SCHEDULE)
    windows = _select_windows(schedule, ("oos_2024_01",))
    contract = build_contract(
        PROFILE, SCHEDULE, CACHE, windows, tmp_path, "cuda", transition="selected"
    )
    assert contract["transition_checkpoint"] == "selected"
    assert [item["window"]["name"] for item in contract["windows"]] == ["oos_2024_01"]
    command = contract["windows"][0]["stages"][1]["command"]
    assert command[command.index("--resume-start-epoch") + 1] == "6"


def test_staged_contract_can_predeclare_exact_and_selected_signal_profiles(tmp_path):
    schedule = load_json(SCHEDULE)
    windows = _select_windows(schedule, ("oos_2024_01",))
    contract = build_contract(
        PROFILE,
        SCHEDULE,
        CACHE,
        windows,
        tmp_path,
        "cuda",
        inference_profiles=("exact", "selected"),
    )
    spec = contract["windows"][0]

    assert contract["inference_profiles"] == ["exact", "selected"]
    assert set(spec["alpha_profiles"]) == {"exact", "selected"}
    assert spec["alpha_profiles"]["exact"]["checkpoint"].endswith("epoch_019.pt")
    assert spec["alpha_profiles"]["selected"]["checkpoint"].endswith("ultimate_v7_best.pt")


def test_full_contract_promotion_is_hash_stable_and_marks_required_gates(tmp_path):
    schedule = load_json(SCHEDULE)
    windows = _select_windows(schedule, ("oos_2024_01",))
    staged = build_contract(PROFILE, SCHEDULE, CACHE, windows, tmp_path, "cuda")

    first = finalize_full_rolling_contract(staged)
    second = finalize_full_rolling_contract(staged)

    assert first == second
    assert first["mode"] == "full_24_window_reconstructed_e19"
    assert first["resource_plan_required"] is True
    assert first["provenance_required"] is True
    assert len(first["contract_sha256"]) == 64


def test_strong_runner_emits_learner_neutral_rolling_manifest(tmp_path):
    alpha = tmp_path / "window_alpha.jsonl"
    alpha.write_text(
        '{"date":"2024-01-02","codes":["000001.SZ"],"alpha":[1.0]}\n',
        encoding="utf-8",
    )
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    contract = {
        "contract_sha256": "contract",
        "transition_checkpoint": "exact",
        "profile": {"path": "profile", "sha256": "p"},
        "schedule": {"path": "schedule", "sha256": "s"},
        "windows": [
            {
                "window": {
                    "name": "oos_2024_01",
                    "train_start": "2019-07-01",
                    "train_end": "2023-06-30",
                    "valid_start": "2023-07-03",
                    "valid_end": "2023-12-29",
                    "predict_start": "2024-01-02",
                    "predict_end": "2024-01-31",
                }
            }
        ],
    }
    progress = {
        "windows": {
            "oos_2024_01": {
                "alpha": {"path": str(alpha), "sha256": "a"},
                "stages": {
                    "multi_downside_e19": {
                        "selected_checkpoint": {"path": str(model), "sha256": "m"}
                    }
                },
            }
        }
    }

    path, manifest = build_standard_rolling_manifest(contract, progress, tmp_path / "out")

    assert path.is_file()
    assert manifest["learner_adapter"] == "torch_strong_alpha"
    assert manifest["windows"][0]["alpha_path"] == str(alpha)
    assert manifest["split_alpha_paths"]["val_2024"]["rows"] == 1


def test_strong_runner_emits_isolated_exact_profile_manifest(tmp_path):
    alpha = tmp_path / "exact_alpha.jsonl"
    alpha.write_text(
        '{"date":"2024-01-02","codes":["000001.SZ"],"alpha":[1.0]}\n',
        encoding="utf-8",
    )
    model = tmp_path / "epoch_019.pt"
    model.write_bytes(b"exact")
    window = {
        "name": "oos_2024_01",
        "train_start": "2019-07-01",
        "train_end": "2023-06-30",
        "valid_start": "2023-07-03",
        "valid_end": "2023-12-29",
        "predict_start": "2024-01-02",
        "predict_end": "2024-01-31",
    }
    contract = {
        "contract_sha256": "contract",
        "profile": {"path": "profile", "sha256": "p"},
        "schedule": {"path": "schedule", "sha256": "s"},
        "windows": [{"window": window}],
    }
    progress = {
        "windows": {
            "oos_2024_01": {
                "alphas": {"exact": {"path": str(alpha), "sha256": "a"}},
                "stages": {
                    "multi_downside_e19": {
                        "exact_checkpoint": {"path": str(model), "sha256": "m"}
                    }
                },
            }
        }
    }

    path, manifest = build_standard_rolling_manifest(
        contract, progress, tmp_path / "out", signal_profile="exact"
    )

    assert path.name == "rolling_manifest_exact.json"
    assert manifest["config"]["signal_profile"] == "exact"
    assert "profiles\\exact\\signals" in manifest["split_alpha_paths"]["val_2024"]["path"]
