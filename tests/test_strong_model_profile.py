import json

import numpy as np

from experiments.recording import canonical_json_hash
from experiments.strong_model_profile import (
    EXPECTED_SOLVED_WEIGHTS,
    build_hardened_e19_profile_v2,
    build_reconstructed_e19_profile,
    parse_saved_command,
    solve_loss_weights,
)


def _synthetic_rows():
    rows = []
    rng = np.random.default_rng(7)
    for epoch in range(5):
        comp = {
            "global_ic": float(rng.normal()),
            "within_ic": float(rng.normal()),
            "multi": float(rng.normal()),
            "div": float(rng.normal()),
            "spread": float(rng.normal()),
            "downside": float(rng.normal()),
            "lag1": float(rng.normal()),
        }
        total = (
            comp["global_ic"]
            + 0.1 * (comp["within_ic"] - comp["global_ic"])
            + 0.1 * comp["multi"]
            + 0.05 * comp["div"]
            + 0.001 * comp["spread"]
            + 0.2 * comp["downside"]
            + 0.25 * comp["lag1"]
        )
        rows.append({"train_components": comp, "train_loss": total})
    return rows


def test_loss_weights_are_solved_from_component_equation():
    result = solve_loss_weights(_synthetic_rows())
    assert result["weights"] == EXPECTED_SOLVED_WEIGHTS
    assert result["lag1_loss_weight"] == 0.25
    assert result["max_abs_residual"] < 1e-8


def test_live_e19_profile_is_hashed_and_never_claims_original_command(project_root):
    profile = build_reconstructed_e19_profile(project_root)

    assert profile["historical_identity"]["status"] == "reconstructed_not_original_command"
    assert profile["historical_identity"]["epoch"] == 19
    assert profile["confirmed"]["loss_weights_from_exact_component_equation"]["weights"] == EXPECTED_SOLVED_WEIGHTS
    assert profile["inferred"]["label_family"] == "oo"
    assert profile["inferred"]["lag1_label_family"] == "oo_lag1"
    assert len(profile["evidence"]) == 4
    assert all(len(item["sha256"]) == 64 for item in profile["evidence"])
    assert len(profile["profile_sha256"]) == 64


def test_saved_early_commands_prove_lag1_was_present_from_epoch_one(project_root):
    command = parse_saved_command(
        project_root / "checkpoints_v14_m0_oo_lag1_w025_e6" / "command.txt"
    )

    assert command["flags"]["--label-family"] == "oo"
    assert command["flags"]["--lag1-label-family"] == "oo_lag1"
    assert float(command["flags"]["--lag1-loss-weight"]) == 0.25
    assert float(command["flags"]["--multi-loss-weight"]) == 0.0


def test_hardened_profile_separates_confirmed_and_reconstructed_fields(project_root):
    profile = build_hardened_e19_profile_v2(project_root)

    assert profile["schema"] == "reconstructed_strong_model_profile_v2"
    assert profile["historical_identity"]["status"] == "reconstructed_not_original_command"
    assert profile["resolved_training_contract"]["horizon_indices"] == [0, 2, 4, 6]
    assert profile["field_provenance"]["epochs_1_to_6_command"] == "confirmed_original_command"
    assert profile["field_provenance"]["epochs_16_to_19_loss_weights"] == "reconstructed_exact_component_equation"
    assert profile["reconstructed_final_stage"]["original_command_available"] is False
    stages = profile["rolling_reconstruction"]["training_stages"]
    assert stages[0]["effective_loss_weights"]["lag1_ic"] == 0.25
    assert stages[1]["effective_loss_weights"] == stages[0]["effective_loss_weights"]
    assert stages[2]["canonical_stage_id"] == "multi_downside_e16_e19"
    assert len(profile["evidence"]) == 6
    assert len(profile["profile_sha256"]) == 64
    payload = dict(profile)
    stored_hash = payload.pop("profile_sha256")
    assert canonical_json_hash(payload) == stored_hash


def test_live_v1_profile_file_remains_byte_semantically_compatible(project_root):
    stored = json.loads(
        (project_root / "configs" / "reconstructed_multi_downside_e19_profile_v1.json").read_text(
            encoding="utf-8"
        )
    )

    rebuilt = json.loads(json.dumps(build_reconstructed_e19_profile(project_root)))
    assert rebuilt == stored


def pytest_generate_tests(metafunc):
    if "project_root" in metafunc.fixturenames:
        from pathlib import Path

        metafunc.parametrize("project_root", [Path(__file__).resolve().parents[1]])
