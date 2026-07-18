"""Auditable reconstruction of the historical multi_downside_e19 profile."""

from __future__ import annotations

import json
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from experiments.recording import canonical_json_hash, sha256_file


EXPECTED_SOLVED_WEIGHTS = {
    "industry_loss_weight": 0.1,
    "multi_loss_weight": 0.1,
    "diversity_loss_weight": 0.05,
    "spread_loss_weight": 0.001,
    "downside_loss_weight": 0.2,
}

EARLY_STAGE_SHARED_FLAGS = (
    "--model",
    "--label-family",
    "--horizon-indices",
    "--horizon-weights",
    "--batch-size",
    "--val-batch-size",
    "--accum-steps",
    "--lag1-loss-weight",
    "--lag1-label-family",
    "--seed",
    "--best-val-metric",
)


def parse_saved_command(path: str | Path) -> dict[str, Any]:
    """Parse a saved argv without executing it."""

    source = Path(path)
    command = source.read_text(encoding="utf-8-sig").strip()
    if not command:
        raise ValueError(f"saved command is empty: {source}")
    argv = shlex.split(command, posix=False)
    flags: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                flags[token] = argv[index + 1]
                index += 2
                continue
            flags[token] = True
        index += 1
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "argv": argv,
        "flags": flags,
    }


def _require_equal_early_contract(stage_1: dict[str, Any], stage_2: dict[str, Any]) -> None:
    for flag in EARLY_STAGE_SHARED_FLAGS:
        left = stage_1["flags"].get(flag)
        right = stage_2["flags"].get(flag)
        if left != right:
            raise ValueError(f"historical early-stage command drift for {flag}: {left!r} != {right!r}")


def read_epoch_metrics(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 5:
        raise ValueError("at least five epoch rows are required to solve five loss weights")
    return rows


def solve_loss_weights(rows: list[dict[str, Any]], *, lag1_weight: float = 0.25) -> dict[str, Any]:
    matrix, target = [], []
    for row in rows:
        comp = row["train_components"]
        matrix.append(
            [
                comp["within_ic"] - comp["global_ic"],
                comp["multi"],
                comp["div"],
                comp["spread"],
                comp["downside"],
            ]
        )
        target.append(row["train_loss"] - comp["global_ic"] - lag1_weight * comp["lag1"])
    matrix = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    if rank < 5:
        raise ValueError("epoch component matrix is rank-deficient")
    residual = matrix @ weights - target
    names = tuple(EXPECTED_SOLVED_WEIGHTS)
    result = {name: float(value) for name, value in zip(names, weights)}
    max_error = float(np.max(np.abs(residual)))
    if max_error > 1e-8:
        raise ValueError(f"loss-weight reconstruction residual is too large: {max_error}")
    for name, expected in EXPECTED_SOLVED_WEIGHTS.items():
        if not np.isclose(result[name], expected, atol=1e-6):
            raise ValueError(f"reconstructed {name}={result[name]} differs from expected grid value {expected}")
        result[name] = expected
    return {"weights": result, "lag1_loss_weight": lag1_weight, "max_abs_residual": max_error}


def build_reconstructed_e19_profile(project_root: str | Path) -> dict[str, Any]:
    import torch

    root = Path(project_root).resolve()
    metrics_path = root / "checkpoints_v14_m0_oo_lag1_multi_downside" / "epochs" / "epoch_metrics.jsonl"
    checkpoint_path = root / "checkpoints_v14_m0_oo_lag1_multi_downside" / "epochs" / "epoch_019.pt"
    parent_command_path = root / "checkpoints_v14_m0_oo_lag1_w025_low_lr_e15" / "command.txt"
    parent_checkpoint_path = root / "checkpoints_v14_m0_oo_lag1_w025_low_lr_e15" / "epochs" / "epoch_015.pt"
    sources = (metrics_path, checkpoint_path, parent_command_path, parent_checkpoint_path)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    rows = read_epoch_metrics(metrics_path)
    solved = solve_loss_weights(rows, lag1_weight=0.25)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 19:
        raise ValueError("historical checkpoint is not epoch 19")
    if checkpoint.get("best_metric") != "rawtopstable_h5_top0p6":
        raise ValueError("historical checkpoint selection metric changed")

    profile = {
        "schema": "reconstructed_strong_model_profile_v1",
        "profile_id": "multi_downside_e19_reconstructed_v1",
        "historical_identity": {
            "status": "reconstructed_not_original_command",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "epoch": 19,
            "selection_metric": checkpoint["best_metric"],
            "selection_score": float(checkpoint["best_score"]),
        },
        "confirmed": {
            "architecture": checkpoint["arch_config"],
            "observed_epochs": [int(row["epoch"]) for row in rows],
            "observed_learning_rates": sorted({float(row["learning_rate"]) for row in rows}),
            "loss_weights_from_exact_component_equation": solved,
        },
        "inferred": {
            "model": "v9",
            "label_family": "oo",
            "lag1_label_family": "oo_lag1",
            "horizon_indices": [0, 2, 4, 6],
            "horizon_weights": [0.15, 0.25, 0.35, 0.25],
            "batch_size": 4,
            "val_batch_size": 2,
            "accum_steps": 4,
            "seed": 42,
            "stage_3_resume_parent": str(parent_checkpoint_path),
            "stage_3_reset_optimizer": True,
            "reason": "inherited from the adjacent parent command and historical continuation layout",
        },
        "rolling_reconstruction": {
            "window_policy": {"train_years": 4, "valid_months": 6, "oos_months": 1},
            "raw_signal_only": True,
            "predictor_mode": "none",
            "training_stages": [
                {
                    "name": "base_1_to_6",
                    "target_epoch": 6,
                    "learning_rate": 0.0001,
                    "status": "inferred_from_parent_chain",
                },
                {
                    "name": "base_7_to_15",
                    "target_epoch": 15,
                    "learning_rate": 0.00001,
                    "reset_optimizer": True,
                    "status": "inferred_from_parent_chain",
                },
                {
                    "name": "multi_downside_16_to_19",
                    "target_epoch": 19,
                    "learning_rate": 0.000005,
                    "reset_optimizer": True,
                    "status": "reconstructed_from_exact_epoch_16_20_equation",
                },
            ],
            "smoke": {"windows": 1, "epochs": 1, "purpose": "interface/runtime validation only"},
            "pilot": {"windows": 3, "epochs": 19, "promotion_allowed": False},
            "full": {"windows": 24, "epochs": 19, "requires_pilot_gate": True},
        },
        "evidence": [
            {"path": str(source), "sha256": sha256_file(source)}
            for source in sources
        ],
    }
    profile["profile_sha256"] = canonical_json_hash(profile)
    return profile


def build_hardened_e19_profile_v2(project_root: str | Path) -> dict[str, Any]:
    """Build a field-provenance profile without rewriting the immutable v1 profile."""

    root = Path(project_root).resolve()
    profile = deepcopy(build_reconstructed_e19_profile(root))
    e6_command_path = root / "checkpoints_v14_m0_oo_lag1_w025_e6" / "command.txt"
    e6_checkpoint_path = root / "checkpoints_v14_m0_oo_lag1_w025_e6" / "epochs" / "epoch_006.pt"
    e15_command_path = root / "checkpoints_v14_m0_oo_lag1_w025_low_lr_e15" / "command.txt"
    e15_checkpoint_path = root / "checkpoints_v14_m0_oo_lag1_w025_low_lr_e15" / "epochs" / "epoch_015.pt"
    for source in (e6_command_path, e6_checkpoint_path, e15_command_path, e15_checkpoint_path):
        if not source.is_file():
            raise FileNotFoundError(source)

    e6_command = parse_saved_command(e6_command_path)
    e15_command = parse_saved_command(e15_command_path)
    _require_equal_early_contract(e6_command, e15_command)
    flags = e6_command["flags"]

    profile["schema"] = "reconstructed_strong_model_profile_v2"
    profile["profile_id"] = "multi_downside_e19_hardened_v2"
    profile["resolved_training_contract"] = {
        "model": flags["--model"],
        "label_family": flags["--label-family"],
        "lag1_label_family": flags["--lag1-label-family"],
        "horizon_indices": [int(value) for value in flags["--horizon-indices"].split(",")],
        "horizon_weights": [float(value) for value in flags["--horizon-weights"].split(",")],
        "batch_size": int(flags["--batch-size"]),
        "val_batch_size": int(flags["--val-batch-size"]),
        "accum_steps": int(flags["--accum-steps"]),
        "seed": int(flags["--seed"]),
        "selection_metric": flags["--best-val-metric"],
    }
    profile["confirmed"]["original_early_stage_commands"] = {
        "oo_lag1_e1_e6": e6_command,
        "oo_lag1_low_lr_e7_e15": e15_command,
    }
    profile["reconstructed_final_stage"] = {
        "canonical_stage_id": "multi_downside_e16_e19",
        "original_command_available": False,
        "learning_rate": profile["confirmed"]["observed_learning_rates"][0],
        "loss_contract": profile["confirmed"]["loss_weights_from_exact_component_equation"],
        "resume_parent": str(e15_checkpoint_path.resolve()),
        "resume_parent_sha256": sha256_file(e15_checkpoint_path),
        "reset_optimizer": {
            "value": True,
            "provenance": "inherited_from_historical_continuation_layout_not_original_e19_command",
        },
    }
    profile["field_provenance"] = {
        "architecture": "confirmed_epoch_019_checkpoint",
        "epochs_1_to_6_command": "confirmed_original_command",
        "epochs_7_to_15_command": "confirmed_original_command",
        "epochs_16_to_19_learning_rate": "confirmed_epoch_metrics",
        "epochs_16_to_19_loss_weights": "reconstructed_exact_component_equation",
        "epochs_16_to_19_resume_and_optimizer": "inherited_not_original_command",
        "rolling_window_policy": "project_research_arm_not_historical_model_identity",
    }
    profile["inferred"]["status"] = "legacy_consumer_view"
    profile["inferred"]["reason"] = (
        "retained for v1 consumer compatibility; use resolved_training_contract and "
        "field_provenance for v2 interpretation"
    )
    profile["rolling_reconstruction"]["training_stages"] = [
        {
            "legacy_stage_id": "base_e6",
            "canonical_stage_id": "oo_lag1_e1_e6",
            "target_epoch": 6,
            "learning_rate": 0.0001,
            "effective_loss_weights": {"global_ic": 1.0, "lag1_ic": 0.25},
            "status": "confirmed_original_command",
        },
        {
            "legacy_stage_id": "lag1_low_lr_e15",
            "canonical_stage_id": "oo_lag1_low_lr_e7_e15",
            "target_epoch": 15,
            "learning_rate": 0.00001,
            "effective_loss_weights": {"global_ic": 1.0, "lag1_ic": 0.25},
            "reset_optimizer": True,
            "status": "confirmed_original_command",
        },
        {
            "legacy_stage_id": "multi_downside_e19",
            "canonical_stage_id": "multi_downside_e16_e19",
            "target_epoch": 19,
            "learning_rate": 0.000005,
            "effective_loss_weights": {
                "global_ic": 0.9,
                "within_industry_ic": 0.1,
                "multi_horizon_ic": 0.1,
                "alpha_diversity": 0.05,
                "top_bottom_spread": 0.001,
                "downside_top": 0.2,
                "lag1_ic": 0.25,
            },
            "reset_optimizer": True,
            "status": "reconstructed_exact_loss_inherited_runtime",
        },
    ]
    evidence_by_path = {item["path"]: item for item in profile["evidence"]}
    for source in (e6_command_path, e6_checkpoint_path, e15_command_path, e15_checkpoint_path):
        resolved = str(source.resolve())
        evidence_by_path[resolved] = {"path": resolved, "sha256": sha256_file(source)}
    profile["evidence"] = [evidence_by_path[path] for path in sorted(evidence_by_path)]
    profile.pop("profile_sha256", None)
    profile["profile_sha256"] = canonical_json_hash(profile)
    return profile


def write_reconstructed_e19_profile(project_root: str | Path, output_path: str | Path) -> Path:
    profile = build_reconstructed_e19_profile(project_root)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_hardened_e19_profile_v2(project_root: str | Path, output_path: str | Path) -> Path:
    profile = build_hardened_e19_profile_v2(project_root)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
