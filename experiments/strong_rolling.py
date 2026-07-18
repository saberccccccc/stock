"""Commands and resumable state for strong-model monthly Rolling/OOF."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from experiments.recording import canonical_json_hash, sha256_file
from experiments.rolling import RollingWindow, build_split_alpha_files


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finalize_full_rolling_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Promote a validated staged contract to the immutable 24-window shape."""

    result = dict(contract)
    result["mode"] = "full_24_window_reconstructed_e19"
    result["resource_plan_required"] = True
    result["provenance_required"] = True
    result["contract_sha256"] = canonical_json_hash(
        {key: value for key, value in result.items() if key != "contract_sha256"}
    )
    return result


def build_standard_rolling_manifest(
    contract: Mapping[str, Any],
    progress: Mapping[str, Any],
    output_dir: str | Path,
    *,
    signal_profile: str = "selected",
) -> tuple[Path, dict[str, Any]]:
    """Expose a staged strong run through the common rolling artifact shape."""

    output = Path(output_dir).resolve()
    if signal_profile not in {"selected", "exact"}:
        raise ValueError("signal_profile must be 'selected' or 'exact'")
    profile_output = output if signal_profile == "selected" else output / "profiles" / signal_profile
    entries = []
    for spec in contract["windows"]:
        window = spec["window"]
        state = progress["windows"][window["name"]]
        alpha = (
            state["alpha"]
            if signal_profile == "selected"
            else state.get("alphas", {}).get(signal_profile)
        )
        if not alpha:
            raise ValueError(
                f"window {window['name']} has no completed {signal_profile} alpha"
            )
        checkpoint = state["stages"]["multi_downside_e19"][
            f"{signal_profile}_checkpoint"
        ]
        entries.append(
            {
                **window,
                "alpha_path": alpha["path"],
                "alpha_sha256": alpha["sha256"],
                "model_path": checkpoint["path"],
                "model_sha256": checkpoint["sha256"],
            }
        )
    split_alpha = build_split_alpha_files(entries, profile_output)
    manifest = {
        "schema_version": 1,
        "run_mode": "formal",
        "learner_adapter": "torch_strong_alpha",
        "contract_sha256": contract["contract_sha256"],
        "config": {
            "profile": contract["profile"],
            "schedule": contract["schedule"],
            "transition_checkpoint": contract.get("transition_checkpoint", "exact"),
            "signal_profile": signal_profile,
            "windows": [spec["window"] for spec in contract["windows"]],
        },
        "windows": entries,
        "split_alpha_paths": split_alpha,
    }
    path = output / (
        "rolling_manifest.json"
        if signal_profile == "selected"
        else f"rolling_manifest_{signal_profile}.json"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path, manifest


def validate_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema") not in {
        "reconstructed_strong_model_profile_v1",
        "reconstructed_strong_model_profile_v2",
    }:
        raise ValueError("unsupported strong-model profile")
    if profile.get("historical_identity", {}).get("status") != "reconstructed_not_original_command":
        raise ValueError("strong profile must preserve reconstructed provenance")
    solved = profile["confirmed"]["loss_weights_from_exact_component_equation"]
    if float(solved["max_abs_residual"]) > 1e-8:
        raise ValueError("strong profile loss reconstruction is not exact")
    if profile["rolling_reconstruction"].get("raw_signal_only") is not True:
        raise ValueError("Q5 first acceptance run must use raw signal")
    if profile.get("schema") == "reconstructed_strong_model_profile_v2":
        stages = profile["rolling_reconstruction"]["training_stages"]
        canonical = [stage.get("canonical_stage_id") for stage in stages]
        if canonical != [
            "oo_lag1_e1_e6",
            "oo_lag1_low_lr_e7_e15",
            "multi_downside_e16_e19",
        ]:
            raise ValueError("hardened strong profile has unexpected canonical stages")


def _flag(command: list[str], name: str, value: Any) -> None:
    command.extend([name, str(value)])


def build_smoke_train_command(
    *,
    python: str | Path,
    profile: Mapping[str, Any],
    window: RollingWindow,
    output_dir: str | Path,
    cache_meta: str | Path,
    epochs: int = 1,
    device: str = "cuda",
) -> list[str]:
    validate_profile(profile)
    inferred = profile["inferred"]
    weights = profile["confirmed"]["loss_weights_from_exact_component_equation"]["weights"]
    command = [str(python), "-X", "utf8", "-u", "run/train.py"]
    values = (
        ("--model", inferred["model"]),
        ("--label-family", inferred["label_family"]),
        ("--horizon-indices", ",".join(map(str, inferred["horizon_indices"]))),
        ("--horizon-weights", ",".join(map(str, inferred["horizon_weights"]))),
        ("--epochs", epochs),
        ("--lr", 0.000005),
        ("--device", device),
        ("--batch-size", inferred["batch_size"]),
        ("--val-batch-size", inferred["val_batch_size"]),
        ("--accum-steps", inferred["accum_steps"]),
        ("--memmap-trim-interval", 32),
        ("--industry-loss-weight", weights["industry_loss_weight"]),
        ("--multi-loss-weight", weights["multi_loss_weight"]),
        ("--diversity-loss-weight", weights["diversity_loss_weight"]),
        ("--spread-loss-weight", weights["spread_loss_weight"]),
        ("--spread-delay-epochs", 0),
        ("--top-focus-loss-weight", 0),
        ("--downside-loss-weight", weights["downside_loss_weight"]),
        ("--downside-delay-epochs", 0),
        ("--lag1-loss-weight", profile["confirmed"]["loss_weights_from_exact_component_equation"]["lag1_loss_weight"]),
        ("--lag1-label-family", inferred["lag1_label_family"]),
        ("--lag1-delay-epochs", 0),
        ("--lag1-top-focus-loss-weight", 0),
        ("--pairwise-top-loss-weight", 0),
        ("--best-val-metric", profile["historical_identity"]["selection_metric"]),
        ("--eval-top-fracs", "0.006,0.01,0.02,0.05,0.10"),
        ("--train-start", window.train_start),
        ("--train-label-end", window.train_end),
        ("--val-start", window.valid_start),
        ("--val-label-end", window.valid_end),
        ("--early-stop-patience", 1),
        ("--seed", inferred["seed"]),
        ("--output-dir", Path(output_dir).resolve()),
        ("--cache-meta", Path(cache_meta).resolve()),
        ("--expected-input-dim", profile["confirmed"]["architecture"]["input_dim"]),
    )
    for name, value in values:
        _flag(command, name, value)
    command.append("--save-every-epoch")
    return command


def _replace_flag(command: list[str], name: str, value: Any) -> None:
    index = command.index(name)
    command[index + 1] = str(value)


def build_reconstructed_training_stages(
    *,
    python: str | Path,
    profile: Mapping[str, Any],
    window: RollingWindow,
    output_dir: str | Path,
    cache_meta: str | Path,
    device: str = "cuda",
    transition_checkpoint: str = "exact",
) -> list[dict[str, Any]]:
    """Build the verified 1-6, 7-15, and 16-19 e19 lineage."""
    if transition_checkpoint not in {"exact", "selected"}:
        raise ValueError("transition_checkpoint must be 'exact' or 'selected'")
    root = Path(output_dir).resolve()
    stage_specs = (
        ("base_e6", "oo_lag1_e1_e6", 6, 0.0001, False),
        ("lag1_low_lr_e15", "oo_lag1_low_lr_e7_e15", 15, 0.00001, False),
        ("multi_downside_e19", "multi_downside_e16_e19", 19, 0.000005, True),
    )
    stages = []
    previous_checkpoint = None
    previous_boundary = None
    for name, canonical_name, epochs, lr, enhanced_losses in stage_specs:
        stage_dir = root / name
        command = build_smoke_train_command(
            python=python,
            profile=profile,
            window=window,
            output_dir=stage_dir,
            cache_meta=cache_meta,
            epochs=epochs,
            device=device,
        )
        _replace_flag(command, "--lr", lr)
        _replace_flag(command, "--early-stop-patience", epochs)
        if not enhanced_losses:
            for flag in (
                "--industry-loss-weight",
                "--multi-loss-weight",
                "--diversity-loss-weight",
                "--spread-loss-weight",
                "--downside-loss-weight",
            ):
                _replace_flag(command, flag, 0)
        if previous_checkpoint is not None:
            command.extend(["--resume-from", str(previous_checkpoint), "--reset-optimizer"])
            if transition_checkpoint == "selected":
                command.extend(["--resume-start-epoch", str(previous_boundary)])
        exact_checkpoint = stage_dir / "epochs" / f"epoch_{epochs:03d}.pt"
        stage_payload = {
                "name": name,
                "target_epoch": epochs,
                "command": command,
                "exact_checkpoint": str(exact_checkpoint),
                "selected_checkpoint": str(stage_dir / "ultimate_v7_best.pt"),
            }
        if profile.get("schema") == "reconstructed_strong_model_profile_v2":
            stage_payload.update(
                {
                    "legacy_stage_id": name,
                    "canonical_stage_id": canonical_name,
                }
            )
        stages.append(stage_payload)
        previous_checkpoint = (
            stage_dir / "ultimate_v7_best.pt"
            if transition_checkpoint == "selected"
            else exact_checkpoint
        )
        previous_boundary = epochs
    return stages


def build_raw_inference_command(
    *,
    python: str | Path,
    window: RollingWindow,
    checkpoint: str | Path,
    output: str | Path,
    cache_meta: str | Path,
    expected_input_dim: int = 250,
    device: str = "cuda",
) -> list[str]:
    return [
        str(python),
        "-X",
        "utf8",
        "-u",
        "run/generate_v9_inference_alpha.py",
        "--checkpoint",
        str(Path(checkpoint).resolve()),
        "--start-date",
        window.predict_start,
        "--emit-start-date",
        window.predict_start,
        "--end-date",
        window.predict_end,
        "--output",
        str(Path(output).resolve()),
        "--predictor-mode",
        "none",
        "--device",
        device,
        "--cache-meta",
        str(Path(cache_meta).resolve()),
        "--expected-input-dim",
        str(expected_input_dim),
    ]


def build_smoke_contract(
    *,
    profile_path: str | Path,
    schedule_path: str | Path,
    window: RollingWindow,
    train_command: list[str],
    inference_command: list[str],
    cache_meta: str | Path,
) -> dict[str, Any]:
    contract = {
        "schema": "strong_rolling_smoke_v1",
        "mode": "smoke_one_window_one_epoch",
        "profile": {"path": str(Path(profile_path).resolve()), "sha256": sha256_file(profile_path)},
        "schedule": {"path": str(Path(schedule_path).resolve()), "sha256": sha256_file(schedule_path)},
        "data_cache": {"path": str(Path(cache_meta).resolve()), "sha256": sha256_file(cache_meta)},
        "window": window.__dict__,
        "train_command": train_command,
        "inference_command": inference_command,
        "selection_allowed": False,
        "promotion_allowed": False,
    }
    contract["contract_sha256"] = canonical_json_hash(contract)
    return contract


def run_with_memory_guard(
    command: list[str],
    *,
    cwd: str | Path,
    min_free_gib: float = 0.75,
    poll_seconds: float = 5.0,
) -> None:
    """Run a child while protecting a 16 GiB workstation from exhaustion."""
    import psutil

    process = subprocess.Popen(command, cwd=cwd)
    low_readings = 0
    try:
        while process.poll() is None:
            free_gib = psutil.virtual_memory().available / (1024**3)
            low_readings = low_readings + 1 if free_gib < min_free_gib else 0
            if low_readings >= 3:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise MemoryError(
                    f"memory guard stopped child after three readings below {min_free_gib:.2f} GiB"
                )
            time.sleep(poll_seconds)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        raise
