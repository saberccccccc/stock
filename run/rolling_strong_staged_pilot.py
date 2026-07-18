"""Run the resumable three-window, three-stage e19 reconstruction pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
root_path = str(ROOT)
if root_path in sys.path:
    sys.path.remove(root_path)
sys.path.insert(0, root_path)

from experiments.recording import (
    MANIFEST_NAME,
    append_event,
    canonical_json_hash,
    create_experiment,
    finalize_artifact_index,
    load_events,
    record_artifact,
    sha256_file,
)
from data.cache_metadata import load_explicit_cross_section_meta
from data.transform_contract import build_v14_transform_contract
from experiments.provenance import (
    RuntimeMetricsSampler,
    command_manifest,
    environment_manifest,
    source_manifest,
    validate_provenance_bundle,
    write_provenance_bundle,
)
from experiments.rolling import RollingWindow, validate_window
from experiments.strong_adapter_runtime import (
    build_strong_window_dataset,
    fit_strong_stage_through_adapter,
)
from experiments.strong_rolling import (
    build_standard_rolling_manifest,
    build_raw_inference_command,
    build_reconstructed_training_stages,
    finalize_full_rolling_contract,
    load_json,
    run_with_memory_guard,
    validate_profile,
)


PILOT_WINDOWS = ("oos_2024_01", "oos_2025_01", "oos_2025_12")
PROGRESS_SCHEMA = "strong_rolling_staged_progress_v1"
STAGE_NAMES = ("base_e6", "lag1_low_lr_e15", "multi_downside_e19")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="configs/reconstructed_multi_downside_e19_profile_v1.json")
    parser.add_argument("--schedule", default="configs/monthly_rolling_compact_4y6m1m_2024_2025.json")
    parser.add_argument("--cache-meta", default="cache/cross_section_v14_multilabel_open_tech_market_funda_shareh_restr_macro_all_s40_t5_h10_min30_mad_rawlab_end20260518_meta.pkl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--transition", choices=("exact", "selected"), default="exact")
    parser.add_argument(
        "--inference-profiles",
        default="selected",
        help="Comma-separated final checkpoint profiles: selected or exact,selected.",
    )
    parser.add_argument("--windows", default=",".join(PILOT_WINDOWS))
    parser.add_argument(
        "--launch-plan",
        help="Require the rebuilt full contract to match this immutable launch plan.",
    )
    parser.add_argument("--min-free-gib", type=float, default=0.75)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pause-after-stage",
        choices=STAGE_NAMES,
        help="Persist the completed stage and exit cleanly; resume with --resume.",
    )
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _select_windows(schedule: dict[str, Any], names=PILOT_WINDOWS) -> list[RollingWindow]:
    by_name = {item["name"]: RollingWindow.from_mapping(item) for item in schedule["windows"]}
    windows = [by_name[name] for name in names]
    for window in windows:
        validate_window(window, research_end=schedule["data"]["research_end"])
    return windows


def build_contract(
    profile_path,
    schedule_path,
    cache_meta,
    windows,
    output,
    device,
    transition="exact",
    inference_profiles=("selected",),
):
    profile = load_json(profile_path)
    inference_profiles = tuple(dict.fromkeys(str(value) for value in inference_profiles))
    if not inference_profiles or any(value not in {"exact", "selected"} for value in inference_profiles):
        raise ValueError("inference_profiles must contain only exact and selected")
    specs = []
    for window in windows:
        window_dir = output / "windows" / window.name
        stages = build_reconstructed_training_stages(
            python=sys.executable,
            profile=profile,
            window=window,
            output_dir=window_dir / "training",
            cache_meta=cache_meta,
            device=device,
            transition_checkpoint=transition,
        )
        selected = Path(stages[-1]["selected_checkpoint"])
        exact = Path(stages[-1]["exact_checkpoint"])
        alpha = window_dir / "alpha_raw.jsonl"
        spec = {
                "window": window.__dict__,
                "stages": stages,
                "selected_checkpoint": str(selected),
                "alpha": str(alpha),
                "inference_command": build_raw_inference_command(
                    python=sys.executable,
                    window=window,
                    checkpoint=selected,
                    output=alpha,
                    cache_meta=cache_meta,
                    expected_input_dim=profile["confirmed"]["architecture"]["input_dim"],
                    device=device,
                ),
            }
        if inference_profiles != ("selected",):
            profile_specs = {}
            for signal_profile in inference_profiles:
                checkpoint = selected if signal_profile == "selected" else exact
                profile_alpha = alpha if signal_profile == "selected" else window_dir / "alpha_exact.jsonl"
                profile_specs[signal_profile] = {
                    "checkpoint": str(checkpoint),
                    "alpha": str(profile_alpha),
                    "inference_command": build_raw_inference_command(
                        python=sys.executable,
                        window=window,
                        checkpoint=checkpoint,
                        output=profile_alpha,
                        cache_meta=cache_meta,
                        expected_input_dim=profile["confirmed"]["architecture"]["input_dim"],
                        device=device,
                    ),
                }
            spec["alpha_profiles"] = profile_specs
        specs.append(spec)
    contract = {
        "schema": "strong_rolling_staged_pilot_v1",
        "mode": "predeclared_windows_reconstructed_e19",
        "transition_checkpoint": transition,
        "profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
        "schedule": {"path": str(schedule_path), "sha256": sha256_file(schedule_path)},
        "data_cache": {"path": str(cache_meta), "sha256": sha256_file(cache_meta)},
        "windows": specs,
        "raw_signal_only": True,
        "selection_allowed": False,
        "promotion_allowed": False,
    }
    if inference_profiles != ("selected",):
        contract["inference_profiles"] = list(inference_profiles)
    contract["contract_sha256"] = canonical_json_hash(contract)
    return contract


def _load_progress(path: Path, contract_hash: str) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": PROGRESS_SCHEMA, "contract_sha256": contract_hash, "windows": {}}
    progress = json.loads(path.read_text(encoding="utf-8-sig"))
    if progress.get("schema") != PROGRESS_SCHEMA:
        raise ValueError("unsupported staged-pilot progress schema")
    if progress.get("contract_sha256") != contract_hash:
        raise ValueError("staged-pilot progress does not match the frozen contract")
    return progress


def _verify_checkpoint(path: Path, *, expected_epoch: int, expected_input_dim: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != int(expected_epoch):
        raise ValueError(f"checkpoint epoch mismatch: expected={expected_epoch} path={path}")
    actual_dim = int(checkpoint.get("arch_config", {}).get("input_dim", -1))
    if actual_dim != int(expected_input_dim):
        raise ValueError(f"checkpoint input_dim mismatch: expected={expected_input_dim} actual={actual_dim}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "epoch": expected_epoch,
        "input_dim": actual_dim,
        "selection_metric": checkpoint.get("best_metric"),
        "selection_score": checkpoint.get("best_score"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
    }


def _latest_stage_checkpoint(stage: dict[str, Any]) -> Path | None:
    exact = Path(stage["exact_checkpoint"])
    files = sorted(exact.parent.glob("epoch_*.pt")) if exact.parent.is_dir() else []
    return files[-1] if files else None


def _resume_command(stage: dict[str, Any]) -> list[str]:
    command = list(stage["command"])
    latest = _latest_stage_checkpoint(stage)
    if latest is None:
        return command
    if "--resume-from" in command:
        index = command.index("--resume-from")
        command[index + 1] = str(latest)
    else:
        command.extend(["--resume-from", str(latest)])
    if "--reset-optimizer" in command:
        command.remove("--reset-optimizer")
    if "--resume-start-epoch" in command:
        index = command.index("--resume-start-epoch")
        del command[index : index + 2]
    return command


def _already_recorded(output: Path, name: str) -> bool:
    return any(
        event.get("event_type") == "artifact_recorded" and event.get("details", {}).get("name") == name
        for event in load_events(output)
    )


def _record_once(output: Path, *, name: str, path: Path, kind: str) -> None:
    if not _already_recorded(output, name):
        record_artifact(output, name=name, path=path, kind=kind)


def _validate_alpha(path: Path, window: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"alpha file is empty: {path}")
    dates = [pd.Timestamp(row["date"]).normalize() for row in rows]
    if len(dates) != len(set(dates)):
        raise ValueError(f"duplicate dates inside alpha: {path}")
    start, end = pd.Timestamp(window["predict_start"]), pd.Timestamp(window["predict_end"])
    if min(dates) < start or max(dates) > end:
        raise ValueError(f"alpha dates escape window {window['name']}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "date_start": str(min(dates).date()),
        "date_end": str(max(dates).date()),
    }


def _stitch(contract: dict[str, Any], output: Path) -> Path:
    owners = {}
    rows = []
    for spec in contract["windows"]:
        for line in Path(spec["alpha"]).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(pd.Timestamp(row["date"]).normalize().date())
            if key in owners:
                raise ValueError(f"duplicate OOS owner for {key}: {owners[key]} and {spec['window']['name']}")
            owners[key] = spec["window"]["name"]
            rows.append(row)
    rows.sort(key=lambda item: pd.Timestamp(item["date"]))
    path = output / "oos_alpha_staged_pilot.jsonl"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return path


def main(argv=None):
    args = parse_args(argv)
    profile_path = (ROOT / args.profile).resolve()
    schedule_path = (ROOT / args.schedule).resolve()
    cache_meta = (ROOT / args.cache_meta).resolve()
    output = (ROOT / args.output_dir).resolve()
    profile = load_json(profile_path)
    schedule = load_json(schedule_path)
    validate_profile(profile)
    if args.windows.strip().lower() == "all":
        window_names = tuple(item["name"] for item in schedule["windows"])
    else:
        window_names = tuple(name.strip() for name in args.windows.split(",") if name.strip())
    if not window_names:
        raise ValueError("--windows must name at least one rolling window")
    windows = _select_windows(schedule, window_names)
    inference_profiles = tuple(
        name.strip() for name in args.inference_profiles.split(",") if name.strip()
    )
    contract = build_contract(
        profile_path,
        schedule_path,
        cache_meta,
        windows,
        output,
        args.device,
        args.transition,
        inference_profiles,
    )
    launch_plan_path = None
    if args.launch_plan:
        launch_plan_path = (ROOT / args.launch_plan).resolve()
        launch_plan = load_json(launch_plan_path)
        if launch_plan.get("disk_gate_pass") is not True:
            raise ValueError("launch plan did not pass its disk reserve gate")
        if Path(launch_plan["planned_output_dir"]).resolve() != output:
            raise ValueError("launch plan output directory differs from --output-dir")
        rebuilt_full = finalize_full_rolling_contract(contract)
        planned_full = launch_plan.get("full_contract")
        if not isinstance(planned_full, dict):
            raise ValueError("launch plan has no full_contract")
        if rebuilt_full["contract_sha256"] != planned_full.get("contract_sha256"):
            raise ValueError("runtime full contract differs from the immutable launch plan")
        if canonical_json_hash(rebuilt_full) != canonical_json_hash(planned_full):
            raise ValueError("runtime full contract payload differs from the immutable launch plan")
        contract = rebuilt_full
    if args.dry_run:
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return

    contract_path = output / "staged_pilot_contract.json"
    progress_path = output / "staged_pilot_progress.json"
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"staged pilot output exists; pass --resume: {output}")
        if not (output / MANIFEST_NAME).is_file() or not contract_path.is_file():
            raise FileNotFoundError("resume requires both experiment manifest and staged contract")
        frozen = json.loads(contract_path.read_text(encoding="utf-8-sig"))
        if frozen.get("contract_sha256") != contract["contract_sha256"]:
            raise ValueError("resume contract differs from the frozen staged pilot")
        append_event(output, status="running", event_type="strong_staged_pilot_resumed")
    else:
        output.mkdir(parents=True)
        _atomic_json(contract_path, contract)
        create_experiment(
            output,
            experiment_id=output.name,
            config=contract,
            protocol={"selection_allowed": False, "promotion_allowed": False},
            cache_contract={
                "profile": contract["profile"],
                "schedule": contract["schedule"],
                "data_cache": contract["data_cache"],
                "data_scope": {
                    "effective_start": windows[0].train_start,
                    "effective_end": windows[-1].predict_end,
                    "physical_coverage": "2010-01-04..2026-05-18",
                    "role": "rolling_train_valid_oos_reconstruction_pilot",
                },
            },
            project_root=ROOT,
        )
        record_artifact(output, name="staged_pilot_contract", path=contract_path, kind="strong_rolling_contract")
        if launch_plan_path is not None:
            record_artifact(
                output,
                name="full_rolling_launch_plan",
                path=launch_plan_path,
                kind="strong_full_rolling_launch_plan",
            )

    progress = _load_progress(progress_path, contract["contract_sha256"])
    expected_dim = int(profile["confirmed"]["architecture"]["input_dim"])
    runtime_sampler = RuntimeMetricsSampler(interval_seconds=0.5)
    runtime_sampler.__enter__()
    runtime_finished = False
    try:
        for spec in contract["windows"]:
            window = spec["window"]
            name = window["name"]
            state = progress["windows"].setdefault(name, {"stages": {}, "alpha": None})
            dataset = None
            dataset_contract = None
            for stage in spec["stages"]:
                stage_name = stage["name"]
                exact = Path(stage["exact_checkpoint"])
                selected = Path(stage["selected_checkpoint"])
                existing = state["stages"].get(stage_name)
                if existing:
                    if sha256_file(exact) != existing["exact_checkpoint"]["sha256"]:
                        raise ValueError(f"completed checkpoint hash drift: {exact}")
                    if sha256_file(selected) != existing["selected_checkpoint"]["sha256"]:
                        raise ValueError(f"completed selected-checkpoint hash drift: {selected}")
                    _record_once(output, name=f"{name}:{stage_name}:exact", path=exact, kind="torch_epoch_checkpoint")
                    _record_once(output, name=f"{name}:{stage_name}:selected", path=selected, kind="torch_selected_checkpoint")
                    continue
                append_event(output, status="running", event_type="strong_stage_started", details={"window": name, "stage": stage_name})
                if dataset is None:
                    dataset, dataset_contract = build_strong_window_dataset(
                        project_root=ROOT,
                        cache_meta=cache_meta,
                        profile=profile,
                        schedule=schedule,
                        window=RollingWindow.from_mapping(window),
                    )
                adapter_metrics = fit_strong_stage_through_adapter(
                    dataset=dataset,
                    stage=stage,
                    command=_resume_command(stage),
                    project_root=ROOT,
                    horizon_indices=profile["inferred"]["horizon_indices"],
                    expected_input_dim=expected_dim,
                    device=args.device,
                    min_free_gib=args.min_free_gib,
                    model_id=f"{name}:{stage_name}",
                )
                exact_info = _verify_checkpoint(exact, expected_epoch=stage["target_epoch"], expected_input_dim=expected_dim)
                if not selected.is_file():
                    raise FileNotFoundError(selected)
                selected_info = {"path": str(selected.resolve()), "sha256": sha256_file(selected)}
                state["stages"][stage_name] = {
                    "exact_checkpoint": exact_info,
                    "selected_checkpoint": selected_info,
                    "adapter_fit": adapter_metrics,
                }
                if dataset_contract is not None:
                    state["dataset"] = dataset_contract
                _atomic_json(progress_path, progress)
                _record_once(output, name=f"{name}:{stage_name}:exact", path=exact, kind="torch_epoch_checkpoint")
                _record_once(output, name=f"{name}:{stage_name}:selected", path=selected, kind="torch_selected_checkpoint")
                append_event(output, status="running", event_type="strong_stage_completed", details={"window": name, "stage": stage_name, "epoch": stage["target_epoch"]})
                if args.pause_after_stage == stage_name:
                    runtime_sampler.finish(
                        status="paused",
                        details={
                            "window": name,
                            "stage": stage_name,
                            "resume_enabled": True,
                            "reason": "predeclared_pause_after_stage",
                        },
                    )
                    runtime_finished = True
                    append_event(
                        output,
                        # Experiment status remains resumable; the event type
                        # carries the controlled operational pause.
                        status="running",
                        event_type="strong_staged_pilot_paused",
                        details={"window": name, "stage": stage_name},
                    )
                    print(
                        json.dumps(
                            {
                                "output": str(output),
                                "status": "paused",
                                "window": name,
                                "stage": stage_name,
                            },
                            ensure_ascii=False,
                        )
                    )
                    return

            profile_specs = spec.get("alpha_profiles") or {
                "selected": {
                    "alpha": spec["alpha"],
                    "inference_command": spec["inference_command"],
                }
            }
            multi_profile = "alpha_profiles" in spec
            state.setdefault("alphas", {})
            if state.get("alpha") and "selected" not in state["alphas"]:
                state["alphas"]["selected"] = state["alpha"]
            for signal_profile, profile_spec in profile_specs.items():
                alpha = Path(profile_spec["alpha"])
                existing_alpha = state["alphas"].get(signal_profile)
                artifact_name = (
                    f"{name}:raw_alpha:{signal_profile}"
                    if multi_profile
                    else f"{name}:raw_alpha"
                )
                if existing_alpha:
                    if sha256_file(alpha) != existing_alpha["sha256"]:
                        raise ValueError(f"completed alpha hash drift: {alpha}")
                    _record_once(output, name=artifact_name, path=alpha, kind="prediction_frame_compatible_alpha")
                    continue
                append_event(
                    output,
                    status="running",
                    event_type="strong_window_inference_started",
                    details={"window": name, "signal_profile": signal_profile},
                )
                run_with_memory_guard(
                    profile_spec["inference_command"],
                    cwd=ROOT,
                    min_free_gib=args.min_free_gib,
                )
                alpha_info = _validate_alpha(alpha, window)
                state["alphas"][signal_profile] = alpha_info
                if signal_profile == "selected":
                    state["alpha"] = alpha_info
                _atomic_json(progress_path, progress)
                _record_once(output, name=artifact_name, path=alpha, kind="prediction_frame_compatible_alpha")
                append_event(
                    output,
                    status="running",
                    event_type="strong_window_inference_completed",
                    details={
                        "window": name,
                        "signal_profile": signal_profile,
                        "rows": alpha_info["rows"],
                    },
                )
            append_event(
                output,
                status="running",
                event_type="strong_window_completed",
                details={"window": name, "profiles": list(profile_specs)},
            )

        stitched = _stitch(contract, output)
        _record_once(output, name="staged_pilot_oos_alpha", path=stitched, kind="stitched_dated_alpha")
        for signal_profile in inference_profiles:
            rolling_manifest, rolling_payload = build_standard_rolling_manifest(
                contract,
                progress,
                output,
                signal_profile=signal_profile,
            )
            for split, item in rolling_payload["split_alpha_paths"].items():
                _record_once(
                    output,
                    name=(
                        f"stitched_alpha:{signal_profile}:{split}"
                        if "inference_profiles" in contract
                        else f"stitched_alpha:{split}"
                    ),
                    path=Path(item["path"]),
                    kind="stitched_dated_alpha",
                )
            _record_once(
                output,
                name=(
                    f"rolling_manifest:{signal_profile}"
                    if "inference_profiles" in contract
                    else "rolling_manifest"
                ),
                path=rolling_manifest,
                kind="rolling_manifest",
            )
        _record_once(output, name="staged_pilot_progress", path=progress_path, kind="rolling_progress")
        runtime_metrics = runtime_sampler.finish(
            status="completed",
            details={
                "windows": len(contract["windows"]),
                "training_launched": True,
                "cache_hit": True,
                "resume_enabled": True,
                "transition_checkpoint": args.transition,
            },
        )
        runtime_finished = True
        meta = load_explicit_cross_section_meta(
            cache_meta,
            project_root=ROOT,
            expected_input_dim=expected_dim,
            required_label_families=(
                profile["inferred"]["label_family"],
                profile["inferred"]["lag1_label_family"],
            ),
            logical_end=windows[-1].predict_end,
        )
        provenance_dir = output / "provenance"
        provenance_bundle = write_provenance_bundle(
            provenance_dir,
            environment=environment_manifest(ROOT),
            source=source_manifest(
                ROOT,
                (
                    __file__,
                    "experiments/strong_rolling.py",
                    "experiments/strong_adapter_runtime.py",
                    "experiments/model_adapters.py",
                    "experiments/provenance.py",
                    "run/train.py",
                    "run/generate_v9_inference_alpha.py",
                    "data/dataset_runtime.py",
                    "data/providers.py",
                    "data/pipeline.py",
                    "data/cache_metadata.py",
                    "backtest/engine.py",
                ),
                exclude_paths=(output,),
            ),
            data={
                "schema": "data_manifest_v1",
                "cache_meta": contract["data_cache"],
                "physical_coverage": {
                    "start": meta["physical_data_start"],
                    "end": meta["physical_data_end"],
                },
                "logical_view": {
                    "start": windows[0].train_start,
                    "end": windows[-1].predict_end,
                    "role": "rolling_train_valid_oos",
                },
                "universe": {
                    "codes": len(meta["all_codes"]),
                    "dates": len(meta["all_dates"]),
                    "input_dim": int(meta["x_dim"]),
                    "risk_dim": int(meta["risk_full_dim"]),
                },
                "label_families": [
                    profile["inferred"]["label_family"],
                    profile["inferred"]["lag1_label_family"],
                ],
                "windows": [item["window"] for item in contract["windows"]],
                "pit_status": "frozen_v14_cache_with_declared_logical_view",
            },
            feature_transform={
                "schema": "feature_transform_manifest_v1",
                "cache_transform": build_v14_transform_contract(meta, meta_path=cache_meta),
                "processor_states": [
                    {
                        "window": spec["window"]["name"],
                        "state": progress["windows"][spec["window"]["name"]]
                        .get("dataset", {})
                        .get("processor_state"),
                    }
                    for spec in contract["windows"]
                ],
                "fit_policy": "each rolling window fits processors on Train only; empty chains preserve cache state",
            },
            command=command_manifest(
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                contract,
            ),
            runtime=runtime_metrics,
        )
        validate_provenance_bundle(provenance_bundle)
        for path in sorted(provenance_dir.glob("*.json")):
            _record_once(
                output,
                name=f"provenance:{path.stem}",
                path=path,
                kind="provenance_manifest_v1",
            )
        append_event(output, status="completed", event_type="strong_staged_pilot_completed")
    except BaseException as exc:
        if not runtime_finished:
            runtime_sampler.finish(
                status="failed",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            runtime_finished = True
        append_event(output, status="failed", event_type="strong_staged_pilot_failed", details={"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        finalize_artifact_index(output)
    print(json.dumps({"output": str(output), "windows": list(window_names), "stitched_alpha": str(stitched)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
