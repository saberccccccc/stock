"""Run the predeclared three-window Q5 strong-model operational pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.recording import (
    append_event,
    create_experiment,
    finalize_artifact_index,
    record_artifact,
)
from experiments.rolling import RollingWindow, validate_window
from experiments.strong_rolling import (
    build_raw_inference_command,
    build_smoke_train_command,
    canonical_json_hash,
    load_json,
    run_with_memory_guard,
    sha256_file,
    validate_profile,
)


PILOT_WINDOWS = ("oos_2024_01", "oos_2025_01", "oos_2025_12")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="configs/reconstructed_multi_downside_e19_profile_v1.json")
    parser.add_argument("--schedule", default="configs/monthly_rolling_compact_4y6m1m_2024_2025.json")
    parser.add_argument("--cache-meta", default="cache/cross_section_v14_multilabel_open_tech_market_funda_shareh_restr_macro_all_s40_t5_h10_min30_mad_rawlab_end20260518_meta.pkl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--min-free-gib", type=float, default=0.75)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _windows(schedule):
    by_name = {item["name"]: RollingWindow.from_mapping(item) for item in schedule["windows"]}
    selected = [by_name[name] for name in PILOT_WINDOWS]
    for window in selected:
        validate_window(window, research_end=schedule["data"]["research_end"])
    return selected


def _contract(profile_path, schedule_path, cache_meta, windows, output, device):
    profile = load_json(profile_path)
    specs = []
    for window in windows:
        training = output / "windows" / window.name / "training"
        checkpoint = training / "ultimate_v7_best.pt"
        alpha = output / "windows" / window.name / "alpha_raw.jsonl"
        specs.append({
            "window": window.__dict__,
            "checkpoint": str(checkpoint),
            "alpha": str(alpha),
            "train_command": build_smoke_train_command(
                python=sys.executable,
                profile=profile,
                window=window,
                output_dir=training,
                cache_meta=cache_meta,
                epochs=1,
                device=device,
            ),
            "inference_command": build_raw_inference_command(
                python=sys.executable,
                window=window,
                checkpoint=checkpoint,
                output=alpha,
                cache_meta=cache_meta,
                expected_input_dim=profile["confirmed"]["architecture"]["input_dim"],
                device=device,
            ),
        })
    contract = {
        "schema": "strong_rolling_pilot_v1",
        "mode": "three_predeclared_windows_one_epoch",
        "profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
        "schedule": {"path": str(schedule_path), "sha256": sha256_file(schedule_path)},
        "data_cache": {"path": str(cache_meta), "sha256": sha256_file(cache_meta)},
        "windows": specs,
        "raw_signal_only": True,
        "selection_allowed": False,
        "promotion_allowed": False,
    }
    contract["contract_sha256"] = canonical_json_hash(contract)
    return contract


def main(argv=None):
    args = parse_args(argv)
    profile_path = (ROOT / args.profile).resolve()
    schedule_path = (ROOT / args.schedule).resolve()
    cache_meta = (ROOT / args.cache_meta).resolve()
    output = (ROOT / args.output_dir).resolve()
    profile = load_json(profile_path)
    schedule = load_json(schedule_path)
    validate_profile(profile)
    windows = _windows(schedule)
    contract = _contract(profile_path, schedule_path, cache_meta, windows, output, args.device)
    if args.dry_run:
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"pilot output already exists: {output}")
    output.mkdir(parents=True)
    contract_path = output / "pilot_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
                "role": "rolling_train_valid_oos_pilot",
            },
        },
        project_root=ROOT,
    )
    record_artifact(output, name="pilot_contract", path=contract_path, kind="strong_rolling_contract")
    try:
        for spec in contract["windows"]:
            name = spec["window"]["name"]
            checkpoint = Path(spec["checkpoint"])
            alpha = Path(spec["alpha"])
            append_event(output, status="running", event_type="pilot_window_training_started", details={"window": name})
            run_with_memory_guard(spec["train_command"], cwd=ROOT, min_free_gib=args.min_free_gib)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"training completed without checkpoint: {checkpoint}")
            record_artifact(output, name=f"{name}_checkpoint", path=checkpoint, kind="torch_checkpoint")
            append_event(output, status="running", event_type="pilot_window_inference_started", details={"window": name})
            run_with_memory_guard(spec["inference_command"], cwd=ROOT, min_free_gib=args.min_free_gib)
            if not alpha.is_file():
                raise FileNotFoundError(f"inference completed without alpha: {alpha}")
            record_artifact(output, name=f"{name}_raw_alpha", path=alpha, kind="prediction_frame_compatible_alpha")
            append_event(output, status="running", event_type="pilot_window_completed", details={"window": name})
        append_event(output, status="completed", event_type="strong_pilot_completed")
    except BaseException as exc:
        append_event(output, status="failed", event_type="strong_pilot_failed", details={"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        finalize_artifact_index(output)
    print(json.dumps({"output": str(output), "windows": list(PILOT_WINDOWS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
