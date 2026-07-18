"""Run the first resumable Q5 strong-model Rolling smoke window."""

from __future__ import annotations

import argparse
import json
import subprocess
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
    build_smoke_contract,
    build_smoke_train_command,
    load_json,
    validate_profile,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="configs/reconstructed_multi_downside_e19_profile_v1.json")
    parser.add_argument("--schedule", default="configs/monthly_rolling_compact_4y6m1m_2024_2025.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", default=None, help="Window name; defaults to the first declared OOS month.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--cache-meta",
        default="cache/cross_section_v14_multilabel_open_tech_market_funda_shareh_restr_macro_all_s40_t5_h10_min30_mad_rawlab_end20260518_meta.pkl",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _select_window(schedule, name):
    windows = [RollingWindow.from_mapping(item) for item in schedule["windows"]]
    selected = windows[0] if name is None else next((item for item in windows if item.name == name), None)
    if selected is None:
        raise ValueError(f"unknown rolling window: {name}")
    validate_window(selected, research_end=schedule["data"]["research_end"])
    return selected


def main(argv=None):
    args = parse_args(argv)
    profile_path = (ROOT / args.profile).resolve()
    schedule_path = (ROOT / args.schedule).resolve()
    cache_meta = (ROOT / args.cache_meta).resolve()
    output = (ROOT / args.output_dir).resolve()
    profile = load_json(profile_path)
    schedule = load_json(schedule_path)
    validate_profile(profile)
    window = _select_window(schedule, args.window)
    training_dir = output / "windows" / window.name / "training"
    checkpoint = training_dir / "ultimate_v7_best.pt"
    alpha = output / "windows" / window.name / "alpha_raw.jsonl"
    train_command = build_smoke_train_command(
        python=sys.executable,
        profile=profile,
        window=window,
        output_dir=training_dir,
        cache_meta=cache_meta,
        device=args.device,
    )
    inference_command = build_raw_inference_command(
        python=sys.executable,
        window=window,
        checkpoint=checkpoint,
        output=alpha,
        cache_meta=cache_meta,
        expected_input_dim=profile["confirmed"]["architecture"]["input_dim"],
        device=args.device,
    )
    contract = build_smoke_contract(
        profile_path=profile_path,
        schedule_path=schedule_path,
        window=window,
        train_command=train_command,
        inference_command=inference_command,
        cache_meta=cache_meta,
    )
    if args.dry_run:
        print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)
        return
    if output.exists():
        raise FileExistsError(f"smoke output already exists: {output}")
    output.mkdir(parents=True)
    contract_path = output / "smoke_contract.json"
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
                "effective_start": window.train_start,
                "effective_end": window.predict_end,
                "physical_coverage": "2010-01-04..2026-05-18",
                "role": "rolling_train_valid_oos",
            },
        },
        project_root=ROOT,
    )
    record_artifact(output, name="smoke_contract", path=contract_path, kind="strong_rolling_contract")
    try:
        append_event(output, status="running", event_type="strong_smoke_training_started")
        subprocess.run(train_command, cwd=ROOT, check=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"training completed without checkpoint: {checkpoint}")
        record_artifact(output, name="window_checkpoint", path=checkpoint, kind="torch_checkpoint")
        append_event(output, status="running", event_type="strong_smoke_inference_started")
        subprocess.run(inference_command, cwd=ROOT, check=True)
        if not alpha.is_file():
            raise FileNotFoundError(f"inference completed without alpha: {alpha}")
        record_artifact(output, name="window_raw_alpha", path=alpha, kind="prediction_frame_compatible_alpha")
        append_event(output, status="completed", event_type="strong_smoke_completed")
    except BaseException as exc:
        append_event(
            output,
            status="failed",
            event_type="strong_smoke_failed",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    finally:
        finalize_artifact_index(output)
    print(json.dumps({"output": str(output), "window": window.name, "alpha": str(alpha)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
