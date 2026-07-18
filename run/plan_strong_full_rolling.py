"""Build the pre-launch time, disk, and contract estimate for full strong Rolling."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.recording import canonical_json_hash, sha256_file
from experiments.strong_rolling import finalize_full_rolling_contract, load_json
from run.rolling_strong_staged_pilot import build_contract, _select_windows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pilot",
        default="reports/experiments/strong_e19_staged_pilot_20260717",
    )
    parser.add_argument("--profile", default="configs/reconstructed_multi_downside_e19_profile_v1.json")
    parser.add_argument("--schedule", default="configs/monthly_rolling_compact_4y6m1m_2024_2025.json")
    parser.add_argument("--cache-meta", default="cache/cross_section_v14_multilabel_open_tech_market_funda_shareh_restr_macro_all_s40_t5_h10_min30_mad_rawlab_end20260518_meta.pkl")
    parser.add_argument("--planned-output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--transition", choices=("exact", "selected"), default="exact")
    parser.add_argument(
        "--inference-profiles",
        default="exact,selected",
        help="Final checkpoint profiles materialized from each trained window.",
    )
    parser.add_argument("--disk-reserve-gib", type=float, default=30.0)
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _pilot_window_seconds(events):
    starts, completed = {}, {}
    for event in events:
        details = event.get("details", {})
        window = details.get("window")
        if event.get("event_type") == "strong_stage_started" and details.get("stage") == "base_e6":
            starts[window] = _timestamp(event["at"])
        elif event.get("event_type") == "strong_window_completed":
            completed[window] = _timestamp(event["at"])
    values = {
        name: (completed[name] - start).total_seconds()
        for name, start in starts.items()
        if name in completed
    }
    if len(values) < 3:
        raise ValueError("resource planning requires three completed pilot windows")
    return values


def main(argv=None):
    args = parse_args(argv)
    source = _resolve(args.source_pilot)
    profile_path = _resolve(args.profile)
    schedule_path = _resolve(args.schedule)
    cache_meta = _resolve(args.cache_meta)
    planned_output = _resolve(args.planned_output_dir)
    output = _resolve(args.output)
    schedule = load_json(schedule_path)
    names = [item["name"] for item in schedule["windows"]]
    windows = _select_windows(schedule, names)
    profile = load_json(profile_path)
    inference_profiles = tuple(
        value.strip() for value in args.inference_profiles.split(",") if value.strip()
    )
    contract = build_contract(
        profile_path,
        schedule_path,
        cache_meta,
        windows,
        planned_output,
        args.device,
        args.transition,
        inference_profiles,
    )
    contract = finalize_full_rolling_contract(contract)

    events = [
        json.loads(line)
        for line in (source / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seconds = _pilot_window_seconds(events)
    pilot_bytes = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    pilot_windows = len(seconds)
    estimated_seconds = float(np.mean(list(seconds.values())) * len(windows))
    extra_profile_count = max(0, len(inference_profiles) - 1)
    pilot_alpha_bytes = sum(
        path.stat().st_size for path in source.glob("windows/*/alpha_raw.jsonl")
    )
    estimated_profile_overhead = int(
        pilot_alpha_bytes / pilot_windows * len(windows) * extra_profile_count
    )
    estimated_bytes = int(pilot_bytes / pilot_windows * len(windows)) + estimated_profile_overhead
    free_bytes = int(shutil.disk_usage(planned_output.anchor).free)
    reserve_bytes = int(args.disk_reserve_gib * 1024**3)
    disk_pass = free_bytes - estimated_bytes >= reserve_bytes
    result = {
        "schema": "strong_full_rolling_launch_plan_v1",
        "profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
        "schedule": {"path": str(schedule_path), "sha256": sha256_file(schedule_path)},
        "cache_meta": {"path": str(cache_meta), "sha256": sha256_file(cache_meta)},
        "planned_output_dir": str(planned_output),
        "transition_checkpoint": args.transition,
        "inference_profiles": list(inference_profiles),
        "windows": len(windows),
        "epochs_per_window": 19,
        "total_target_epochs": len(windows) * 19,
        "pilot_window_seconds": seconds,
        "estimated_wall_seconds": estimated_seconds,
        "estimated_wall_hours": estimated_seconds / 3600.0,
        "pilot_artifact_bytes": pilot_bytes,
        "estimated_artifact_bytes": estimated_bytes,
        "estimated_extra_profile_bytes": estimated_profile_overhead,
        "estimated_artifact_gib": estimated_bytes / 1024**3,
        "disk_free_bytes": free_bytes,
        "disk_free_gib": free_bytes / 1024**3,
        "required_post_run_reserve_gib": args.disk_reserve_gib,
        "disk_gate_pass": disk_pass,
        "resume": "per stage exact epoch checkpoint plus immutable progress hashes",
        "memory_guard_min_free_gib": 0.75,
        "selection_allowed_during_signal_generation": False,
        "forward_used": False,
        "full_contract": contract,
    }
    result["launch_plan_sha256"] = canonical_json_hash(result)
    if not disk_pass:
        raise RuntimeError("full strong Rolling fails the predeclared disk reserve gate")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "windows": len(windows),
                "estimated_wall_hours": round(estimated_seconds / 3600.0, 2),
                "estimated_artifact_gib": round(estimated_bytes / 1024**3, 2),
                "disk_free_gib": round(free_bytes / 1024**3, 2),
                "disk_gate_pass": disk_pass,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
