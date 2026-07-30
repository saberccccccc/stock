"""Resume NT6 market-backend evidence in order; never auto-promote."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.nt6_market_backend_closure import (
    closure_paths,
    inspect_nt6_market_backend_closure,
)


DEFAULT_ROOT = (
    "reports/non_training_closure_20260719/"
    "nt6_market_data_pilot_20260730"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incremental-evidence",
        default=f"{DEFAULT_ROOT}/md8_performance_acceptance/incremental_benchmark.json",
    )
    parser.add_argument(
        "--clean-matrix-root",
        default=f"{DEFAULT_ROOT}/md8_performance_acceptance/clean_matrix",
    )
    parser.add_argument(
        "--clean-acceptance",
        default=f"{DEFAULT_ROOT}/md8_performance_acceptance/clean_acceptance.json",
    )
    parser.add_argument(
        "--dual-read-root",
        default=f"{DEFAULT_ROOT}/md9_dual_read_observation",
    )
    parser.add_argument(
        "--policy",
        default="configs/execution_market_backend_policy.json",
    )
    parser.add_argument(
        "--status-output",
        default=f"{DEFAULT_ROOT}/nt6_market_backend_closure_status.json",
    )
    parser.add_argument("--min-free-memory-gib", type=float, default=3.0)
    parser.add_argument("--estimated-peak-memory-gib", type=float, default=0.75)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _inspect(args) -> dict:
    return inspect_nt6_market_backend_closure(
        ROOT,
        **closure_paths(
            incremental_evidence=args.incremental_evidence,
            clean_matrix_root=args.clean_matrix_root,
            clean_acceptance=args.clean_acceptance,
            dual_read_root=args.dual_read_root,
            policy_path=args.policy,
        ),
    )


def _command_for_phase(args, phase: str) -> list[str] | None:
    if phase == "incremental_benchmark":
        return [
            sys.executable,
            "run/benchmark_market_daily_incremental.py",
            "--source-store-root",
            "data/market_daily_candidate_v2",
            "--output",
            str(_resolve(args.incremental_evidence)),
        ]
    if phase == "clean_matrix":
        return [
            sys.executable,
            "run/run_open_ledger_backend_parity_matrix.py",
            "--output-root",
            str(_resolve(args.clean_matrix_root)),
            "--min-free-memory-gib",
            str(args.min_free_memory_gib),
            "--estimated-peak-memory-gib",
            str(args.estimated_peak_memory_gib),
        ]
    if phase == "clean_acceptance":
        return [
            sys.executable,
            "run/audit_market_data_performance.py",
            "--evidence-root",
            str(_resolve(args.clean_matrix_root)),
            "--incremental-evidence",
            str(_resolve(args.incremental_evidence)),
            "--output",
            str(_resolve(args.clean_acceptance)),
        ]
    if phase == "dual_read":
        return [
            sys.executable,
            "run/run_market_data_dual_read_matrix.py",
            "--output-dir",
            str(_resolve(args.dual_read_root)),
            "--min-free-memory-gib",
            str(args.min_free_memory_gib),
            "--estimated-peak-memory-gib",
            str(args.estimated_peak_memory_gib),
        ]
    return None


def main(argv=None):
    args = parse_args(argv)
    status_path = _resolve(args.status_output)
    status = _inspect(args)
    status["updated_at"] = _utc_now()
    status["commands"] = []
    required_gib = args.min_free_memory_gib + args.estimated_peak_memory_gib

    while status["status"] != "ready_for_manual_promotion":
        phase = status["next_phase"]
        command = _command_for_phase(args, phase)
        if command is None:
            status["status"] = "blocked"
            status["blocked_reason"] = "promotion audit is incomplete"
            break
        record = {"phase": phase, "command": command}
        if args.dry_run:
            record["status"] = "dry_run"
            status["commands"].append(record)
            status["status"] = "dry_run"
            break
        if phase in {"clean_matrix", "dual_read"}:
            available_gib = psutil.virtual_memory().available / (1024**3)
            record["available_memory_gib_before"] = round(available_gib, 3)
            record["required_memory_gib_before"] = required_gib
            if available_gib < required_gib:
                record["status"] = "blocked_low_memory"
                status["commands"].append(record)
                status["status"] = "blocked_low_memory"
                status["blocked_reason"] = (
                    f"{available_gib:.3f} GiB available; "
                    f"{required_gib:.3f} GiB required"
                )
                break
        completed = subprocess.run(command, cwd=ROOT)
        record["returncode"] = completed.returncode
        record["status"] = "completed" if completed.returncode == 0 else "failed"
        status["commands"].append(record)
        if completed.returncode != 0:
            status["status"] = (
                "blocked_low_memory" if completed.returncode == 3 else "failed"
            )
            break
        refreshed = _inspect(args)
        if refreshed["next_phase"] == phase:
            status["status"] = "failed"
            status["blocked_reason"] = f"{phase} command did not pass its artifact gate"
            break
        refreshed["commands"] = status["commands"]
        status = refreshed

    status["updated_at"] = _utc_now()
    _atomic_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    if status["status"] == "failed":
        return 1
    if status["status"] in {"blocked", "blocked_low_memory"}:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
