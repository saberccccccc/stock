"""Evaluate one dated research alpha through the formal realistic ledger.

This is intentionally an experiment-layer adapter, not a new backtester.  It
records the exact alpha and command under an existing immutable experiment
manifest, then delegates fills and performance calculation to the project's
open-price ledger sweep runner.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ledger_evidence import build_ledger_command, validate_experiment_alpha
from experiments.recording import (
    MANIFEST_NAME,
    append_event,
    finalize_artifact_index,
    record_artifact,
    validate_manifest_for_formal_use,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--alpha-path", required=True)
    parser.add_argument("--split", required=True, choices=("val_2024", "test_2025"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def write_request(path, request):
    if path.exists():
        raise FileExistsError(f"ledger request already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    args = parse_args(argv)
    experiment_dir = Path(args.experiment_dir).resolve()
    manifest_path = experiment_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"experiment manifest is missing: {manifest_path}")
    validate_manifest_for_formal_use(manifest_path)

    alpha_path = Path(args.alpha_path).resolve()
    alpha_info = validate_experiment_alpha(alpha_path, args.split)
    ledger_dir = Path(args.output_dir).resolve() if args.output_dir else experiment_dir / "ledger" / args.split
    command = build_ledger_command(
        alpha_path=alpha_path,
        experiment_id=args.experiment_id,
        split=args.split,
        output_dir=ledger_dir,
        python=args.python,
    )
    if args.resume:
        command.append("--resume")

    request_path = ledger_dir / "ledger_request.json"
    request = {
        "experiment_id": args.experiment_id,
        "split": args.split,
        "alpha": {"path": str(alpha_path), **alpha_info},
        "ledger_dir": str(ledger_dir),
        "command": command,
        "execution_contract": "realistic_open_price_share_ledger",
    }
    write_request(request_path, request)
    record_artifact(experiment_dir, name=f"alpha:{args.split}", path=alpha_path, kind="dated_alpha")
    record_artifact(experiment_dir, name=f"ledger_request:{args.split}", path=request_path, kind="ledger_request")

    if args.dry_run:
        append_event(
            experiment_dir,
            status="running",
            event_type="ledger_dry_run_prepared",
            details={"split": args.split, "ledger_dir": str(ledger_dir)},
        )
        print(json.dumps(request, ensure_ascii=False, indent=2))
        return

    append_event(
        experiment_dir,
        status="running",
        event_type="ledger_started",
        details={"split": args.split, "ledger_dir": str(ledger_dir), "command": command},
    )
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        append_event(
            experiment_dir,
            status="failed",
            event_type="ledger_failed",
            details={"split": args.split, "returncode": int(result.returncode)},
        )
        raise SystemExit(result.returncode)

    summary = ledger_dir / "open_price_ledger_param_sweep_summary.csv"
    if not summary.is_file():
        append_event(
            experiment_dir,
            status="failed",
            event_type="ledger_output_missing",
            details={"split": args.split, "expected_summary": str(summary)},
        )
        raise FileNotFoundError(f"ledger completed without summary: {summary}")
    record_artifact(experiment_dir, name=f"ledger_summary:{args.split}", path=summary, kind="ledger_summary")
    append_event(
        experiment_dir,
        status="completed",
        event_type="ledger_completed",
        details={"split": args.split, "summary": str(summary)},
    )
    finalize_artifact_index(experiment_dir)
    print(json.dumps({"summary": str(summary), "split": args.split}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
