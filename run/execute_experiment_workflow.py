"""Execute a frozen workflow stage graph with hash-checked resumable receipts."""

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
    MANIFEST_NAME,
    append_event,
    canonical_json_hash,
    finalize_artifact_index,
    record_artifact,
    utc_now,
    validate_manifest_for_formal_use,
)


ALLOWED_STAGE_SCRIPTS = {
    "rolling_lgbm_alpha": "run/rolling_lgbm_alpha.py",
    "torch_strong_alpha": "run/rolling_strong_staged_pilot.py",
    "frozen_dated_predictions": "run/materialize_frozen_predictions.py",
    "workflow_candidate_registry": "run/materialize_workflow_candidates.py",
    "official_open_ledger": "run/official_backtest_from_registry.py",
    "registry_scorecard": "run/scorecard_from_registry.py",
    "workflow_standard_records": "run/materialize_workflow_records.py",
}
NOOP_ADAPTERS = set()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-workflow", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _load(path: Path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _validate_command(stage):
    adapter = str(stage.get("adapter", ""))
    command = stage.get("command")
    if adapter in NOOP_ADAPTERS:
        if command is not None:
            raise ValueError(f"no-op adapter {adapter} must not contain a command")
        return
    expected_script = ALLOWED_STAGE_SCRIPTS.get(adapter)
    if expected_script is None:
        raise ValueError(f"workflow execution rejects unknown adapter {adapter!r}")
    if not isinstance(command, list) or len(command) < 2 or command[1] != expected_script:
        raise ValueError(f"adapter {adapter} must execute {expected_script}")


def _write_receipt(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _failed_receipt_path(receipts_dir: Path, stage_name: str, ended_at: str) -> Path:
    timestamp = ended_at.replace(":", "").replace("-", "").replace("+", "_")
    return receipts_dir / f"{stage_name}.failed.{timestamp}.json"


def execute_workflow(compiled_path: str | Path, *, project_root: str | Path, resume: bool = False):
    compiled_path = Path(compiled_path).resolve()
    compiled = _load(compiled_path)
    output_dir = Path(compiled["output_dir"]).resolve()
    manifest_path = output_dir / MANIFEST_NAME
    validate_manifest_for_formal_use(
        manifest_path,
        require_completed=not resume,
        require_current_index=not resume,
    )
    if canonical_json_hash(_load(output_dir / "workflow_config.json")) != compiled["config_sha256"]:
        raise ValueError("frozen workflow config hash does not match compiled workflow")
    if Path(compiled["project_root"]).resolve() != Path(project_root).resolve():
        raise ValueError("compiled workflow project_root does not match executor root")

    receipts_dir = output_dir / "stage_receipts"
    completed = set()
    append_event(output_dir, status="running", event_type="workflow_execution_started")
    try:
        for stage in compiled["stages"]:
            name = str(stage["name"])
            _validate_command(stage)
            missing = set(stage.get("depends_on", [])) - completed
            if missing:
                raise ValueError(f"stage {name} has incomplete dependencies: {sorted(missing)}")
            receipt_path = receipts_dir / f"{name}.json"
            command_hash = canonical_json_hash({"adapter": stage["adapter"], "command": stage.get("command")})
            if receipt_path.exists():
                receipt = _load(receipt_path)
                if not resume:
                    raise FileExistsError(f"stage receipt already exists: {receipt_path}")
                if receipt.get("status") != "completed" or receipt.get("command_sha256") != command_hash:
                    raise ValueError(f"stage receipt cannot be resumed safely: {receipt_path}")
                completed.add(name)
                continue

            started_at = utc_now()
            command = stage.get("command")
            if command is None:
                returncode = 0
            else:
                execution_command = list(command)
                if resume and stage["adapter"] in {"rolling_lgbm_alpha", "torch_strong_alpha"} and "--resume" not in execution_command:
                    execution_command.append("--resume")
                result = subprocess.run(execution_command, cwd=project_root)
                returncode = int(result.returncode)
            ended_at = utc_now()
            receipt = {
                "schema_version": 1,
                "stage": name,
                "adapter": stage["adapter"],
                "command": command,
                "command_sha256": command_hash,
                "started_at": started_at,
                "ended_at": ended_at,
                "returncode": returncode,
                "status": "completed" if returncode == 0 else "failed",
            }
            written_receipt_path = (
                receipt_path
                if returncode == 0
                else _failed_receipt_path(receipts_dir, name, ended_at)
            )
            _write_receipt(written_receipt_path, receipt)
            artifact_name = (
                f"stage_receipt:{name}"
                if returncode == 0
                else f"stage_failed_receipt:{name}:{ended_at}"
            )
            record_artifact(
                output_dir,
                name=artifact_name,
                path=written_receipt_path,
                kind="workflow_stage_receipt",
            )
            if returncode != 0:
                raise RuntimeError(f"workflow stage {name} failed with returncode={returncode}")
            completed.add(name)
        append_event(output_dir, status="completed", event_type="workflow_execution_completed")
        finalize_artifact_index(output_dir)
        return {"status": "completed", "completed_stages": sorted(completed)}
    except Exception as exc:
        append_event(
            output_dir,
            status="failed",
            event_type="workflow_execution_failed",
            details={"error": f"{type(exc).__name__}: {exc}"},
        )
        finalize_artifact_index(output_dir)
        raise


def main(argv=None):
    args = parse_args(argv)
    result = execute_workflow(args.compiled_workflow, project_root=ROOT, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
