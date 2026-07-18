"""Manual, append-only Shadow lifecycle inspired by Qlib OnlineManager."""

from __future__ import annotations

import json
from datetime import date as date_type
from pathlib import Path
from typing import Any, Mapping

from experiments.recording import canonical_json_hash, sha256_file, utc_now


MANIFEST_NAME = "lifecycle_manifest.json"
EVENTS_NAME = "lifecycle_events.jsonl"
STATE_NAME = "lifecycle_state.json"
ALLOWED_TRANSITIONS = {
    "prepared": {"shadow", "retired"},
    "shadow": {"paused", "retired"},
    "paused": {"shadow", "retired"},
    "retired": set(),
}


def _load_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_state(path: Path, value: Mapping[str, Any]):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_events(path: Path):
    if not path.is_file():
        return []
    events = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def validate_event_chain(events):
    previous = None
    for index, event in enumerate(events):
        if event.get("sequence") != index + 1:
            raise ValueError("Shadow lifecycle event sequence is not contiguous")
        if event.get("previous_event_sha256") != previous:
            raise ValueError("Shadow lifecycle event hash chain is broken")
        payload = dict(event)
        expected = payload.pop("event_sha256", None)
        if not expected or canonical_json_hash(payload) != expected:
            raise ValueError("Shadow lifecycle event hash is invalid")
        previous = expected
    return previous


def _append_event(lifecycle_dir: Path, event_type: str, state: str, actor: str, reason: str, details=None):
    events_path = lifecycle_dir / EVENTS_NAME
    events = _read_events(events_path)
    previous = validate_event_chain(events)
    event = {
        "schema_version": 1,
        "sequence": len(events) + 1,
        "timestamp": utc_now(),
        "event_type": str(event_type),
        "state": str(state),
        "actor": str(actor).strip(),
        "reason": str(reason).strip(),
        "details": dict(details or {}),
        "previous_event_sha256": previous,
    }
    if not event["actor"] or not event["reason"]:
        raise ValueError("manual Shadow events require actor and reason")
    event["event_sha256"] = canonical_json_hash(event)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return event


def _artifact(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}


def validate_record_bundle(bundle_path: str | Path):
    bundle_path = Path(bundle_path).resolve()
    bundle = _load_json(bundle_path)
    if bundle.get("schema") != "standard_record_bundle_v1":
        raise ValueError("Shadow lifecycle requires standard_record_bundle_v1")
    records = bundle.get("records", {})
    expected = {"signal", "signal_analysis", "portfolio", "risk_attribution", "stress", "decision"}
    if set(records) != expected:
        raise ValueError("Shadow lifecycle Record bundle is incomplete")
    for name, descriptor in records.items():
        manifest_path = Path(str(descriptor.get("manifest_path", ""))).resolve()
        expected_hash = str(descriptor.get("manifest_sha256", ""))
        if not manifest_path.is_file() or not expected_hash or sha256_file(manifest_path) != expected_hash:
            raise ValueError(f"Shadow lifecycle Record manifest is invalid: {name}")
    return bundle


def create_shadow_lifecycle(
    lifecycle_dir: str | Path,
    *,
    lifecycle_id: str,
    candidate_id: str,
    workflow_dir: str | Path,
    actor: str,
    reason: str,
):
    target = Path(lifecycle_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Shadow lifecycle directory is not empty: {target}")
    workflow = Path(workflow_dir).resolve()
    bundle = workflow / "records" / "bundle_manifest.json"
    workflow_config = workflow / "workflow_config.json"
    compiled = workflow / "compiled_workflow.json"
    source_manifest = workflow / "experiment_manifest.json"
    validate_record_bundle(bundle)
    if not str(lifecycle_id).strip() or not str(candidate_id).strip():
        raise ValueError("Shadow lifecycle_id and candidate_id are required")
    artifacts = {
        "record_bundle": _artifact(bundle),
        "workflow_config": _artifact(workflow_config),
        "compiled_workflow": _artifact(compiled),
    }
    if source_manifest.is_file():
        artifacts["workflow_manifest"] = _artifact(source_manifest)
    manifest = {
        "schema_version": 1,
        "lifecycle_id": str(lifecycle_id),
        "candidate_id": str(candidate_id),
        "workflow_dir": str(workflow),
        "created_at": utc_now(),
        "initial_state": "prepared",
        "allowed_states": list(ALLOWED_TRANSITIONS),
        "controls": {
            "manual_approval_required": True,
            "automatic_trading": False,
            "automatic_retraining": False,
            "automatic_promotion": False,
            "forward_selection_allowed": False,
        },
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    _write_exclusive(target / MANIFEST_NAME, manifest)
    event = _append_event(target, "lifecycle_created", "prepared", actor, reason)
    _write_state(
        target / STATE_NAME,
        {
            "schema_version": 1,
            "lifecycle_id": str(lifecycle_id),
            "state": "prepared",
            "last_event_sequence": event["sequence"],
            "last_event_sha256": event["event_sha256"],
            "updated_at": event["timestamp"],
        },
    )
    return target / MANIFEST_NAME


def validate_shadow_lifecycle(lifecycle_dir: str | Path):
    target = Path(lifecycle_dir).resolve()
    manifest = _load_json(target / MANIFEST_NAME)
    expected_manifest_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if canonical_json_hash(unhashed) != expected_manifest_hash:
        raise ValueError("Shadow lifecycle manifest hash is invalid")
    for descriptor in manifest["artifacts"].values():
        path = Path(descriptor["path"])
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"Shadow lifecycle frozen artifact changed: {path}")
    events = _read_events(target / EVENTS_NAME)
    last_hash = validate_event_chain(events)
    for event in events:
        for descriptor in event.get("details", {}).get("artifacts", {}).values():
            path = Path(descriptor["path"])
            if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
                raise ValueError(f"Shadow observation artifact changed: {path}")
    state = _load_json(target / STATE_NAME)
    if not events or state["last_event_sequence"] != events[-1]["sequence"] or state["last_event_sha256"] != last_hash:
        raise ValueError("Shadow lifecycle state projection is stale")
    if state["state"] != events[-1]["state"]:
        raise ValueError("Shadow lifecycle state disagrees with event log")
    return {"manifest": manifest, "state": state, "events": events}


def transition_shadow_lifecycle(
    lifecycle_dir: str | Path,
    *,
    target_state: str,
    actor: str,
    reason: str,
    manual_approval: bool,
):
    if not manual_approval:
        raise ValueError("Shadow transition requires explicit manual approval")
    target = Path(lifecycle_dir).resolve()
    snapshot = validate_shadow_lifecycle(target)
    current = str(snapshot["state"]["state"])
    target_state = str(target_state)
    if target_state not in ALLOWED_TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid Shadow lifecycle transition: {current} -> {target_state}")
    event = _append_event(
        target,
        "state_transition",
        target_state,
        actor,
        reason,
        details={"from_state": current, "to_state": target_state, "manual_approval": True},
    )
    _write_state(
        target / STATE_NAME,
        {
            "schema_version": 1,
            "lifecycle_id": snapshot["manifest"]["lifecycle_id"],
            "state": target_state,
            "last_event_sequence": event["sequence"],
            "last_event_sha256": event["event_sha256"],
            "updated_at": event["timestamp"],
        },
    )
    return target / STATE_NAME


def record_shadow_observation(
    lifecycle_dir: str | Path,
    *,
    observation_date: str,
    artifacts: Mapping[str, str | Path],
    actor: str,
    reason: str,
):
    target = Path(lifecycle_dir).resolve()
    snapshot = validate_shadow_lifecycle(target)
    if snapshot["state"]["state"] != "shadow":
        raise ValueError("Shadow observations can only be recorded in shadow state")
    date = str(observation_date)
    try:
        date = date_type.fromisoformat(date).isoformat()
    except ValueError as exc:
        raise ValueError("Shadow observation date must be ISO YYYY-MM-DD") from exc
    if any(
        event.get("event_type") == "shadow_observation"
        and event.get("details", {}).get("observation_date") == date
        for event in snapshot["events"]
    ):
        raise ValueError(f"duplicate Shadow observation date: {date}")
    descriptors = {str(name): _artifact(Path(path).resolve()) for name, path in artifacts.items()}
    event = _append_event(
        target,
        "shadow_observation",
        "shadow",
        actor,
        reason,
        details={
            "observation_date": date,
            "artifacts": descriptors,
            "selection_eligible": False,
            "forward_used_for_selection": False,
        },
    )
    _write_state(
        target / STATE_NAME,
        {
            **snapshot["state"],
            "last_event_sequence": event["sequence"],
            "last_event_sha256": event["event_sha256"],
            "updated_at": event["timestamp"],
        },
    )
    return target / STATE_NAME
