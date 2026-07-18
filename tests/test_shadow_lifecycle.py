import json

import pytest

from experiments.recording import sha256_file
from experiments.shadow_lifecycle import (
    create_shadow_lifecycle,
    record_shadow_observation,
    transition_shadow_lifecycle,
    validate_shadow_lifecycle,
)


def _workflow(tmp_path):
    workflow = tmp_path / "workflow"
    records = workflow / "records"
    records.mkdir(parents=True)
    record_entries = {}
    for name in ("signal", "signal_analysis", "portfolio", "risk_attribution", "stress", "decision"):
        manifest = records / name / "record_manifest.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({"schema": "standard_record_v1", "name": name}) + "\n", encoding="utf-8")
        record_entries[name] = {"manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest)}
    (records / "bundle_manifest.json").write_text(
        json.dumps({"schema": "standard_record_bundle_v1", "records": record_entries}) + "\n",
        encoding="utf-8",
    )
    (workflow / "workflow_config.json").write_text('{"schema_version":2}\n', encoding="utf-8")
    (workflow / "compiled_workflow.json").write_text('{"stages":[]}\n', encoding="utf-8")
    (workflow / "experiment_manifest.json").write_text('{"formal":true}\n', encoding="utf-8")
    return workflow


def _create(tmp_path):
    lifecycle = tmp_path / "lifecycle"
    create_shadow_lifecycle(
        lifecycle,
        lifecycle_id="shadow-candidate-v1",
        candidate_id="candidate",
        workflow_dir=_workflow(tmp_path),
        actor="researcher",
        reason="manual framework acceptance",
    )
    return lifecycle


def test_shadow_lifecycle_freezes_record_bundle_and_disables_automation(tmp_path):
    lifecycle = _create(tmp_path)

    snapshot = validate_shadow_lifecycle(lifecycle)

    assert snapshot["state"]["state"] == "prepared"
    assert snapshot["manifest"]["artifacts"]["record_bundle"]["sha256"]
    assert snapshot["manifest"]["controls"] == {
        "manual_approval_required": True,
        "automatic_trading": False,
        "automatic_retraining": False,
        "automatic_promotion": False,
        "forward_selection_allowed": False,
    }


def test_shadow_transition_requires_manual_approval_and_valid_state_edge(tmp_path):
    lifecycle = _create(tmp_path)

    with pytest.raises(ValueError, match="manual approval"):
        transition_shadow_lifecycle(
            lifecycle, target_state="shadow", actor="researcher", reason="start", manual_approval=False
        )
    transition_shadow_lifecycle(
        lifecycle, target_state="shadow", actor="researcher", reason="start", manual_approval=True
    )
    transition_shadow_lifecycle(
        lifecycle, target_state="paused", actor="researcher", reason="inspect", manual_approval=True
    )
    transition_shadow_lifecycle(
        lifecycle, target_state="retired", actor="researcher", reason="closed", manual_approval=True
    )

    with pytest.raises(ValueError, match="retired -> shadow"):
        transition_shadow_lifecycle(
            lifecycle, target_state="shadow", actor="researcher", reason="restart", manual_approval=True
        )


def test_shadow_observation_is_dated_unique_and_hash_checked(tmp_path):
    lifecycle = _create(tmp_path)
    observation = tmp_path / "daily_scorecard.csv"
    observation.write_text("date,return\n2026-01-05,0.01\n", encoding="utf-8")

    with pytest.raises(ValueError, match="shadow state"):
        record_shadow_observation(
            lifecycle,
            observation_date="2026-01-05",
            artifacts={"scorecard": observation},
            actor="researcher",
            reason="daily observation",
        )
    transition_shadow_lifecycle(
        lifecycle, target_state="shadow", actor="researcher", reason="start", manual_approval=True
    )
    record_shadow_observation(
        lifecycle,
        observation_date="2026-01-05",
        artifacts={"scorecard": observation},
        actor="researcher",
        reason="daily observation",
    )
    with pytest.raises(ValueError, match="duplicate"):
        record_shadow_observation(
            lifecycle,
            observation_date="2026-01-05",
            artifacts={"scorecard": observation},
            actor="researcher",
            reason="duplicate",
        )
    observation.write_text("date,return\n2026-01-05,0.02\n", encoding="utf-8")
    with pytest.raises(ValueError, match="observation artifact changed"):
        validate_shadow_lifecycle(lifecycle)


def test_shadow_event_hash_chain_rejects_tampering(tmp_path):
    lifecycle = _create(tmp_path)
    events_path = lifecycle / "lifecycle_events.jsonl"
    event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    event["reason"] = "tampered"
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="event hash"):
        validate_shadow_lifecycle(lifecycle)


def test_shadow_creation_rejects_broken_record_bundle(tmp_path):
    workflow = _workflow(tmp_path)
    (workflow / "records" / "signal" / "record_manifest.json").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Record manifest is invalid"):
        create_shadow_lifecycle(
            tmp_path / "lifecycle",
            lifecycle_id="shadow-candidate-v1",
            candidate_id="candidate",
            workflow_dir=workflow,
            actor="researcher",
            reason="manual framework acceptance",
        )
