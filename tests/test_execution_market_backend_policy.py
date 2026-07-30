import json
import hashlib

import pytest

import backtest.market_data_contract as market_data_contract
from data.execution_market_backend_policy import (
    audit_promotion,
    load_policy,
    promote_policy,
    rollback_policy,
    write_policy_atomic,
)


def _policy():
    return {
        "schema": "execution_market_backend_policy_v1",
        "state": "legacy_active",
        "active_backend": "legacy",
        "candidate_backend": "monthly",
        "rollback_backend": "legacy",
        "shadow_backend": "csv",
        "required_dual_read_splits": [
            "val_2024",
            "test_2025",
            "forward_2026",
        ],
        "evidence": {
            "behavior_parity": "parity.json",
            "performance_acceptance": "performance.json",
            "call_site_audit": "calls.json",
            "dual_read_reports": {
                "val_2024": "dual.val.json",
                "test_2025": "dual.test.json",
                "forward_2026": "dual.forward.json",
            },
        },
    }


SCHEMAS = {
    "parity.json": "open_ledger_backend_parity_matrix_v1",
    "performance.json": "market_data_performance_acceptance_v1",
    "calls.json": "market_data_call_site_audit_v1",
    "dual.val.json": "execution_market_dual_read_v1",
    "dual.test.json": "execution_market_dual_read_v1",
    "dual.forward.json": "execution_market_dual_read_v1",
}


def _write_status(root, relative, status):
    path = root / relative
    value = {"status": status, "schema": SCHEMAS[relative]}
    if relative == "performance.json":
        evidence_root = root / "performance_sources"
        source_paths = {
            "matrix_status": evidence_root / "matrix_status.json",
            "csv_test_2025_performance": (
                evidence_root / "csv" / "test_2025" / "performance.json"
            ),
            "monthly_test_2025_performance": (
                evidence_root / "monthly" / "test_2025" / "performance.json"
            ),
            "csv_forward_2026_performance": (
                evidence_root / "csv" / "forward_2026" / "performance.json"
            ),
            "monthly_forward_2026_performance": (
                evidence_root / "monthly" / "forward_2026" / "performance.json"
            ),
            "incremental_benchmark": root / "incremental.json",
        }
        for key, source_path in source_paths.items():
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(key, encoding="utf-8")
        value["gates"] = {
            key: True
            for key in (
                "parity_passed",
                "runtime_passed",
                "memory_passed",
                "process_io_recorded",
                "incremental_passed",
            )
        }
        value["evidence_sha256"] = {
            key: hashlib.sha256(source_path.read_bytes()).hexdigest()
            for key, source_path in source_paths.items()
        }
        value["evidence_root"] = str(evidence_root)
        value["incremental"] = {"path": str(root / "incremental.json")}
    if relative.startswith("dual."):
        value.update({"primary_backend": "monthly", "shadow_backend": "csv"})
    path.write_text(json.dumps(value), encoding="utf-8")


def test_promotion_requires_every_evidence_gate(tmp_path):
    policy = _policy()
    for relative in (
        "parity.json",
        "performance.json",
        "calls.json",
        "dual.val.json",
        "dual.test.json",
        "dual.forward.json",
    ):
        _write_status(tmp_path, relative, "passed")

    audit = audit_promotion(tmp_path, policy)

    assert audit["status"] == "passed"


def test_promotion_blocks_missing_forward_dual_read(tmp_path):
    policy = _policy()
    for relative in (
        "parity.json",
        "performance.json",
        "calls.json",
        "dual.val.json",
        "dual.test.json",
    ):
        _write_status(tmp_path, relative, "passed")

    audit = audit_promotion(tmp_path, policy)

    assert audit["status"] == "blocked"
    assert audit["checks"]["dual_read"]["forward_2026"]["status"] == "missing"


def test_promotion_rejects_performance_without_frozen_source_hashes(tmp_path):
    policy = _policy()
    for relative in SCHEMAS:
        _write_status(tmp_path, relative, "passed")
    performance = tmp_path / "performance.json"
    value = json.loads(performance.read_text())
    del value["evidence_sha256"]["incremental_benchmark"]
    performance.write_text(json.dumps(value), encoding="utf-8")

    audit = audit_promotion(tmp_path, policy)

    assert audit["status"] == "blocked"
    assert audit["checks"]["performance_acceptance"][
        "required_hashes_present"
    ] is False


def test_promotion_rejects_tampered_performance_source(tmp_path):
    policy = _policy()
    for relative in SCHEMAS:
        _write_status(tmp_path, relative, "passed")
    source = (
        tmp_path
        / "performance_sources"
        / "monthly"
        / "test_2025"
        / "performance.json"
    )
    source.write_text("tampered", encoding="utf-8")

    audit = audit_promotion(tmp_path, policy)

    assert audit["status"] == "blocked"
    assert audit["checks"]["performance_acceptance"][
        "source_hashes_match"
    ] is False


def test_policy_atomic_write_round_trips(tmp_path):
    path = tmp_path / "policy.json"
    write_policy_atomic(path, _policy())

    assert load_policy(path)["active_backend"] == "legacy"
    assert not path.with_suffix(".json.tmp").exists()


def test_promote_then_rollback_preserves_append_only_transition_history(
    tmp_path,
    monkeypatch,
):
    policy = _policy()
    for relative in SCHEMAS:
        _write_status(tmp_path, relative, "passed")
    audit = audit_promotion(tmp_path, policy)

    promoted = promote_policy(
        policy,
        audit,
        actor="tester",
        reason="complete evidence",
        changed_at="2026-07-31T01:00:00+00:00",
    )
    policy_path = tmp_path / "policy.json"
    write_policy_atomic(policy_path, promoted)
    monkeypatch.setattr(
        market_data_contract,
        "BACKEND_POLICY_PATH",
        policy_path,
    )

    assert policy["active_backend"] == "legacy"
    assert promoted["active_backend"] == "monthly"
    assert market_data_contract.configured_default_backend() == "monthly"
    assert promoted["last_transition"]["from_backend"] == "legacy"
    assert promoted["last_transition"]["to_backend"] == "monthly"
    assert set(promoted["last_transition"]["evidence_sha256"]) == {
        "behavior_parity",
        "performance_acceptance",
        "call_site_audit",
    }

    rolled_back = rollback_policy(
        promoted,
        actor="tester",
        reason="drill",
        changed_at="2026-07-31T01:05:00+00:00",
    )
    write_policy_atomic(policy_path, rolled_back)

    assert market_data_contract.configured_default_backend() == "legacy"
    assert [item["action"] for item in rolled_back["transition_history"]] == [
        "promote",
        "rollback",
    ]
    assert rolled_back["transition_history"][0] == promoted["last_transition"]
    assert rolled_back["last_transition"]["from_backend"] == "monthly"
    assert rolled_back["last_transition"]["to_backend"] == "legacy"


def test_rollback_rejects_fake_transition_while_legacy_is_active():
    with pytest.raises(
        ValueError,
        match="monthly backend to be active",
    ):
        rollback_policy(
            _policy(),
            actor="tester",
            reason="invalid drill",
            changed_at="2026-07-31T01:00:00+00:00",
        )


def test_policy_rejects_broken_transition_history(tmp_path):
    policy = _policy()
    policy["transition_history"] = [
        {
            "action": "rollback",
            "actor": "tester",
            "reason": "broken",
            "changed_at": "2026-07-31T01:00:00+00:00",
            "from_backend": "legacy",
            "to_backend": "monthly",
        }
    ]
    policy["last_transition"] = policy["transition_history"][0]
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid backend path"):
        load_policy(path)
