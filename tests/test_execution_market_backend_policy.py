import json

from data.execution_market_backend_policy import (
    audit_promotion,
    load_policy,
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


def test_policy_atomic_write_round_trips(tmp_path):
    path = tmp_path / "policy.json"
    write_policy_atomic(path, _policy())

    assert load_policy(path)["active_backend"] == "legacy"
    assert not path.with_suffix(".json.tmp").exists()
