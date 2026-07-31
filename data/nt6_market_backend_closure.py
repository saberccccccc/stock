"""Read-only NT6 market-backend closure state inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from data.execution_market_backend_policy import audit_promotion, load_policy


def _recovery_drill(
    policy: Mapping[str, Any],
    promotion_audit: Mapping[str, Any],
) -> dict[str, Any]:
    history = list(policy.get("transition_history", []))
    expected = (
        ("promote", "legacy", "monthly"),
        ("rollback", "monthly", "legacy"),
        ("promote", "legacy", "monthly"),
    )
    identity_fields = (
        "candidate_store_root",
        "candidate_monthly_cache_root",
        "candidate_manifest_sha256",
        "evidence_sha256",
        "dual_read_sha256",
    )
    windows = [
        history[index : index + 3]
        for index in range(max(len(history) - 2, 0))
    ]
    sequence_windows = [
        transitions
        for transitions in windows
        if all(
            (
                transition.get("action"),
                transition.get("from_backend"),
                transition.get("to_backend"),
            )
            == required
            for transition, required in zip(transitions, expected)
        )
    ]
    identity_windows = [
        transitions
        for transitions in sequence_windows
        if all(
            transitions[0].get(field) == transitions[2].get(field)
            for field in identity_fields
        )
    ]
    sequence_matches = bool(sequence_windows)
    promotion_identity_matches = bool(identity_windows)
    final_transition = history[-1] if history else {}
    expected_evidence = {
        name: item["sha256"]
        for name, item in promotion_audit.get("checks", {}).items()
        if name not in {"dual_read", "candidate_identity"} and "sha256" in item
    }
    expected_dual = {
        split: item["sha256"]
        for split, item in promotion_audit.get("checks", {})
        .get("dual_read", {})
        .items()
        if "sha256" in item
    }
    candidate_identity = promotion_audit.get("checks", {}).get(
        "candidate_identity", {}
    )
    final_evidence_matches = (
        final_transition.get("action") == "promote"
        and final_transition.get("candidate_store_root")
        == policy.get("candidate_store_root")
        and final_transition.get("candidate_monthly_cache_root")
        == policy.get("candidate_monthly_cache_root")
        and final_transition.get("candidate_manifest_sha256")
        == candidate_identity.get("active_manifest_sha256")
        and final_transition.get("evidence_sha256") == expected_evidence
        and final_transition.get("dual_read_sha256") == expected_dual
    )
    final_state_matches = (
        policy.get("state") == "monthly_active"
        and policy.get("active_backend") == "monthly"
        and bool(history)
        and policy.get("last_transition") == final_transition
    )
    passed = (
        promotion_audit.get("status") == "passed"
        and sequence_matches
        and promotion_identity_matches
        and final_evidence_matches
        and final_state_matches
    )
    return {
        "status": "passed" if passed else "incomplete",
        "passed": passed,
        "sequence_matches": sequence_matches,
        "promotion_identity_matches": promotion_identity_matches,
        "final_evidence_matches": final_evidence_matches,
        "final_state_matches": final_state_matches,
        "transition_count": len(history),
    }


def _artifact(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "schema": None,
            "status": "missing",
            "passed": False,
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return {
            "path": str(path),
            "schema": None,
            "status": "invalid",
            "error": str(exc),
            "passed": False,
        }
    return {
        "path": str(path),
        "schema": value.get("schema"),
        "status": value.get("status"),
        "passed": value.get("schema") == schema and value.get("status") == "passed",
    }


def inspect_nt6_market_backend_closure(
    project_root: str | Path,
    *,
    incremental_evidence: str | Path,
    clean_matrix_status: str | Path,
    clean_acceptance: str | Path,
    dual_read_matrix_status: str | Path,
    policy_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()

    def resolve(path: str | Path) -> Path:
        value = Path(path)
        return value.resolve() if value.is_absolute() else (root / value).resolve()

    artifacts = {
        "incremental_evidence": _artifact(
            resolve(incremental_evidence),
            "market_daily_incremental_benchmark_v1",
        ),
        "clean_matrix": _artifact(
            resolve(clean_matrix_status),
            "open_ledger_backend_parity_matrix_v1",
        ),
        "clean_acceptance": _artifact(
            resolve(clean_acceptance),
            "market_data_performance_acceptance_v1",
        ),
        "dual_read_matrix": _artifact(
            resolve(dual_read_matrix_status),
            "execution_market_dual_read_matrix_v1",
        ),
    }
    policy = load_policy(resolve(policy_path))
    promotion_audit = audit_promotion(root, policy)
    recovery_drill = _recovery_drill(policy, promotion_audit)

    ordered = (
        ("incremental_evidence", "incremental_benchmark"),
        ("clean_matrix", "clean_matrix"),
        ("clean_acceptance", "clean_acceptance"),
        ("dual_read_matrix", "dual_read"),
    )
    next_phase = next(
        (phase for artifact, phase in ordered if not artifacts[artifact]["passed"]),
        None,
    )
    if next_phase is None:
        if promotion_audit["status"] != "passed":
            next_phase = "promotion_audit"
        elif recovery_drill["passed"]:
            next_phase = "complete"
        elif policy["active_backend"] == "monthly":
            next_phase = "rollback_recovery_drill"
        else:
            next_phase = "manual_promotion"
    statuses = {
        "complete": "completed",
        "manual_promotion": "ready_for_manual_promotion",
        "rollback_recovery_drill": "recovery_drill_required",
    }
    return {
        "schema": "nt6_market_backend_closure_v1",
        "status": statuses.get(next_phase, "incomplete"),
        "next_phase": next_phase,
        "artifacts": artifacts,
        "promotion_audit": promotion_audit,
        "recovery_drill": recovery_drill,
        "automatic_promotion": False,
    }


def closure_paths(
    *,
    incremental_evidence: str | Path,
    clean_matrix_root: str | Path,
    clean_acceptance: str | Path,
    dual_read_root: str | Path,
    policy_path: str | Path,
) -> Mapping[str, str | Path]:
    return {
        "incremental_evidence": incremental_evidence,
        "clean_matrix_status": Path(clean_matrix_root) / "matrix_status.json",
        "clean_acceptance": clean_acceptance,
        "dual_read_matrix_status": (
            Path(dual_read_root) / "dual_read_matrix_status.json"
        ),
        "policy_path": policy_path,
    }
