"""Read-only NT6 market-backend closure state inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from data.execution_market_backend_policy import audit_promotion, load_policy


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

    ordered = (
        ("incremental_evidence", "incremental_benchmark"),
        ("clean_matrix", "clean_matrix"),
        ("clean_acceptance", "clean_acceptance"),
        ("dual_read_matrix", "dual_read"),
    )
    next_phase = next(
        (phase for artifact, phase in ordered if not artifacts[artifact]["passed"]),
        "manual_promotion" if promotion_audit["status"] == "passed" else "promotion_audit",
    )
    ready = next_phase == "manual_promotion"
    return {
        "schema": "nt6_market_backend_closure_v1",
        "status": "ready_for_manual_promotion" if ready else "incomplete",
        "next_phase": next_phase,
        "artifacts": artifacts,
        "promotion_audit": promotion_audit,
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
