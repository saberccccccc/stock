"""Promotion and rollback gates for the execution-market backend."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "execution_market_backend_policy_v1"
STATES = {"legacy_active", "dual_read_ready", "monthly_active", "rolled_back"}


def load_policy(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    if value.get("schema") != SCHEMA:
        raise ValueError("invalid execution-market backend policy schema")
    if value.get("state") not in STATES:
        raise ValueError("invalid execution-market backend policy state")
    if value.get("active_backend") not in {"legacy", "monthly"}:
        raise ValueError("policy active_backend must be legacy or monthly")
    if value.get("rollback_backend") != "legacy":
        raise ValueError("policy rollback_backend must remain legacy")
    if value.get("candidate_backend") != "monthly":
        raise ValueError("policy candidate_backend must remain monthly")
    if value.get("shadow_backend") != "csv":
        raise ValueError("policy shadow_backend must remain csv")
    required = list(value.get("required_dual_read_splits", []))
    if required != ["val_2024", "test_2025", "forward_2026"]:
        raise ValueError("policy requires Val, Test and Forward dual-read evidence")
    return value


def _read_status(
    root: Path,
    relative: str,
    *,
    expected_schema: str,
    expected_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        return {"status": "missing", "path": str(path), "passed": False}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    status = str(value.get("status", "missing"))
    schema_matches = value.get("schema") == expected_schema
    fields_match = all(
        value.get(key) == expected
        for key, expected in (expected_fields or {}).items()
    )
    return {
        "status": status,
        "path": str(path),
        "schema": value.get("schema"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "passed": status == "passed" and schema_matches and fields_match,
    }


def audit_promotion(project_root: str | Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    evidence = policy["evidence"]
    checks = {}
    schemas = {
        "behavior_parity": "open_ledger_backend_parity_matrix_v1",
        "performance_acceptance": "market_data_performance_acceptance_v1",
        "call_site_audit": "market_data_call_site_audit_v1",
    }
    for name, schema in schemas.items():
        checks[name] = _read_status(
            root,
            evidence[name],
            expected_schema=schema,
        )
    dual = {}
    for split in policy["required_dual_read_splits"]:
        dual[split] = _read_status(
            root,
            evidence["dual_read_reports"][split],
            expected_schema="execution_market_dual_read_v1",
            expected_fields={
                "primary_backend": "monthly",
                "shadow_backend": "csv",
            },
        )
    checks["dual_read"] = dual
    passed = all(
        item["passed"]
        for name, item in checks.items()
        if name != "dual_read"
    ) and all(item["passed"] for item in dual.values())
    return {
        "schema": "execution_market_backend_promotion_audit_v1",
        "status": "passed" if passed else "blocked",
        "active_backend": policy["active_backend"],
        "candidate_backend": policy["candidate_backend"],
        "checks": checks,
    }


def write_policy_atomic(path: str | Path, policy: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(policy), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
