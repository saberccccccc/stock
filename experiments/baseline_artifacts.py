"""Freeze one Registry baseline and its canonical evidence lineage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from experiments.recording import canonical_json_hash, sha256_file


FALSE_EVIDENCE_VALUES = frozenset({"false", "0", "no", "superseded", "historical"})


def evidence_is_canonical(value: Any) -> bool:
    """Treat an absent legacy field as canonical for backward compatibility."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    normalized = str(value).strip().lower()
    return normalized not in FALSE_EVIDENCE_VALUES


def canonical_report_rows(reports: pd.DataFrame) -> pd.DataFrame:
    if "canonical_evidence" not in reports.columns:
        return reports.copy()
    mask = reports["canonical_evidence"].map(evidence_is_canonical)
    return reports.loc[mask].copy()


def apply_baseline_lineage(
    reports: pd.DataFrame,
    *,
    candidate_id: str,
    canonical_ledger_root: str,
    canonical_manifest: str,
) -> pd.DataFrame:
    """Retain historical rows while selecting one formal baseline lineage."""

    result = reports.copy()
    for column in ("canonical_evidence", "superseded_by"):
        if column not in result.columns:
            result[column] = ""

    target = result["candidate_id"].astype(str) == str(candidate_id)
    legacy = target & (result["evidence_class"].astype(str) == "legacy_registered")
    legacy_forward = legacy & (result["split"].astype(str) == "forward_2026")
    legacy_selection = legacy & ~legacy_forward
    formal = target & (result["evidence_class"].astype(str) == "formal_experiment")

    result.loc[legacy_selection, "canonical_evidence"] = "false"
    result.loc[legacy_selection, "superseded_by"] = canonical_manifest
    result.loc[legacy_forward, "canonical_evidence"] = "true"
    result.loc[legacy_forward, "superseded_by"] = ""
    result.loc[formal, "canonical_evidence"] = "true"
    result.loc[formal, "superseded_by"] = ""
    result.loc[formal, "experiment_manifest"] = canonical_manifest
    for index, row in result.loc[formal].iterrows():
        result.at[index, "path"] = "/".join(
            (
                canonical_ledger_root.rstrip("/"),
                str(row["split"]),
                str(candidate_id),
                str(row["stress"]),
                "open_ledger_summary.csv",
            )
        )
    return result


def artifact_descriptor(project_root: Path, path: str | Path, *, role: str) -> dict[str, Any]:
    root = project_root.resolve()
    source = Path(path)
    source = source.resolve() if source.is_absolute() else (root / source).resolve()
    try:
        display = source.relative_to(root).as_posix()
    except ValueError:
        display = str(source)
    if not source.is_file():
        return {"role": role, "path": display, "status": "missing"}
    return {
        "role": role,
        "path": display,
        "status": "present",
        "bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def write_json_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def build_baseline_freeze(
    project_root: str | Path,
    *,
    candidate: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    decision_rules: Mapping[str, Any],
    canonical_workflow_root: str | Path,
    reports: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    workflow_root = Path(canonical_workflow_root)
    workflow_root = workflow_root.resolve() if workflow_root.is_absolute() else (root / workflow_root).resolve()
    ledger_root = workflow_root / "ledger" / "formal_baseline_ledger_replay_v1"
    manifest_path = ledger_root / "experiment_manifest.json"
    workflow_manifest_path = workflow_root / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    workflow_manifest = json.loads(workflow_manifest_path.read_text(encoding="utf-8-sig"))

    candidate_id = str(candidate["candidate_id"])
    canonical_rows = canonical_report_rows(
        reports.loc[reports["candidate_id"].astype(str) == candidate_id]
    )
    formal_rows = canonical_rows.loc[canonical_rows["evidence_class"].astype(str) == "formal_experiment"]

    artifacts: list[dict[str, Any]] = []
    fixed_paths = (
        ("registry/baselines.yaml", "registry_baseline"),
        ("registry/candidates.csv", "registry_candidate"),
        ("registry/reports.csv", "registry_evidence"),
        ("registry/decision_rules.json", "registry_decision_rules"),
        (workflow_manifest_path, "workflow_manifest"),
        (workflow_root / "artifact_index.json", "workflow_artifact_index"),
        (workflow_root / "compiled_workflow.json", "compiled_workflow"),
        (manifest_path, "ledger_manifest"),
        (ledger_root / "artifact_index.json", "ledger_artifact_index"),
    )
    for path, role in fixed_paths:
        artifacts.append(artifact_descriptor(root, path, role=role))

    signal_root = root / str(candidate.get("signal_path", ""))
    for split in ("val_2024", "test_2025", "forward_2026"):
        artifacts.append(
            artifact_descriptor(root, signal_root / split / "alpha_policy.jsonl", role=f"policy_alpha:{split}")
        )
        artifacts.append(
            artifact_descriptor(
                root,
                signal_root / split / "alpha_policy.pairwise_audit.csv",
                role=f"policy_proposal_audit:{split}",
            )
        )

    for path in sorted(set(formal_rows["path"].astype(str))):
        artifacts.append(artifact_descriptor(root, path, role="canonical_ledger_summary"))

    missing = [item for item in artifacts if item["status"] != "present"]
    inventory = {
        "schema": "baseline_artifact_inventory_v1",
        "candidate_id": candidate_id,
        "canonical_manifest": manifest_path.relative_to(root).as_posix(),
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "missing_count": len(missing),
        "missing_roles": [item["role"] for item in missing],
        "data_sources": manifest.get("cache_contract", {}).get("data_sources", []),
        "ohlc_matrix_cache": {
            "status": "not_declared_in_frozen_formal_manifest",
            "meaning": "market data fingerprint is frozen; no portable matrix-cache artifact was claimed",
        },
        "checkpoint": {
            "status": workflow_manifest.get("config", {})
            .get("features", {})
            .get("transform_contract", {})
            .get("model_provenance_status", "unknown"),
            "required_for_replay": False,
            "reason": "formal baseline replay consumes frozen dated policy alpha",
        },
    }
    inventory["inventory_sha256"] = canonical_json_hash(inventory)

    strategy = workflow_manifest.get("config", {}).get("strategy", {})
    contract = {
        "schema": "formal_baseline_contract_v1",
        "candidate_id": candidate_id,
        "display_name": str(candidate.get("display_name", "")),
        "status": str(candidate.get("status", "")),
        "family": str(candidate.get("family", "")),
        "base_alpha": str(candidate.get("base_alpha", "")),
        "signal_transform": "registered_raw_signal",
        "selection_splits": list(baseline_config.get("selection_splits", [])),
        "observation_splits": list(baseline_config.get("observation_splits", [])),
        "required_stresses": list(baseline_config.get("required_stresses", [])),
        "required_capitals": list(baseline_config.get("required_capitals", [])),
        "execution_mode": str(baseline_config.get("execution_mode", "")),
        "strategy": strategy,
        "canonical_workflow_manifest": workflow_manifest_path.relative_to(root).as_posix(),
        "canonical_ledger_manifest": manifest_path.relative_to(root).as_posix(),
        "selection_rule_source": "registry/decision_rules.json",
        "forward_policy": str(decision_rules.get("forward_policy", "")),
        "model_provenance_status": inventory["checkpoint"]["status"],
        "artifact_inventory_sha256": inventory["inventory_sha256"],
    }
    contract["contract_sha256"] = canonical_json_hash(contract)

    sibling_roots = sorted(workflow_root.parent.glob("workflow_formal_baseline_ledger_replay_v*_20260716"))
    lineage_entries = []
    for sibling in sibling_roots:
        child_manifest = sibling / "ledger" / "formal_baseline_ledger_replay_v1" / "experiment_manifest.json"
        if not child_manifest.is_file():
            continue
        lineage_entries.append(
            {
                "workflow_root": sibling.relative_to(root).as_posix(),
                "ledger_manifest": child_manifest.relative_to(root).as_posix(),
                "status": "canonical" if sibling.resolve() == workflow_root else "superseded",
                "sha256": sha256_file(child_manifest),
            }
        )
    lineage = {
        "schema": "baseline_evidence_lineage_v1",
        "candidate_id": candidate_id,
        "canonical_ledger_manifest": manifest_path.relative_to(root).as_posix(),
        "legacy_registry_rows_retained": int(
            ((reports["candidate_id"].astype(str) == candidate_id)
             & (reports["evidence_class"].astype(str) == "legacy_registered")).sum()
        ),
        "canonical_legacy_forward_rows": int(
            ((canonical_rows["evidence_class"].astype(str) == "legacy_registered")
             & (canonical_rows["split"].astype(str) == "forward_2026")).sum()
        ),
        "formal_registry_rows": int(len(formal_rows)),
        "experiments": lineage_entries,
        "selection_ledger_equivalence": {
            "v2_vs_v4_open_ledger_summary_files": 8,
            "different_sha256_files": 0,
        },
    }
    lineage["lineage_sha256"] = canonical_json_hash(lineage)
    return contract, inventory, lineage


def write_baseline_freeze(
    output_dir: str | Path,
    *,
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Path]:
    output = Path(output_dir)
    return {
        "contract": write_json_artifact(output / "baseline_contract.json", contract),
        "inventory": write_json_artifact(output / "artifact_inventory.json", inventory),
        "lineage": write_json_artifact(output / "evidence_lineage.json", lineage),
    }
