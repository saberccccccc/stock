"""Audit references to checkpoint/model cleanup candidates."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from experiments.archive_plan import ArchivePlanRow, load_archive_plan


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}

SEARCH_DIRS = {
    "configs",
    "core",
    "experiments",
    "reports",
    "run",
    "scripts",
    "tests",
}

EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "archive",
    "cache",
    "data",
}

GENERATED_CLEANUP_FILES = {
    "reports/codebase_cleanup_20260618/archive_plan.csv",
    "reports/codebase_cleanup_20260618/archive_plan.md",
    "reports/codebase_cleanup_20260618/source_inventory.csv",
    "reports/codebase_cleanup_20260618/source_inventory.md",
    "reports/codebase_cleanup_20260618/checkpoint_reference_audit.csv",
    "reports/codebase_cleanup_20260618/checkpoint_reference_audit.md",
    "experiments/checkpoint_reference_audit.py",
    "tests/test_checkpoint_reference_audit.py",
}

GENERATED_FILE_NAMES = {
    "archive_plan.csv",
    "archive_plan.md",
    "source_inventory.csv",
    "source_inventory.md",
    "checkpoint_reference_audit.csv",
    "checkpoint_reference_audit.md",
}


@dataclass(frozen=True)
class ReferenceHit:
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class CheckpointAuditRow:
    name: str
    action: str
    group: str
    reference_count: int
    reference_files: str
    decision: str
    note: str


def checkpoint_group(name: str) -> str:
    if name.startswith("checkpoints_loss_ablation_M"):
        return "protected_m_baseline"
    if name.startswith("checkpoints_loss_ablation_A"):
        return "loss_ablation_a_series"
    if name.startswith("checkpoints_loss_ablation_D") or name.startswith(
        "checkpoints_loss_ablation_LAG"
    ) or name.startswith("checkpoints_loss_ablation_T"):
        return "downside_lag_topfocus_series"
    if name.startswith("checkpoints_reranker_oof_"):
        return "reranker_oof"
    if name.startswith("checkpoints_exp_topfocus"):
        return "legacy_topfocus"
    if name.startswith("checkpoints_exp") or name == "checkpoints":
        return "alpha_checkpoint"
    if name.startswith("models_multi_v9"):
        return "legacy_v9_model"
    if name.startswith("switch_value_models"):
        return "switch_value_model"
    return "other_model"


def iter_reference_files(root: str | Path) -> list[Path]:
    root = Path(root)
    paths: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if child.name in EXCLUDED_DIRS:
            continue
        if child.is_dir():
            if child.name not in SEARCH_DIRS:
                continue
            for path in sorted(child.rglob("*"), key=lambda item: str(item).lower()):
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
                    continue
                if rel in GENERATED_CLEANUP_FILES:
                    continue
                if path.name in GENERATED_FILE_NAMES:
                    continue
                if path.suffix.lower() in TEXT_SUFFIXES:
                    paths.append(path)
        elif (
            child.is_file()
            and child.suffix.lower() in TEXT_SUFFIXES
            and child.name not in GENERATED_FILE_NAMES
        ):
            paths.append(child)
    return paths


def find_references(root: str | Path, names: list[str]) -> dict[str, list[ReferenceHit]]:
    root = Path(root)
    hits = {name: [] for name in names}
    for path in iter_reference_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            for name in names:
                if name in line:
                    hits[name].append(ReferenceHit(path=rel, line=line_no, text=line.strip()))
    return hits


def load_checkpoint_candidates(archive_plan_csv: str | Path) -> list[ArchivePlanRow]:
    return [
        row
        for row in load_archive_plan(archive_plan_csv)
        if row.item_class == "checkpoint_or_model"
    ]


def decide_checkpoint_action(row: ArchivePlanRow, refs: list[ReferenceHit]) -> tuple[str, str]:
    group = checkpoint_group(row.name)
    if row.action == "protect":
        return "keep", "protected by archive plan"
    if group in {"protected_m_baseline", "alpha_checkpoint", "legacy_v9_model"}:
        return "hold", "high-risk model/checkpoint family; audit manually before moving"
    if refs:
        return "hold", "referenced by text sources; inspect references before moving"
    if group in {"reranker_oof", "switch_value_model", "other_model"}:
        return "archive_after_matching_artifact_ledger", "no direct text references found"
    if group in {"loss_ablation_a_series", "downside_lag_topfocus_series", "legacy_topfocus"}:
        return "archive_after_result_summary", "no direct text references found"
    return "review", "manual review"


def build_checkpoint_reference_audit(
    archive_plan_csv: str | Path,
    root: str | Path = ".",
) -> list[CheckpointAuditRow]:
    candidates = load_checkpoint_candidates(archive_plan_csv)
    names = [row.name for row in candidates]
    references = find_references(root, names)
    rows: list[CheckpointAuditRow] = []
    for candidate in candidates:
        refs = references[candidate.name]
        decision, note = decide_checkpoint_action(candidate, refs)
        reference_files = ";".join(sorted({hit.path for hit in refs}))
        rows.append(
            CheckpointAuditRow(
                name=candidate.name,
                action=candidate.action,
                group=checkpoint_group(candidate.name),
                reference_count=len(refs),
                reference_files=reference_files,
                decision=decision,
                note=note,
            )
        )
    return rows


def write_checkpoint_audit_csv(rows: list[CheckpointAuditRow], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "action",
                "group",
                "reference_count",
                "reference_files",
                "decision",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row.name,
                    "action": row.action,
                    "group": row.group,
                    "reference_count": row.reference_count,
                    "reference_files": row.reference_files,
                    "decision": row.decision,
                    "note": row.note,
                }
            )


def checkpoint_audit_markdown(rows: list[CheckpointAuditRow]) -> str:
    lines = [
        "# Checkpoint Reference Audit 2026-06-19",
        "",
        "This audit searches lightweight text sources for references to checkpoint/model",
        "archive candidates. It is a cleanup guide, not permission for broad checkpoint moves.",
        "",
        "## Summary",
        "",
        "| Decision | Count |",
        "|---|---:|",
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.decision] = counts.get(row.decision, 0) + 1
    for decision, count in sorted(counts.items()):
        lines.append(f"| `{decision}` | {count} |")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Name | Action | Group | References | Decision | Note |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row.name}` | `{row.action}` | `{row.group}` | "
            f"{row.reference_count} | `{row.decision}` | {row.note} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_checkpoint_audit_markdown(rows: list[CheckpointAuditRow], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(checkpoint_audit_markdown(rows), encoding="utf-8")
