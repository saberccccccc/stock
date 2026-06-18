"""Build an index for manually reviewed root-level project documents."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DOC_METADATA = {
    "README.md": {
        "role": "entrypoint",
        "cleanup_decision": "keep",
        "consolidation_target": "README.md",
        "notes": "General project overview and quick-start commands.",
    },
    "CLAUDE.md": {
        "role": "agent_guide",
        "cleanup_decision": "keep",
        "consolidation_target": "CLAUDE.md",
        "notes": "Operational notes, architecture overview, and experiment-branch history.",
    },
    "RESEARCH_PROTOCOL.md": {
        "role": "research_control",
        "cleanup_decision": "keep",
        "consolidation_target": "RESEARCH_PROTOCOL.md",
        "notes": "Research cutoff, data boundary, and model-selection protocol.",
    },
    "FROZEN_FORWARD_STRATEGY.md": {
        "role": "strategy_manifest",
        "cleanup_decision": "keep",
        "consolidation_target": "reports/codebase_cleanup_20260618/official_baselines.md",
        "notes": "Frozen live/shadow strategy decision made before forward observation.",
    },
    "TEST_PLAN.md": {
        "role": "strategy_plan",
        "cleanup_decision": "keep_or_consolidate",
        "consolidation_target": "reports/codebase_cleanup_20260618/official_baselines.md",
        "notes": "V9 small-account test plan and locked research result.",
    },
    "SHARPE_OPTIMIZATION_REPORT.md": {
        "role": "strategy_report",
        "cleanup_decision": "keep_or_consolidate",
        "consolidation_target": "reports/codebase_cleanup_20260618/official_baselines.md",
        "notes": "Small-account Sharpe optimization results and decision evidence.",
    },
    "LOSS_ABLATION_PLAN.md": {
        "role": "training_plan",
        "cleanup_decision": "keep_or_consolidate",
        "consolidation_target": "reports/codebase_cleanup_20260618/training_research_index.md",
        "notes": "Loss ablation protocol and promotion gates.",
    },
    "CANDIDATE_MODEL_VALIDATION_PLAN_20260614.md": {
        "role": "validation_plan",
        "cleanup_decision": "keep_or_consolidate",
        "consolidation_target": "reports/codebase_cleanup_20260618/training_research_index.md",
        "notes": "Purged V9 checkpoint validation plan and final decision.",
    },
    "PURGED_ALPHA_OPTIMIZATION_PLAN.md": {
        "role": "training_plan",
        "cleanup_decision": "keep_or_consolidate",
        "consolidation_target": "reports/codebase_cleanup_20260618/training_research_index.md",
        "notes": "Purged alpha optimization sequence and historical confirmation caveat.",
    },
    "RERANKER_IMPLEMENTATION_PLAN_20260614.md": {
        "role": "reranker_research",
        "cleanup_decision": "keep_or_consolidate",
        "consolidation_target": "reports/codebase_cleanup_20260618/reranker_research_index.md",
        "notes": "M0, V2, V3, V4, V4.1 reranker history and decisions.",
    },
    "RERANKER_V4_PLAN_20260615.md": {
        "role": "reranker_research",
        "cleanup_decision": "keep_or_consolidate",
        "consolidation_target": "reports/codebase_cleanup_20260618/reranker_research_index.md",
        "notes": "Confidence-gated V4 and V4.1 result summary.",
    },
    "EXPERIMENTS.md": {
        "role": "legacy_experiment_log",
        "cleanup_decision": "archive_after_consolidation",
        "consolidation_target": "reports/codebase_cleanup_20260618/training_research_index.md",
        "notes": "Short early experiment-branch log; keep until key points are merged.",
    },
}


@dataclass(frozen=True)
class ReviewDoc:
    name: str
    title: str
    role: str
    cleanup_decision: str
    consolidation_target: str
    notes: str
    headings: str
    length: int
    last_write: str


def _read_title_and_headings(path: Path) -> tuple[str, list[str]]:
    title = path.name
    headings = []
    seen_title = False
    in_code_block = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped.startswith("# ") and not seen_title:
            title = stripped[2:].strip()
            seen_title = True
        if stripped.startswith("## "):
            headings.append(stripped[3:].strip())
    return title, headings


def load_review_markdown_names(archive_plan_csv: str | Path) -> list[str]:
    with Path(archive_plan_csv).open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            row["name"]
            for row in rows
            if row["action"] == "review"
            and row["kind"] == "file"
            and row["name"].lower().endswith(".md")
        ]


def build_review_doc_index(
    root: str | Path,
    archive_plan_csv: str | Path,
) -> list[ReviewDoc]:
    root = Path(root)
    docs = []
    for name in load_review_markdown_names(archive_plan_csv):
        path = root / name
        meta = DOC_METADATA.get(
            name,
            {
                "role": "unknown",
                "cleanup_decision": "manual_review",
                "consolidation_target": "",
                "notes": "No metadata rule yet.",
            },
        )
        stat = path.stat()
        title, headings = _read_title_and_headings(path)
        docs.append(
            ReviewDoc(
                name=name,
                title=title,
                role=meta["role"],
                cleanup_decision=meta["cleanup_decision"],
                consolidation_target=meta["consolidation_target"],
                notes=meta["notes"],
                headings="; ".join(headings[:8]),
                length=int(stat.st_size),
                last_write=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
    return sorted(docs, key=lambda doc: (doc.role, doc.name.lower()))


def write_review_doc_csv(docs: list[ReviewDoc], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "title",
                "role",
                "cleanup_decision",
                "consolidation_target",
                "notes",
                "headings",
                "length",
                "last_write",
            ],
        )
        writer.writeheader()
        for doc in docs:
            writer.writerow(
                {
                    "name": doc.name,
                    "title": doc.title,
                    "role": doc.role,
                    "cleanup_decision": doc.cleanup_decision,
                    "consolidation_target": doc.consolidation_target,
                    "notes": doc.notes,
                    "headings": doc.headings,
                    "length": doc.length,
                    "last_write": doc.last_write,
                }
            )


def review_doc_markdown(docs: list[ReviewDoc]) -> str:
    counts = {}
    for doc in docs:
        counts[doc.cleanup_decision] = counts.get(doc.cleanup_decision, 0) + 1
    lines = [
        "# Review Document Index 2026-06-18",
        "",
        "This index covers root-level markdown files that remain in manual review.",
        "It does not move files; it records the intended cleanup path for each document.",
        "",
        "## Summary",
        "",
        "| Cleanup decision | Count |",
        "|---|---:|",
    ]
    for decision, count in sorted(counts.items()):
        lines.append(f"| {decision} | {count} |")
    lines.extend(
        [
            "",
            "## Documents",
            "",
            "| Name | Role | Decision | Consolidation target | Notes |",
            "|---|---|---|---|---|",
        ]
    )
    for doc in docs:
        lines.append(
            f"| {doc.name} | {doc.role} | {doc.cleanup_decision} | "
            f"{doc.consolidation_target} | {doc.notes} |"
        )
    lines.extend(
        [
            "",
            "## Heading Preview",
            "",
            "| Name | Title | Headings |",
            "|---|---|---|",
        ]
    )
    for doc in docs:
        lines.append(f"| {doc.name} | {doc.title} | {doc.headings} |")
    lines.append("")
    return "\n".join(lines)


def write_review_doc_markdown(docs: list[ReviewDoc], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(review_doc_markdown(docs), encoding="utf-8")
