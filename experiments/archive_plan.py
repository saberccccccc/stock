"""Build non-destructive archive plans from source inventory rows."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


PROTECTED_PATHS = frozenset(
    {
        "alpha",
        "backtest",
        "cache",
        "configs",
        "core",
        "data",
        "experiments",
        "forward_results",
        "reports",
        "run",
        "scripts",
        "tests",
        "v9_avgw3_open_ledger_20260617",
        "v9_avgw3_extend_to_20260518_20260616",
        "checkpoints_exp_topfocus_w005_topic",
        "checkpoints_loss_ablation_M0_nomulti",
        "checkpoints_loss_ablation_M1_nomulti_topfocus_w005",
    }
)

PROTECTED_PREFIXES = (
    "forward_results/",
    "v9_avgw3_open_ledger_20260617/",
)

ARCHIVE_TARGETS = {
    "runtime_log_or_pid": "archive/logs_202606",
    "archive_or_cache": "archive/cache_202606",
    "experiment_output": "archive/experiments_202606",
    "checkpoint_or_model": "archive/checkpoints_202606",
}


@dataclass(frozen=True)
class ArchivePlanRow:
    name: str
    kind: str
    item_class: str
    action: str
    target: str
    reason: str


@dataclass(frozen=True)
class ArchiveMove:
    name: str
    source: Path
    target: Path
    action: str
    reason: str


def is_protected(name: str) -> bool:
    normalized = name.replace("\\", "/").strip("/")
    return normalized in PROTECTED_PATHS or any(
        normalized.startswith(prefix) for prefix in PROTECTED_PREFIXES
    )


def plan_inventory_row(row: dict[str, str]) -> ArchivePlanRow:
    name = row["name"]
    kind = row["kind"]
    item_class = row["class"]
    if is_protected(name):
        return ArchivePlanRow(
            name=name,
            kind=kind,
            item_class=item_class,
            action="protect",
            target="",
            reason="active source, official baseline, or forward-observation dependency",
        )
    if item_class in ARCHIVE_TARGETS:
        return ArchivePlanRow(
            name=name,
            kind=kind,
            item_class=item_class,
            action="archive_candidate",
            target=ARCHIVE_TARGETS[item_class],
            reason=f"classified as {item_class}",
        )
    return ArchivePlanRow(
        name=name,
        kind=kind,
        item_class=item_class,
        action="review",
        target="",
        reason="manual review before any move",
    )


def load_inventory(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_archive_plan(path: str | Path) -> list[ArchivePlanRow]:
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                ArchivePlanRow(
                    name=row["name"],
                    kind=row["kind"],
                    item_class=row["class"],
                    action=row["action"],
                    target=row["target"],
                    reason=row["reason"],
                )
            )
    return rows


def build_archive_plan(inventory_csv: str | Path) -> list[ArchivePlanRow]:
    return [plan_inventory_row(row) for row in load_inventory(inventory_csv)]


def write_archive_plan_csv(rows: list[ArchivePlanRow], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "kind", "class", "action", "target", "reason"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row.name,
                    "kind": row.kind,
                    "class": row.item_class,
                    "action": row.action,
                    "target": row.target,
                    "reason": row.reason,
                }
            )


def archive_plan_markdown(rows: list[ArchivePlanRow]) -> str:
    counts = {}
    for row in rows:
        counts[row.action] = counts.get(row.action, 0) + 1
    lines = [
        "# Archive Plan 2026-06-18",
        "",
        "This is a non-destructive plan. No files are moved by this report.",
        "",
        "## Summary",
        "",
        "| Action | Count |",
        "|---|---:|",
    ]
    for action, count in sorted(counts.items()):
        lines.append(f"| {action} | {count} |")
    lines.extend(
        [
            "",
            "## Protected Paths",
            "",
        ]
    )
    for name in sorted(PROTECTED_PATHS):
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "## Archive Candidates",
            "",
            "| Name | Kind | Class | Target |",
            "|---|---|---|---|",
        ]
    )
    for row in rows:
        if row.action == "archive_candidate":
            lines.append(f"| {row.name} | {row.kind} | {row.item_class} | {row.target} |")
    lines.extend(
        [
            "",
            "## Manual Review",
            "",
            "| Name | Kind | Class | Reason |",
            "|---|---|---|---|",
        ]
    )
    for row in rows:
        if row.action == "review":
            lines.append(f"| {row.name} | {row.kind} | {row.item_class} | {row.reason} |")
    lines.append("")
    return "\n".join(lines)


def write_archive_plan_markdown(rows: list[ArchivePlanRow], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(archive_plan_markdown(rows), encoding="utf-8")


def build_archive_moves(
    rows: list[ArchivePlanRow],
    root: str | Path = ".",
    include_missing: bool = False,
) -> list[ArchiveMove]:
    root = Path(root)
    moves = []
    for row in rows:
        if row.action != "archive_candidate":
            continue
        if not row.target:
            raise ValueError(f"Archive candidate has no target: {row.name}")
        if is_protected(row.name):
            raise ValueError(f"Refusing to move protected path: {row.name}")
        source = root / row.name
        if not source.exists() and not include_missing:
            continue
        moves.append(
            ArchiveMove(
                name=row.name,
                source=source,
                target=root / row.target / row.name,
                action=row.action,
                reason=row.reason,
            )
        )
    return moves


def execute_archive_moves(moves: list[ArchiveMove]) -> None:
    for move in moves:
        if is_protected(move.name):
            raise ValueError(f"Refusing to move protected path: {move.name}")
        if not move.source.exists():
            raise FileNotFoundError(move.source)
        if move.target.exists():
            raise FileExistsError(move.target)
        move.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(move.source), str(move.target))
