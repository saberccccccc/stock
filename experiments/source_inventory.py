"""Generate top-level project inventory reports for cleanup planning."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SOURCE_DIRS = {
    "alpha",
    "backtest",
    "configs",
    "core",
    "data",
    "experiments",
    "reports",
    "run",
    "scripts",
    "tests",
}

SOURCE_FILES = {
    "README.md",
    "CLAUDE.md",
    "EXPERIMENTS.md",
    "RESEARCH_PROTOCOL.md",
    "requirements.txt",
    "TEST_PLAN.md",
    "SHARPE_OPTIMIZATION_REPORT.md",
    "__init__.py",
}

VOLATILE_TOP_LEVEL_CACHE = {
    ".pytest_cache",
    "__pycache__",
}


@dataclass(frozen=True)
class InventoryItem:
    name: str
    kind: str
    item_class: str
    length: int | None
    last_write: str


def classify_top_level(name: str, kind: str) -> str:
    lower = name.lower()
    if name in SOURCE_DIRS or name in SOURCE_FILES or lower.endswith((".md", ".txt", ".ps1")):
        if lower.endswith((".pid", ".log")) or "_stdout.log" in lower or "_stderr.log" in lower:
            return "runtime_log_or_pid"
        if any(token in lower for token in ("plan", "report", "strategy", "protocol")):
            return "source_or_docs"
        if name in SOURCE_DIRS or name in SOURCE_FILES:
            return "source_or_docs"
    if lower in {".pytest_cache", "__pycache__", "archive", "cache"} or lower.startswith("_archive"):
        return "archive_or_cache"
    if kind == "dir" and lower == "logs":
        return "runtime_log_or_pid"
    if kind == "file" and (
        lower.endswith(".pid")
        or lower.endswith(".log")
        or "_stdout.log" in lower
        or "_stderr.log" in lower
        or lower.endswith(".err.log")
        or lower.endswith(".out.log")
    ):
        return "runtime_log_or_pid"
    if kind == "dir" and (
        lower.startswith("checkpoints")
        or lower.startswith("models_")
        or lower.startswith("switch_value_models")
    ):
        return "checkpoint_or_model"
    if kind == "dir" and (
        lower.startswith("backtest_results")
        or lower.startswith("reranker")
        or lower.startswith("v9_")
        or lower.startswith("open_reranker")
        or lower.startswith("forward_results")
        or lower.startswith("candidate_")
        or lower.startswith("loss_ablation")
        or lower.startswith("multi_loss")
        or lower.startswith("m0_")
        or lower.startswith("downside")
        or lower.startswith("lag1_")
        or lower.startswith("breadth_")
        or lower.startswith("state_")
        or lower.startswith("conditional_")
        or lower.startswith("unified_")
        or lower.startswith("locked_")
        or lower.startswith("diagnostics_")
    ):
        return "experiment_output"
    if kind == "file" and (
        lower.startswith("backtest_results")
        or lower.startswith("candidate_")
        or lower.startswith("loss_ablation")
        or lower.startswith("purged_")
        or lower.startswith("validate_")
        or lower.startswith("run_")
        or lower.startswith("resume_")
        or lower.startswith("forward_")
    ):
        return "experiment_output"
    return "misc"


def scan_top_level(root: str | Path) -> list[InventoryItem]:
    root = Path(root)
    items = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        name = path.name
        if name == ".git" or name in VOLATILE_TOP_LEVEL_CACHE:
            continue
        stat = path.stat()
        kind = "dir" if path.is_dir() else "file"
        length = None if path.is_dir() else int(stat.st_size)
        last_write = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        items.append(
            InventoryItem(
                name=name,
                kind=kind,
                item_class=classify_top_level(name, kind),
                length=length,
                last_write=last_write,
            )
        )
    return items


def write_inventory_csv(items: list[InventoryItem], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "kind", "class", "length", "last_write"])
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "name": item.name,
                    "kind": item.kind,
                    "class": item.item_class,
                    "length": "" if item.length is None else item.length,
                    "last_write": item.last_write,
                }
            )


def inventory_markdown(items: list[InventoryItem]) -> str:
    counts = Counter((item.item_class, item.kind) for item in items)
    lines = [
        "# Source Inventory 2026-06-18",
        "",
        "Generated without moving or modifying project outputs.",
        "",
        "## Summary",
        "",
        "| Group | Count |",
        "|---|---:|",
    ]
    for (item_class, kind), count in sorted(counts.items()):
        lines.append(f"| {item_class}, {kind} | {count} |")
    lines.extend(
        [
            "",
            "## Top-Level Items",
            "",
            "| Name | Kind | Class | Last Write |",
            "|---|---|---|---|",
        ]
    )
    for item in sorted(items, key=lambda row: (row.item_class, row.kind, row.name.lower())):
        lines.append(f"| {item.name} | {item.kind} | {item.item_class} | {item.last_write} |")
    lines.append("")
    return "\n".join(lines)


def write_inventory_markdown(items: list[InventoryItem], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(inventory_markdown(items), encoding="utf-8")
