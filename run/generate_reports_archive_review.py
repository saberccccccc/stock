"""Create a non-destructive archive review for reports/.

Registered evidence, official outputs, and current state-aware research are
protected automatically. Older, unregistered research folders are marked as
archive candidates only; this script never moves or deletes anything.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--registry-dir", default="registry")
    parser.add_argument("--output-dir", default="reports/archive_review_20260710")
    parser.add_argument("--candidate-before", default="20260701")
    return parser.parse_args(argv)


def registered_report_roots(registry_dir):
    roots = set()
    for name in ("reports.csv", "candidates.csv", "attributions.csv"):
        path = registry_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for value in row.values():
                    if not value:
                        continue
                    for fragment in str(value).split(";"):
                        normalized = fragment.replace("\\", "/").strip('" ')
                        if normalized.startswith("reports/"):
                            parts = normalized.split("/")
                            if len(parts) >= 2:
                                roots.add(parts[1])
    return roots


def directory_size(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def dated_before(name, cutoff):
    return any(date < cutoff for date in re.findall(r"(?<!\d)(20\d{6})(?!\d)", name))


def classify(name, registered, cutoff):
    if name in registered:
        return "protect", "referenced by registry evidence"
    if name in {"official", "archive", "archive_review_20260710"} or name.startswith("official_registry_"):
        return "protect", "official or archive-management output"
    if name.startswith(("state_aware_", "sa_20260710", "recent_candidate_", "cond_pairrisk_")):
        return "protect", "current state-aware research output"
    if dated_before(name, cutoff):
        return "archive_candidate", f"unregistered report folder dated before {cutoff}"
    return "review", "unregistered recent or undated report folder"


def main(argv=None):
    args = parse_args(argv)
    reports_dir = ROOT / args.reports_dir
    registry_dir = ROOT / args.registry_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    registered = registered_report_roots(registry_dir)
    rows = []
    for path in sorted(reports_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir():
            continue
        action, reason = classify(path.name, registered, args.candidate_before)
        size_bytes = directory_size(path)
        rows.append(
            {
                "name": path.name,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "action": action,
                "size_bytes": size_bytes,
                "size_gb": round(size_bytes / (1024 ** 3), 3),
                "reason": reason,
            }
        )
    csv_path = output_dir / "reports_archive_review.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    sizes = {}
    for row in rows:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
        sizes[row["action"]] = sizes.get(row["action"], 0.0) + row["size_gb"]
    lines = [
        "# Reports Archive Review - 2026-07-10",
        "",
        "This is a non-destructive review. No files were moved or deleted.",
        "",
        "| action | folders | size_gb |",
        "|---|---:|---:|",
    ]
    for action in sorted(counts):
        lines.append(f"| {action} | {counts[action]} | {sizes[action]:.3f} |")
    lines.extend(["", "## Archive Candidates", "", "| name | size_gb | reason |", "|---|---:|---|"])
    for row in rows:
        if row["action"] == "archive_candidate":
            lines.append(f"| {row['name']} | {row['size_gb']:.3f} | {row['reason']} |")
    (output_dir / "reports_archive_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"output_dir": str(output_dir), "counts": counts, "size_gb": sizes}, flush=True)


if __name__ == "__main__":
    main()
