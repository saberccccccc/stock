"""Generate checkpoint/model reference audit reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.checkpoint_reference_audit import (
    build_checkpoint_reference_audit,
    write_checkpoint_audit_csv,
    write_checkpoint_audit_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate checkpoint reference audit")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--archive-plan-csv",
        default="reports/codebase_cleanup_20260618/archive_plan.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/codebase_cleanup_20260618",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = build_checkpoint_reference_audit(args.archive_plan_csv, root=args.root)
    csv_path = output_dir / "checkpoint_reference_audit.csv"
    md_path = output_dir / "checkpoint_reference_audit.md"
    write_checkpoint_audit_csv(rows, csv_path)
    write_checkpoint_audit_markdown(rows, md_path)
    print(f"wrote {csv_path} rows={len(rows)}", flush=True)
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
