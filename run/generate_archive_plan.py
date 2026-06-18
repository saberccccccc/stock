"""Generate a non-destructive archive plan from the source inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.archive_plan import (
    build_archive_plan,
    write_archive_plan_csv,
    write_archive_plan_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cleanup archive plan")
    parser.add_argument(
        "--inventory-csv",
        default="reports/codebase_cleanup_20260618/source_inventory.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/codebase_cleanup_20260618",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = build_archive_plan(args.inventory_csv)
    csv_path = output_dir / "archive_plan.csv"
    md_path = output_dir / "archive_plan.md"
    write_archive_plan_csv(rows, csv_path)
    write_archive_plan_markdown(rows, md_path)
    print(f"wrote {csv_path} rows={len(rows)}", flush=True)
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
