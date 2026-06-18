"""Generate cleanup source inventory reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.source_inventory import (
    scan_top_level,
    write_inventory_csv,
    write_inventory_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate top-level cleanup inventory")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="reports/codebase_cleanup_20260618",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    items = scan_top_level(args.root)
    csv_path = output_dir / "source_inventory.csv"
    md_path = output_dir / "source_inventory.md"
    write_inventory_csv(items, csv_path)
    write_inventory_markdown(items, md_path)
    print(f"wrote {csv_path} rows={len(items)}", flush=True)
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
