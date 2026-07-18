"""Audit physical market-cache coverage and effective research/forward bounds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.boundary_audit import audit_data_root, resolve_data_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--dataset-role", choices=("research", "forward"), default="research")
    parser.add_argument("--effective-end-date", default=None)
    parser.add_argument(
        "--output",
        default="reports/qlib_research_framework_20260712/data_boundary_audit.json",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = audit_data_root(
        resolve_data_dir(args.data_dir, args.dataset_role),
        dataset_role=args.dataset_role,
        effective_end_date=args.effective_end_date,
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
