"""Write a machine-readable coverage audit for realistic open-ledger inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from backtest.execution_coverage import audit_execution_coverage


def resolve_data_dir(data_dir, dataset_role):
    """Resolve the role-specific cache root unless the caller overrides it."""
    if data_dir:
        return Path(data_dir)
    return Path("data/forward_raw" if dataset_role == "forward" else "data/raw")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the cache root; defaults to data/raw for research or data/forward_raw for forward",
    )
    parser.add_argument("--matrix-cache-dir", default="cache/open_ledger_ohlc_matrix")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--dataset-role", choices=("research", "forward"), default="research")
    parser.add_argument("--output", default="reports/qlib_research_framework_20260712/execution_coverage.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir, args.dataset_role)
    report = audit_execution_coverage(
        ROOT / data_dir,
        ROOT / args.matrix_cache_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        dataset_role=args.dataset_role,
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "gaps": report["gaps"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
