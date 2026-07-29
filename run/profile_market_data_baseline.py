"""Freeze the read-only MD0 CSV, cache and ledger-parity baseline."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.market_data_profile import build_baseline_profile, write_profile


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", action="append", dest="csv_roots")
    parser.add_argument("--benchmark-root", default="data/forward_raw")
    parser.add_argument("--benchmark-start", default="2026-01-01")
    parser.add_argument("--benchmark-end", default="2026-07-29")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--matrix-cache-dir", default="cache/open_ledger_ohlc_matrix")
    parser.add_argument("--baseline-contract", default="registry/baseline_contract.json")
    parser.add_argument(
        "--artifact-inventory",
        default=(
            "reports/non_training_closure_20260719/"
            "nt1_baseline_freeze/artifact_inventory.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "reports/non_training_closure_20260719/"
            "nt6_market_data_baseline_20260730"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    csv_roots = args.csv_roots or ["data/raw", "data/forward_raw"]
    payload = build_baseline_profile(
        project_root=ROOT,
        csv_roots=[ROOT / path for path in csv_roots],
        benchmark_root=ROOT / args.benchmark_root,
        benchmark_start=args.benchmark_start,
        benchmark_end=args.benchmark_end,
        matrix_cache_dir=ROOT / args.matrix_cache_dir,
        baseline_contract=ROOT / args.baseline_contract,
        artifact_inventory=ROOT / args.artifact_inventory,
        max_files=args.max_files,
    )
    json_path, md_path = write_profile(ROOT / args.output_dir, payload)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
