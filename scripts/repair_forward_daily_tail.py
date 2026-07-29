"""Repair duplicate or out-of-order rows in recent forward daily CSV tails."""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.forward_daily_update import repair_recent_tail


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/forward_raw")
    parser.add_argument("--cutoff", required=True)
    parser.add_argument(
        "--durable",
        action="store_true",
        help="Force each repaired file to stable storage; substantially slower",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    metrics = Counter()
    for path in sorted(data_dir.glob("*.??.csv")):
        result = repair_recent_tail(path, args.cutoff, durable=args.durable)
        metrics["files"] += 1
        if result.changed:
            metrics["changed_files"] += 1
            metrics["removed_rows"] += result.removed_rows
            metrics["reordered_files"] += int(result.reordered)
        if metrics["files"] % 500 == 0:
            print(dict(metrics), flush=True)
    print(dict(metrics))


if __name__ == "__main__":
    main()
