"""Create low-memory negative-filter alpha files from a full rerank result."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.transforms import write_negative_filter_alpha


def parse_args():
    parser = argparse.ArgumentParser(description="Create negative-filter alpha JSONL")
    parser.add_argument("--base-alpha", required=True)
    parser.add_argument("--full-rerank-alpha", required=True)
    parser.add_argument("--output-alpha", required=True)
    parser.add_argument("--start-rank", type=int, default=30)
    parser.add_argument("--end-rank", type=int, default=100)
    parser.add_argument("--drop-n", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    start_rank = max(0, int(args.start_rank))
    end_rank = int(args.end_rank)
    drop_n = max(0, int(args.drop_n))
    if end_rank <= start_rank:
        raise ValueError("--end-rank must be greater than --start-rank")

    stats = write_negative_filter_alpha(
        args.base_alpha,
        args.full_rerank_alpha,
        args.output_alpha,
        start_rank=start_rank,
        end_rank=end_rank,
        drop_n=drop_n,
    )
    print(
        f"wrote={stats['output']} rows={stats['rows']} "
        f"changed_dates={stats['changed_dates']}/{stats['rows']} "
        f"dropped_total={stats['dropped_total']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
