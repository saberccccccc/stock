"""Build low-memory edge-only rerank alpha files from an existing full rerank."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.transforms import write_edge_rerank_alpha


def parse_args():
    parser = argparse.ArgumentParser(description="Create edge-only rerank alpha JSONL")
    parser.add_argument("--base-alpha", required=True)
    parser.add_argument("--full-rerank-alpha", required=True)
    parser.add_argument("--output-alpha", required=True)
    parser.add_argument("--start-rank", type=int, required=True)
    parser.add_argument("--end-rank", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    start_rank = max(0, int(args.start_rank))
    end_rank = int(args.end_rank)
    if end_rank <= start_rank:
        raise ValueError("--end-rank must be greater than --start-rank")

    stats = write_edge_rerank_alpha(
        args.base_alpha,
        args.full_rerank_alpha,
        args.output_alpha,
        start_rank=start_rank,
        end_rank=end_rank,
    )
    print(
        f"wrote={stats['output']} rows={stats['rows']} "
        f"changed_dates={stats['changed_dates']}/{stats['rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
