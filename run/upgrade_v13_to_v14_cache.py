"""Upgrade a v13 cross-sectional cache without rebuilding feature matrices."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.cache_upgrade import upgrade_v13_cache


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-meta", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-meta", default=None)
    parser.add_argument("--stock-chunk", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    upgrade_v13_cache(
        args.source_meta,
        data_dir=args.data_dir,
        output_meta_path=args.output_meta,
        overwrite=args.overwrite,
        stock_chunk=args.stock_chunk,
    )


if __name__ == "__main__":
    main()
