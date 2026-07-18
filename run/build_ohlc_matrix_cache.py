"""Build the global OHLC matrix cache used by open-ledger backtests."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

from backtest.ohlc_matrix_cache import build_ohlc_matrix_cache, matrix_cache_is_current


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build global OHLC matrix cache")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--cache-dir", default="cache/open_ledger_ohlc_matrix")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.force and matrix_cache_is_current(args.data_dir, args.cache_dir):
        print(f"OHLC matrix cache is current: {args.cache_dir}", flush=True)
        return
    meta = build_ohlc_matrix_cache(
        args.data_dir,
        args.cache_dir,
        progress_every=args.progress_every,
    )
    print(
        f"Built OHLC matrix cache: {args.cache_dir} "
        f"codes={len(meta['codes'])} dates={len(meta['dates'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
