"""Dry-run or execute archive moves from an archive plan CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.archive_plan import (
    build_archive_moves,
    execute_archive_moves,
    load_archive_plan,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Archive files from a generated archive plan")
    parser.add_argument(
        "--plan-csv",
        default="reports/codebase_cleanup_20260618/archive_plan.csv",
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--class", dest="item_class", default=None, help="Only include one inventory class.")
    parser.add_argument("--target", default=None, help="Only include one archive target path.")
    parser.add_argument("--name-prefix", default=None, help="Only include archive candidates with this name prefix.")
    parser.add_argument("--name-glob", default=None, help="Only include archive candidates matching this glob.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move archive candidates. Omit for dry-run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_archive_plan(args.plan_csv)
    moves = build_archive_moves(
        rows,
        root=args.root,
        item_class=args.item_class,
        target=args.target,
        name_prefix=args.name_prefix,
        name_glob=args.name_glob,
    )
    if args.limit is not None:
        moves = moves[: args.limit]
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"{mode} archive moves: {len(moves)}", flush=True)
    for move in moves:
        print(f"{move.source} -> {move.target}", flush=True)
    if args.execute:
        execute_archive_moves(moves)
        print(f"moved={len(moves)}", flush=True)


if __name__ == "__main__":
    main()
