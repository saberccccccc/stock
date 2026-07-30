"""Benchmark isolated one-day market-data commit and monthly-cache refresh."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.market_daily_incremental_benchmark import (
    benchmark_daily_incremental_refresh,
)
from backtest.market_data_contract import configured_candidate_paths


def parse_args(argv=None):
    configured_store, _ = configured_candidate_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-store-root",
        default=configured_store,
    )
    parser.add_argument("--month")
    parser.add_argument("--workspace-parent")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def main(argv=None):
    args = parse_args(argv)
    result = benchmark_daily_incremental_refresh(
        _resolve(args.source_store_root),
        month=args.month,
        workspace_parent=(
            _resolve(args.workspace_parent) if args.workspace_parent else None
        ),
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
