"""Audit the complete active MarketDailyStore manifest graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.market_daily_store import MarketDailyStore


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--skip-physical-hashes",
        action="store_true",
        help="Validate metadata only; full acceptance must not use this option.",
    )
    args = parser.parse_args()

    result = MarketDailyStore(args.store_root).audit(
        verify_physical_hashes=not args.skip_physical_hashes
    )
    if args.output:
        _write_json_atomic(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
