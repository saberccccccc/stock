"""Audit governed market-data call sites and forbid unregistered legacy imports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.market_data_access_audit import audit_market_data_call_sites


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="configs/market_data_call_sites.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = audit_market_data_call_sites(ROOT, policy_path=args.policy)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
