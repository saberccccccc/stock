"""Audit, promote, or roll back the execution-market backend policy."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.execution_market_backend_policy import (
    audit_promotion,
    load_policy,
    promote_policy,
    rollback_policy,
    write_policy_atomic,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "promote", "rollback"))
    parser.add_argument(
        "--policy", default="configs/execution_market_backend_policy.json"
    )
    parser.add_argument("--actor", default="")
    parser.add_argument("--reason", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    policy_path = Path(args.policy)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_policy(policy_path)
    audit = audit_promotion(ROOT, policy)
    if args.command == "audit":
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if not args.actor.strip() or not args.reason.strip():
        raise SystemExit("promote/rollback requires --actor and --reason")
    changed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.command == "promote":
        if audit["status"] != "passed":
            print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
            raise SystemExit("promotion blocked by incomplete evidence")
        try:
            policy = promote_policy(
                policy,
                audit,
                actor=args.actor,
                reason=args.reason,
                changed_at=changed_at,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        try:
            policy = rollback_policy(
                policy,
                actor=args.actor,
                reason=args.reason,
                changed_at=changed_at,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    write_policy_atomic(policy_path, policy)
    print(json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
