"""Create and manually operate a project-native Shadow lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
root_path = str(ROOT)
if root_path in sys.path:
    sys.path.remove(root_path)
sys.path.insert(0, root_path)

from experiments.shadow_lifecycle import (
    create_shadow_lifecycle,
    record_shadow_observation,
    transition_shadow_lifecycle,
    validate_shadow_lifecycle,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--lifecycle-dir", required=True)
    create.add_argument("--lifecycle-id", required=True)
    create.add_argument("--candidate-id", required=True)
    create.add_argument("--workflow-dir", required=True)
    create.add_argument("--actor", required=True)
    create.add_argument("--reason", required=True)
    transition = sub.add_parser("transition")
    transition.add_argument("--lifecycle-dir", required=True)
    transition.add_argument("--target-state", required=True, choices=("shadow", "paused", "retired"))
    transition.add_argument("--actor", required=True)
    transition.add_argument("--reason", required=True)
    transition.add_argument("--manual-approval", action="store_true")
    observe = sub.add_parser("observe")
    observe.add_argument("--lifecycle-dir", required=True)
    observe.add_argument("--date", required=True)
    observe.add_argument("--artifact", action="append", default=[], help="name=path")
    observe.add_argument("--actor", required=True)
    observe.add_argument("--reason", required=True)
    status = sub.add_parser("status")
    status.add_argument("--lifecycle-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "create":
        result = create_shadow_lifecycle(
            args.lifecycle_dir,
            lifecycle_id=args.lifecycle_id,
            candidate_id=args.candidate_id,
            workflow_dir=args.workflow_dir,
            actor=args.actor,
            reason=args.reason,
        )
    elif args.command == "transition":
        result = transition_shadow_lifecycle(
            args.lifecycle_dir,
            target_state=args.target_state,
            actor=args.actor,
            reason=args.reason,
            manual_approval=args.manual_approval,
        )
    elif args.command == "observe":
        artifacts = dict(item.split("=", 1) for item in args.artifact)
        result = record_shadow_observation(
            args.lifecycle_dir,
            observation_date=args.date,
            artifacts=artifacts,
            actor=args.actor,
            reason=args.reason,
        )
    else:
        snapshot = validate_shadow_lifecycle(args.lifecycle_dir)
        print(json.dumps(snapshot["state"], ensure_ascii=False, indent=2))
        return
    print(result, flush=True)


if __name__ == "__main__":
    main()
