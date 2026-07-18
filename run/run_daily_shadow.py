"""Run or validate the project-native deterministic daily Shadow package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shadow_daily import replay_daily_shadow_run, run_daily_shadow, validate_daily_shadow_run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--lifecycle-dir", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--alpha-path", required=True)
    run.add_argument("--candidate-id", required=True)
    run.add_argument("--data-dir", default="data/forward_raw")
    run.add_argument("--start-date", required=True)
    run.add_argument("--end-date", required=True)
    run.add_argument("--max-data-date", required=True)
    run.add_argument("--portfolio-value", type=float, choices=(500000.0, 1000000.0), required=True)
    run.add_argument(
        "--mode",
        choices=("historical_replay", "shadow_observation"),
        default="historical_replay",
    )
    run.add_argument("--actor", default="")
    run.add_argument("--reason", default="")
    validate = sub.add_parser("validate")
    validate.add_argument("--run-dir", required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--source-run-dir", required=True)
    replay.add_argument("--replay-run-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "validate":
        manifest = validate_daily_shadow_run(args.run_dir)
        print(json.dumps({"status": "valid", "daily_count": manifest["daily_count"], "semantic_sha256": manifest["semantic_sha256"]}, ensure_ascii=False, indent=2))
        return
    if args.command == "replay":
        manifest = replay_daily_shadow_run(
            args.source_run_dir,
            args.replay_run_dir,
            project_root=ROOT,
        )
        print(manifest, flush=True)
        return
    manifest = run_daily_shadow(
        project_root=ROOT,
        lifecycle_dir=args.lifecycle_dir,
        run_dir=args.run_dir,
        alpha_path=args.alpha_path,
        candidate_id=args.candidate_id,
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        max_data_date=args.max_data_date,
        portfolio_value=args.portfolio_value,
        mode=args.mode,
        actor=args.actor,
        reason=args.reason,
    )
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
