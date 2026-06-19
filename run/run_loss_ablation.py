"""Run registered V9 loss ablations sequentially."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from core.training_presets import load_training_suite


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run loss ablation experiments")
    parser.add_argument(
        "--config",
        default="configs/loss_ablation_20260613.json",
    )
    parser.add_argument("--start", default=None, help="First experiment ID to run")
    parser.add_argument("--only", default=None, help="Run one experiment ID")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-pid-file", default=None)
    return parser.parse_args()


def build_command(experiment):
    """Build train.py argv from the validated shared preset representation."""

    return [sys.executable, "run/train.py", *experiment.train_argv()]


def main():
    args = parse_args()
    if args.wait_pid_file:
        pid_path = ROOT / args.wait_pid_file
        if pid_path.exists():
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            print(f"Waiting for PID {pid} from {pid_path}", flush=True)
            while True:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    break
                time.sleep(30)
            print(f"PID {pid} finished; starting loss ablations", flush=True)
    config_path = ROOT / args.config
    suite = load_training_suite(config_path)
    experiments = list(suite.experiments)
    if args.only:
        experiments = [exp for exp in experiments if exp.experiment_id == args.only]
    elif args.start:
        ids = [exp.experiment_id for exp in experiments]
        if args.start not in ids:
            raise ValueError(f"Unknown experiment ID: {args.start}")
        experiments = experiments[ids.index(args.start):]
    if not experiments:
        raise ValueError("No experiments selected")

    for experiment in experiments:
        output_dir = ROOT / experiment.output_dir
        metrics = output_dir / "epochs" / "epoch_metrics.jsonl"
        if metrics.exists():
            print(
                f"SKIP {experiment.experiment_id}: metrics already exist at {metrics}",
                flush=True,
            )
            continue
        command = build_command(experiment)
        print(f"RUN {experiment.experiment_id}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(
                f"Experiment {experiment.experiment_id} failed with exit code {result.returncode}"
            )


if __name__ == "__main__":
    main()
