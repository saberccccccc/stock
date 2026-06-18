"""Render training preset commands without launching training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.training_presets import load_training_suite, load_training_suites


def parse_args():
    parser = argparse.ArgumentParser(description="Render train.py commands from JSON presets")
    parser.add_argument("--config", default=None, help="Single config JSON to render.")
    parser.add_argument("--config-dir", default="configs", help="Directory of config JSON files.")
    parser.add_argument("--experiment-id", default=None, help="Only render one experiment id.")
    parser.add_argument("--python-exe", default="python", help="Python executable to print.")
    parser.add_argument("--train-script", default="run/train.py", help="Training script path to print.")
    return parser.parse_args()


def iter_suites(args):
    if args.config:
        return (load_training_suite(args.config),)
    return load_training_suites(args.config_dir)


def main():
    args = parse_args()
    rendered = 0
    for suite in iter_suites(args):
        for experiment in suite.experiments:
            if args.experiment_id and experiment.experiment_id != args.experiment_id:
                continue
            print(f"# {suite.name}:{experiment.experiment_id}", flush=True)
            print(
                experiment.train_command(
                    python_exe=args.python_exe,
                    train_script=args.train_script,
                ),
                flush=True,
            )
            rendered += 1
    if rendered == 0:
        raise SystemExit("No matching training experiments.")


if __name__ == "__main__":
    main()
