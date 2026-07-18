"""Materialize one immutable Qlib-aligned project-native record bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.record_templates import materialize_record_spec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--experiment-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bundle = materialize_record_spec(args.spec, args.experiment_dir)
    print(bundle, flush=True)


if __name__ == "__main__":
    main()
