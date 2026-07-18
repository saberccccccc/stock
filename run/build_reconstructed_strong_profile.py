"""Build the evidence-hashed reconstructed multi_downside_e19 profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.strong_model_profile import (
    write_hardened_e19_profile_v2,
    write_reconstructed_e19_profile,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or (
        "configs/reconstructed_multi_downside_e19_profile_v1.json"
        if args.version == "v1"
        else "configs/reconstructed_multi_downside_e19_profile_v2.json"
    )
    writer = write_reconstructed_e19_profile if args.version == "v1" else write_hardened_e19_profile_v2
    path = writer(ROOT, ROOT / output)
    print(path, flush=True)


if __name__ == "__main__":
    main()
