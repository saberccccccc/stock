"""Generate a read-only v14 transform/PIT contract audit from cache metadata."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.transform_contract import build_v14_transform_contract, write_transform_contract


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True, help="v14 cache metadata pickle")
    parser.add_argument("--output", required=True, help="new JSON contract output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    meta_path = Path(args.meta)
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    contract = build_v14_transform_contract(meta, meta_path=meta_path)
    output = write_transform_contract(contract, args.output)
    print(json.dumps({"output": str(output), "status": contract["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
