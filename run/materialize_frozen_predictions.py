"""Materialize read-only dated prediction lineage for frozen registry candidates."""

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

from experiments.prediction_artifacts import materialize_frozen_prediction_manifest
from experiments.recording import record_artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", default="registry/candidates.csv")
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workflow-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = materialize_frozen_prediction_manifest(
        project_root=ROOT,
        candidates_csv=args.candidates_csv,
        candidate_ids=args.candidate_id,
        splits=args.split,
        output_dir=args.output_dir,
    )
    record_artifact(
        Path(args.workflow_dir).resolve(),
        name="frozen_dated_prediction_manifest",
        path=path,
        kind="dated_prediction_manifest_v1",
    )
    print(json.dumps({"dated_prediction_manifest": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
