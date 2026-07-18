"""Adapt a completed exploratory strong run to the standard rolling artifact shape."""

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

from experiments.recording import (
    MANIFEST_NAME,
    append_event,
    create_experiment,
    finalize_artifact_index,
    load_events,
    record_artifact,
    sha256_file,
)
from experiments.strong_rolling import build_standard_rolling_manifest, load_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args(argv)


def _verify_source(source: Path, progress: dict) -> None:
    if not (source / MANIFEST_NAME).is_file():
        raise FileNotFoundError(source / MANIFEST_NAME)
    events = load_events(source)
    if not events or events[-1].get("status") != "completed":
        raise ValueError("source strong experiment is not completed")
    for state in progress["windows"].values():
        artifacts = [state["alpha"]]
        for stage in state["stages"].values():
            artifacts.extend((stage["exact_checkpoint"], stage["selected_checkpoint"]))
        for artifact in artifacts:
            path = Path(artifact["path"])
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"source artifact hash mismatch: {path}")


def materialize(source_experiment, output_dir, experiment_id):
    source = Path(source_experiment).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    contract_path = source / "staged_pilot_contract.json"
    progress_path = source / "staged_pilot_progress.json"
    contract, progress = load_json(contract_path), load_json(progress_path)
    _verify_source(source, progress)

    source_ref = {
        "schema": "strong_rolling_source_reference_v1",
        "source_experiment": str(source),
        "source_manifest_sha256": sha256_file(source / MANIFEST_NAME),
        "source_contract_sha256": sha256_file(contract_path),
        "source_progress_sha256": sha256_file(progress_path),
        "formal_eligibility": False,
        "reason": "compatibility materialization of an exploratory pilot",
    }
    output.mkdir(parents=True)
    source_ref_path = output / "source_reference.json"
    source_ref_path.write_text(
        json.dumps(source_ref, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    create_experiment(
        output,
        experiment_id=experiment_id,
        config={"source_reference": source_ref, "contract": contract},
        protocol={"selection_allowed": False, "promotion_allowed": False},
        cache_contract={"source_experiment": source_ref},
        project_root=ROOT,
    )
    record_artifact(output, name="source_reference", path=source_ref_path, kind="source_reference")
    manifest_path, manifest = build_standard_rolling_manifest(contract, progress, output)
    for split, item in manifest["split_alpha_paths"].items():
        record_artifact(
            output,
            name=f"stitched_alpha:{split}",
            path=item["path"],
            kind="stitched_dated_alpha",
        )
    record_artifact(output, name="rolling_manifest", path=manifest_path, kind="rolling_manifest")
    append_event(output, status="completed", event_type="strong_rolling_manifest_materialized")
    finalize_artifact_index(output)
    return manifest_path


def main(argv=None):
    args = parse_args(argv)
    path = materialize(args.source_experiment, args.output_dir, args.experiment_id)
    print(json.dumps({"rolling_manifest": str(path), "executed_training": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
