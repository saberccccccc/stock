"""Compile and freeze one declarative experiment workflow without executing it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.recording import (
    append_event,
    create_experiment,
    finalize_artifact_index,
    record_artifact,
)
from experiments.workflow import compile_workflow, load_workflow_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"workflow output directory is not empty: {output_dir}")
    config = load_workflow_config(config_path)
    compiled = compile_workflow(
        config,
        project_root=ROOT,
        output_dir=output_dir,
        python=args.python,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_config = output_dir / "workflow_config.json"
    compiled_path = output_dir / "compiled_workflow.json"
    frozen_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compiled_path.write_text(json.dumps(compiled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evaluation = config["evaluation"]
    splits = evaluation.get("splits")
    if splits is None:
        splits = list(evaluation.get("selection_splits", [])) + list(evaluation.get("observation_splits", []))
    manifest_path = create_experiment(
        output_dir,
        experiment_id=config["experiment_id"],
        config=config,
        protocol={
            "type": f"declarative_workflow_v{config['schema_version']}",
            "splits": splits,
        },
        cache_contract={"compiled_workflow": str(compiled_path)},
        project_root=ROOT,
        formal=True,
        experiment_scope=compiled["scope"],
    )
    record_artifact(output_dir, name="workflow_source_config", path=config_path, kind="workflow_source")
    record_artifact(output_dir, name="workflow_config", path=frozen_config, kind="frozen_workflow_config")
    record_artifact(output_dir, name="compiled_workflow", path=compiled_path, kind="compiled_workflow")
    append_event(output_dir, status="completed", event_type="workflow_compiled")
    index_path = finalize_artifact_index(output_dir)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "compiled_workflow": str(compiled_path),
                "artifact_index": str(index_path),
                "executed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
