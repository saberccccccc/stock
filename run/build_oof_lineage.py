"""Build a validated chronological OOF lineage manifest from rolling runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.oof_lineage import build_lineage_manifest
from core.research_protocol import SPLIT_SPECS
from experiments.recording import (
    canonical_json_hash,
    create_experiment,
    declared_range,
    not_applicable_range,
    record_artifact,
    sha256_file,
)


def parse_components(raw: str) -> list[tuple[str, str]]:
    components = []
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            raise ValueError("components must use id=rolling_manifest syntax")
        component_id, path = item.split("=", 1)
        component_id, path = component_id.strip(), path.strip()
        if not component_id or not path:
            raise ValueError("component id and manifest path are required")
        components.append((component_id, path))
    return components


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", required=True, help="Comma-separated id=rolling_manifest paths")
    parser.add_argument("--output", required=True, help="Output lineage manifest JSON")
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args(argv)
    components = parse_components(args.components)
    payload = build_lineage_manifest(components, output_path=args.output)
    output_path = Path(args.output).resolve()
    experiment_dir = output_path.parent
    windows = [
        window
        for component in payload["components"]
        for window in component["windows"]
    ]
    signal_start = min(window["signal_start"] for window in windows)
    signal_end = max(window["signal_end"] for window in windows)
    split_roles = []
    for split, spec in SPLIT_SPECS.items():
        if signal_start <= str(spec.end.date()) and signal_end >= str(spec.start.date()):
            split_roles.append(
                {
                    "split": split,
                    "selection_eligible": spec.selection_eligible,
                    "forward_used": spec.is_forward,
                }
            )
    if not split_roles:
        split_roles = [
            {
                "split": "historical_oos",
                "role": "research_oos_preselection",
                "selection_eligible": False,
                "forward_used": False,
            }
        ]
    scope = {
        "stage": "oof_lineage",
        "data_sources": [
            {
                "role": f"rolling_component:{component_id}",
                "root": str(Path(path).resolve()),
                "fingerprint": {"kind": "file_sha256", "value": sha256_file(path)},
            }
            for component_id, path in components
        ],
        "ranges": {
            "feature_warmup": not_applicable_range("owned by parent rolling experiments"),
            "train": declared_range(
                min(window["train_start"] for window in windows),
                max(window["train_end"] for window in windows),
            ),
            "valid": declared_range(
                min(window["valid_start"] for window in windows),
                max(window["valid_end"] for window in windows),
            ),
            "signal": declared_range(signal_start, signal_end),
            "backtest": not_applicable_range("lineage stage emits no portfolio result"),
        },
        "max_data_date": signal_end,
        "split_roles": split_roles,
        "transform": {
            "state_sha256": canonical_json_hash(payload),
            "fit_range": not_applicable_range("owned by parent rolling experiments"),
        },
        "lineage": {},
    }
    create_experiment(
        experiment_dir,
        experiment_id=args.experiment_id,
        config={
            "name": args.experiment_id,
            "components": [{"component_id": component_id, "rolling_manifest": str(Path(path).resolve())} for component_id, path in components],
            "lineage_manifest": str(output_path),
        },
        protocol={
            "type": "chronological_oof_lineage",
            "research_end": payload["research_end"],
            "selection_splits": payload["selection_splits"],
            "forward_is_observation_only": True,
        },
        cache_contract={"research_end": payload["research_end"], "label_family": "oo_lag1"},
        project_root=ROOT,
        formal=True,
        experiment_scope=scope,
    )
    record_artifact(experiment_dir, name="lineage_manifest", path=output_path, kind="oof_lineage_manifest")
    print(
        f"Saved OOF lineage: components={len(payload['components'])} "
        f"research_end={payload['research_end']} output={output_path}"
    )


if __name__ == "__main__":
    main()
