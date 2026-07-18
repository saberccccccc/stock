"""Build predeclared rank blends from a validated chronological OOF lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import load_alpha_rows
from experiments.oof_lineage import load_lineage_manifest
from experiments.recording import (
    MANIFEST_NAME,
    append_event,
    finalize_artifact_index,
    record_artifact,
    sha256_file,
    validate_manifest_for_formal_use,
)
from run.combine_alpha_jsonl import combine_for_date


def _read_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _component_maps(lineage):
    result = {}
    for component in lineage["components"]:
        date_rows = {}
        date_lineage = {}
        for window in component["windows"]:
            rows = load_alpha_rows(window["alpha_path"])
            for row in rows:
                date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
                if date in date_rows:
                    raise ValueError(f"duplicate component date {component['component_id']}:{date}")
                date_rows[date] = row
                date_lineage[date] = {
                    "prediction_date": date,
                    "component_id": component["component_id"],
                    "model_id": window["model_id"],
                    "train_end": window["train_end"],
                    "valid_end": window["valid_end"],
                }
        result[component["component_id"]] = {"rows": date_rows, "lineage": date_lineage}
    return result


def _parse_weights(raw, count):
    if raw is None:
        return [1.0 / count] * count
    values = [float(value) for value in raw]
    if len(values) != count or any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("blend weights must be non-negative and match component count")
    total = sum(values)
    return [value / total for value in values]


def _validate_blend(blend, known_ids):
    name = str(blend.get("name", "")).strip()
    if not name:
        raise ValueError("blend name is required")
    mode = str(blend.get("mode", "rank_mean"))
    if mode != "rank_mean":
        raise ValueError("Phase 4 only permits predeclared rank_mean blends")
    components = blend.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError(f"blend {name} must list components")
    if len(components) != len(set(components)):
        raise ValueError(f"blend {name} contains duplicate components")
    unknown = set(components) - set(known_ids)
    if unknown:
        raise ValueError(f"blend {name} contains unknown components: {sorted(unknown)}")
    weights = _parse_weights(blend.get("weights"), len(components))
    return name, mode, components, weights


def build_blends(lineage_path, spec_path, output_dir):
    lineage_path = Path(lineage_path).expanduser().resolve()
    lineage = load_lineage_manifest(lineage_path)
    spec = _read_json(spec_path)
    blends = spec.get("blends")
    if not isinstance(blends, list) or not blends:
        raise ValueError("blend spec must contain at least one blend")
    maps = _component_maps(lineage)
    known_ids = list(maps)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_dir = output_root.parent
    if not (experiment_dir / MANIFEST_NAME).is_file():
        raise FileNotFoundError(f"OOF experiment manifest is missing: {experiment_dir / MANIFEST_NAME}")
    # The lineage and blends are two stages of one immutable OOF experiment.
    # Scope is validated here; terminal artifacts are sealed after blend output.
    validate_manifest_for_formal_use(experiment_dir / MANIFEST_NAME, require_artifacts=False)
    append_event(
        experiment_dir,
        status="running",
        event_type="oof_blend_build_started",
        details={"lineage_manifest": str(lineage_path), "blend_spec": str(Path(spec_path).resolve())},
    )
    record_artifact(experiment_dir, name="blend_spec", path=spec_path, kind="oof_blend_spec")
    output_entries = []
    for blend in blends:
        name, mode, component_ids, weights = _validate_blend(blend, known_ids)
        date_sets = [set(maps[component_id]["rows"]) for component_id in component_ids]
        common_dates = sorted(set.intersection(*date_sets))
        if not common_dates:
            raise ValueError(f"blend {name} has no common dates")
        output_path = output_root / f"{name}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for date in common_dates:
                rows = [maps[component_id]["rows"][date] for component_id in component_ids]
                combined = combine_for_date(rows, weights, mode)
                row = {
                    "date": date,
                    **combined,
                    "blend_name": name,
                    "blend_mode": mode,
                    "blend_components": component_ids,
                    "blend_weights": weights,
                    "oof_lineage": [maps[component_id]["lineage"][date] for component_id in component_ids],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_entries.append(
            {
                "name": name,
                "mode": mode,
                "components": component_ids,
                "weights": weights,
                "signal_start": common_dates[0],
                "signal_end": common_dates[-1],
                "prediction_count": len(common_dates),
                "alpha_path": str(output_path),
                "alpha_sha256": sha256_file(output_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "type": "oof_blend_manifest",
        "source_lineage_manifest": str(lineage_path),
        "source_lineage_sha256": sha256_file(lineage_path),
        "research_end": lineage["research_end"],
        "selection_splits": lineage.get("selection_splits", []),
        "blends": output_entries,
    }
    manifest_path = output_root / "oof_blend_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for entry in output_entries:
        record_artifact(experiment_dir, name=f"blend:{entry['name']}", path=entry["alpha_path"], kind="oof_blend_alpha")
    record_artifact(experiment_dir, name="oof_blend_manifest", path=manifest_path, kind="oof_blend_manifest")
    append_event(
        experiment_dir,
        status="completed",
        event_type="oof_blends_completed",
        details={"blend_count": len(output_entries), "manifest": str(manifest_path)},
    )
    finalize_artifact_index(experiment_dir)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--blend-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    manifest = build_blends(args.lineage_manifest, args.blend_spec, args.output_dir)
    print(f"Saved {len(manifest['blends'])} OOF blends to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
