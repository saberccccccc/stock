"""Validation and provenance helpers for chronological OOF alpha components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from alpha.io import load_alpha_rows
from core.research_protocol import RESEARCH_END_DATE
from experiments.recording import sha256_file, validate_manifest_for_formal_use


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _as_date(value: Any, field: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not a valid date: {value!r}") from exc


def _date_text(value: Any) -> str:
    return _as_date(value, "date").strftime("%Y-%m-%d")


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    relative_to_manifest = (base_dir / path).resolve()
    if relative_to_manifest.is_file():
        return relative_to_manifest
    relative_to_project = (PROJECT_ROOT / path).resolve()
    if relative_to_project.is_file():
        return relative_to_project
    return relative_to_manifest


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _formal_evidence(manifest_path: Path, manifest: Mapping[str, Any]) -> str:
    """Classify executed rolling evidence without overstating governance status."""

    if manifest.get("run_mode") == "formal":
        experiment_manifest_path = manifest_path.parent / "experiment_manifest.json"
        if experiment_manifest_path.is_file():
            experiment_manifest = _read_json(experiment_manifest_path)
            if experiment_manifest.get("experiment_class") == "formal":
                validate_manifest_for_formal_use(experiment_manifest_path)
                return "governed_formal_experiment"
        # Rolling CLIs historically used "formal" to mean an executed run as
        # opposed to dry-run. It does not imply Registry/governance acceptance.
        return "executed_rolling_manifest"
    experiment_manifest_path = manifest_path.parent / "experiment_manifest.json"
    events_path = manifest_path.parent / "events.jsonl"
    artifact_index_path = manifest_path.parent / "artifact_index.json"
    if not (
        experiment_manifest_path.is_file()
        and events_path.is_file()
        and artifact_index_path.is_file()
    ):
        raise ValueError(f"OOF component must have formal rolling evidence: {manifest_path}")
    experiment_manifest = _read_json(experiment_manifest_path)
    protocol = experiment_manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"formal experiment protocol is missing: {experiment_manifest_path}")
    completed = False
    for line in events_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") == "rolling_training_completed" and event.get("status") == "completed":
            completed = True
    if not completed:
        raise ValueError(f"formal rolling completion event is missing: {events_path}")
    artifact_index = _read_json(artifact_index_path)
    artifacts = artifact_index.get("artifacts", [])
    if not any(item.get("name") == "rolling_manifest" for item in artifacts if isinstance(item, dict)):
        raise ValueError(f"formal rolling artifact record is missing: {artifact_index_path}")
    return "legacy_completed_experiment_record"


def _window_by_name(windows: list[Mapping[str, Any]], name: str, source: Path) -> Mapping[str, Any]:
    matches = [window for window in windows if str(window.get("name")) == name]
    if len(matches) != 1:
        raise ValueError(f"manifest {source} must contain exactly one window named {name}")
    return matches[0]


def _validate_window_dates(window: Mapping[str, Any], component_id: str) -> dict[str, str]:
    fields = ("train_start", "train_end", "valid_start", "valid_end", "predict_start", "predict_end")
    dates = {field: _date_text(window.get(field)) for field in fields}
    ordered = [_as_date(dates[field], field) for field in fields]
    if not (ordered[0] <= ordered[1] < ordered[2] <= ordered[3] < ordered[4] <= ordered[5]):
        raise ValueError(f"{component_id}:{window.get('name')} has overlapping or unordered dates")
    return dates


def inspect_component(component_id: str, rolling_manifest_path: str | Path) -> dict[str, Any]:
    """Validate one formal rolling experiment and return serializable lineage."""

    if not component_id or component_id.strip() != component_id:
        raise ValueError("component_id must be a non-empty trimmed string")
    manifest_path = Path(rolling_manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_path)
    formal_evidence = _formal_evidence(manifest_path, manifest)

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"rolling manifest config is missing: {manifest_path}")
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"rolling manifest data contract is missing: {manifest_path}")
    # Legacy manifests used research_end for the readable cache ceiling. That
    # is not model lineage. Leakage is determined from each window's actual
    # train/valid/predict dates below.
    research_end = _as_date(data.get("research_end"), "research_end")
    label_family = str(data.get("label_family", ""))
    horizon_index = int(data.get("horizon_index"))
    configured_windows = config.get("windows")
    artifact_windows = manifest.get("windows")
    if not isinstance(configured_windows, list) or not isinstance(artifact_windows, list):
        raise ValueError(f"rolling manifest windows are missing: {manifest_path}")

    windows: list[dict[str, Any]] = []
    for configured in configured_windows:
        name = str(configured.get("name", ""))
        if not name:
            raise ValueError(f"window name is missing in {manifest_path}")
        dates = _validate_window_dates(configured, component_id)
        if _as_date(dates["predict_end"], "predict_end") > RESEARCH_END_DATE:
            raise ValueError(
                f"{component_id}:{name} predict_end {dates['predict_end']} exceeds "
                f"selection boundary {RESEARCH_END_DATE.date()}"
            )
        artifact = _window_by_name(artifact_windows, name, manifest_path)
        alpha_value = artifact.get("alpha_path")
        model_value = artifact.get("model_path")
        if not alpha_value or not model_value:
            raise ValueError(f"{component_id}:{name} is missing alpha/model artifact")
        alpha_path = _resolve_path(alpha_value, manifest_path.parent)
        model_path = _resolve_path(model_value, manifest_path.parent)
        if not alpha_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"{component_id}:{name} alpha/model artifact is missing")

        rows = load_alpha_rows(alpha_path)
        if not rows:
            raise ValueError(f"{component_id}:{name} alpha artifact is empty")
        prediction_dates = [_date_text(row["date"]) for row in rows]
        min_prediction = _as_date(prediction_dates[0], "prediction date")
        max_prediction = _as_date(prediction_dates[-1], "prediction date")
        if min_prediction < _as_date(dates["predict_start"], "predict_start"):
            raise ValueError(f"{component_id}:{name} predicts before its configured OOS start")
        if max_prediction > _as_date(dates["predict_end"], "predict_end"):
            raise ValueError(f"{component_id}:{name} predicts after its configured OOS end")
        if _as_date(dates["train_end"], "train_end") >= min_prediction:
            raise ValueError(f"{component_id}:{name} train_end is not before prediction dates")
        if _as_date(dates["valid_end"], "valid_end") >= min_prediction:
            raise ValueError(f"{component_id}:{name} valid_end is not before prediction dates")

        expected_count = artifact.get("counts", {}).get("predict")
        if expected_count is not None and int(expected_count) != len(rows):
            raise ValueError(
                f"{component_id}:{name} alpha row count {len(rows)} != manifest predict count {expected_count}"
            )
        prediction_lineage = [
            {
                "prediction_date": date,
                "component_id": component_id,
                "model_id": f"{component_id}:{name}",
                "train_end": dates["train_end"],
                "valid_end": dates["valid_end"],
            }
            for date in prediction_dates
        ]
        windows.append(
            {
                "name": name,
                **dates,
                "model_id": f"{component_id}:{name}",
                "alpha_path": str(alpha_path),
                "alpha_sha256": sha256_file(alpha_path),
                "model_path": str(model_path),
                "model_sha256": sha256_file(model_path),
                "signal_start": prediction_dates[0],
                "signal_end": prediction_dates[-1],
                "prediction_count": len(rows),
                "prediction_lineage": prediction_lineage,
            }
        )

    return {
        "component_id": component_id,
        "formal_evidence": formal_evidence,
        "rolling_manifest": str(manifest_path),
        "rolling_manifest_sha256": sha256_file(manifest_path),
        "feature_set": str(manifest.get("feature_set", "")),
        "label_family": label_family,
        "horizon_index": horizon_index,
        "label_end_offset": int(manifest.get("label_end_offset")),
        "research_end": research_end.strftime("%Y-%m-%d"),
        "windows": windows,
    }


def build_lineage_manifest(
    components: list[tuple[str, str | Path]],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate components and write a frozen OOF lineage manifest."""

    if not components:
        raise ValueError("at least one OOF component is required")
    component_ids = [component_id for component_id, _ in components]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("OOF component ids must be unique")
    entries = [inspect_component(component_id, path) for component_id, path in components]
    reference = entries[0]
    for entry in entries[1:]:
        for field in ("research_end", "label_family", "horizon_index", "label_end_offset"):
            if entry[field] != reference[field]:
                raise ValueError(f"OOF component contract mismatch for {field}")
        reference_windows = {window["name"]: window for window in reference["windows"]}
        for window in entry["windows"]:
            ref = reference_windows.get(window["name"])
            if ref is None:
                raise ValueError(f"OOF component window mismatch: {window['name']}")
            for field in ("train_end", "valid_end", "predict_start", "predict_end"):
                if window[field] != ref[field]:
                    raise ValueError(f"OOF component window {window['name']} mismatch for {field}")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "type": "chronological_oof_lineage",
        "research_end": reference["research_end"],
        "selection_splits": ["val_2024", "test_2025"],
        "components": entries,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def load_lineage_manifest(path: str | Path) -> dict[str, Any]:
    payload = _read_json(Path(path).expanduser().resolve())
    if payload.get("type") != "chronological_oof_lineage":
        raise ValueError("invalid OOF lineage manifest type")
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("OOF lineage manifest has no components")
    return payload
