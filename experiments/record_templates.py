"""Dependent, immutable evidence records for one governed experiment."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator

from experiments.recording import MANIFEST_NAME, canonical_json_hash, record_artifact, sha256_file


STANDARD_RECORD_ORDER = (
    "signal",
    "signal_analysis",
    "portfolio",
    "risk_attribution",
    "stress",
    "decision",
)
STANDARD_RECORD_DEPENDENCIES = {
    "signal": (),
    "signal_analysis": ("signal",),
    "portfolio": ("signal",),
    "risk_attribution": ("portfolio",),
    "stress": ("portfolio",),
    "decision": ("signal_analysis", "risk_attribution", "stress"),
}
STANDARD_REQUIRED_FIELDS = {
    "signal": ("schema", "signal_start", "signal_end", "rows", "asof_start", "asof_end"),
    "signal_analysis": ("selection_splits", "observation_splits", "metrics"),
    "portfolio": ("execution_adapter", "fill_price", "backtest_start", "backtest_end", "metrics"),
    "risk_attribution": ("metrics",),
    "stress": ("capitals", "stresses", "cells"),
    "decision": (
        "selection_splits",
        "observation_splits",
        "selection_result",
        "forward_observation",
        "forward_used_for_selection",
    ),
}
STANDARD_REQUIRED_ARTIFACTS = {
    "signal": ("prediction", "label"),
    "signal_analysis": ("signal_metrics",),
    "portfolio": ("equity_curve", "positions", "orders", "rejections", "costs"),
    "risk_attribution": ("risk_metrics",),
    "stress": ("stress_scorecard",),
    "decision": ("decision_report",),
}
RECORD_SPEC_SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "record_bundle_spec_v1.schema.json"


@dataclass(frozen=True)
class RecordContext:
    experiment_dir: Path
    workflow: Mapping[str, Any]
    selection_splits: tuple[str, ...]
    observation_splits: tuple[str, ...]

    @classmethod
    def from_workflow(cls, experiment_dir: str | Path, workflow: Mapping[str, Any]) -> "RecordContext":
        evaluation = workflow.get("evaluation", {})
        governance = workflow.get("governance", {})
        selection = evaluation.get("selection_splits", governance.get("selection_splits", ()))
        observation = evaluation.get("observation_splits", governance.get("observation_splits", ()))
        return cls(
            experiment_dir=Path(experiment_dir).resolve(),
            workflow=dict(workflow),
            selection_splits=tuple(map(str, selection)),
            observation_splits=tuple(map(str, observation)),
        )


@dataclass(frozen=True)
class RecordBuild:
    payload: Mapping[str, Any]
    artifacts: Mapping[str, str | Path]


@dataclass(frozen=True)
class RecordResult:
    name: str
    directory: Path
    manifest_path: Path
    payload: Mapping[str, Any]
    artifacts: Mapping[str, Mapping[str, Any]]


RecordProducer = Callable[[RecordContext, Mapping[str, RecordResult], Path], RecordBuild]


@dataclass(frozen=True)
class RecordTemplate:
    name: str
    depends_on: tuple[str, ...]
    producer: RecordProducer

    def __post_init__(self) -> None:
        if self.name not in STANDARD_RECORD_ORDER:
            raise ValueError(f"unknown standard record: {self.name}")
        expected = STANDARD_RECORD_DEPENDENCIES[self.name]
        if tuple(self.depends_on) != expected:
            raise ValueError(f"record {self.name} dependencies must be {expected}")


def standard_record_templates(producers: Mapping[str, RecordProducer]) -> list[RecordTemplate]:
    missing = set(STANDARD_RECORD_ORDER) - set(producers)
    extra = set(producers) - set(STANDARD_RECORD_ORDER)
    if missing or extra:
        raise ValueError(f"record producer mismatch missing={sorted(missing)} extra={sorted(extra)}")
    return [
        RecordTemplate(name, STANDARD_RECORD_DEPENDENCIES[name], producers[name])
        for name in STANDARD_RECORD_ORDER
    ]


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _validate_iso_range(payload: Mapping[str, Any], start_field: str, end_field: str) -> None:
    start = str(payload[start_field])
    end = str(payload[end_field])
    if len(start) < 10 or len(end) < 10 or start[:10] > end[:10]:
        raise ValueError(f"record range {start_field}/{end_field} is invalid")


def validate_standard_record(
    name: str,
    payload: Mapping[str, Any],
    artifacts: Mapping[str, str | Path],
    context: RecordContext,
) -> None:
    payload = _require_mapping(payload, f"record {name}.payload")
    missing_fields = [field for field in STANDARD_REQUIRED_FIELDS[name] if field not in payload]
    if missing_fields:
        raise ValueError(f"record {name} missing fields: {missing_fields}")
    missing_artifacts = [field for field in STANDARD_REQUIRED_ARTIFACTS[name] if field not in artifacts]
    if missing_artifacts:
        raise ValueError(f"record {name} missing artifacts: {missing_artifacts}")

    if name == "signal":
        if payload["schema"] != "prediction_frame_v1" or int(payload["rows"]) <= 0:
            raise ValueError("signal record requires non-empty prediction_frame_v1")
        _validate_iso_range(payload, "signal_start", "signal_end")
        _validate_iso_range(payload, "asof_start", "asof_end")
    elif name == "signal_analysis":
        _require_mapping(payload["metrics"], "signal_analysis.metrics")
    elif name == "portfolio":
        if payload["execution_adapter"] != "official_open_ledger" or payload["fill_price"] != "open":
            raise ValueError("portfolio record must delegate to official open-price ledger")
        _validate_iso_range(payload, "backtest_start", "backtest_end")
        _require_mapping(payload["metrics"], "portfolio.metrics")
    elif name == "risk_attribution":
        _require_mapping(payload["metrics"], "risk_attribution.metrics")
    elif name == "stress":
        if set(map(int, payload["capitals"])) != {500_000, 1_000_000}:
            raise ValueError("stress record must cover 50w and 100w")
        if set(map(str, payload["stresses"])) != {"normal", "lag1", "cost2x", "capacity_3pct"}:
            raise ValueError("stress record must cover the four formal stress scenarios")
        if int(payload["cells"]) < 8:
            raise ValueError("stress record must contain at least 8 capital/stress cells")
    elif name == "decision":
        if bool(payload["forward_used_for_selection"]):
            raise ValueError("Forward observation cannot participate in selection")
        if tuple(map(str, payload["selection_splits"])) != context.selection_splits:
            raise ValueError("decision selection_splits disagree with Workflow")
        if tuple(map(str, payload["observation_splits"])) != context.observation_splits:
            raise ValueError("decision observation_splits disagree with Workflow")
        _require_mapping(payload["selection_result"], "decision.selection_result")
        _require_mapping(payload["forward_observation"], "decision.forward_observation")


def _artifact_entry(source: Path, final_path: Path | None) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(final_path or source.resolve()),
        "sha256": sha256_file(source),
        "bytes": int(source.stat().st_size),
    }


class RecordBundleRunner:
    """Generate a complete record chain without embedding analytics or execution."""

    def __init__(self, context: RecordContext, templates: Sequence[RecordTemplate]):
        self.context = context
        self.templates = {template.name: template for template in templates}
        if tuple(self.templates) != STANDARD_RECORD_ORDER:
            raise ValueError("formal record bundle must declare all six standard records in canonical order")
        self.records_dir = context.experiment_dir / "records"
        self.results: dict[str, RecordResult] = {}

    def run_record(self, name: str) -> RecordResult:
        if name not in self.templates:
            raise ValueError(f"unknown record template: {name}")
        template = self.templates[name]
        missing = [parent for parent in template.depends_on if parent not in self.results]
        if missing:
            raise RuntimeError(f"record {name} missing parent records: {missing}")
        target = self.records_dir / name
        if target.exists():
            raise FileExistsError(f"immutable record already exists: {target}")
        self.records_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.records_dir / f".{name}.building"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            build = template.producer(self.context, dict(self.results), temporary)
            validate_standard_record(name, build.payload, build.artifacts, self.context)
            artifact_entries = {}
            for artifact_name, raw_path in build.artifacts.items():
                source = Path(raw_path).resolve()
                try:
                    relative = source.relative_to(temporary.resolve())
                except ValueError:
                    final_path = None
                else:
                    final_path = target / relative
                artifact_entries[str(artifact_name)] = _artifact_entry(source, final_path)
            dependencies = {
                parent: {
                    "manifest_path": str(self.results[parent].manifest_path),
                    "manifest_sha256": sha256_file(self.results[parent].manifest_path),
                }
                for parent in template.depends_on
            }
            manifest = {
                "schema": "standard_record_v1",
                "name": name,
                "depends_on": dependencies,
                "payload": dict(build.payload),
                "artifacts": artifact_entries,
            }
            manifest["record_sha256"] = canonical_json_hash(manifest)
            manifest_path = temporary / "record_manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

        final_manifest = target / "record_manifest.json"
        result = RecordResult(name, target, final_manifest, dict(build.payload), artifact_entries)
        self.results[name] = result
        if (self.context.experiment_dir / MANIFEST_NAME).is_file():
            record_artifact(
                self.context.experiment_dir,
                name=f"record:{name}",
                path=final_manifest,
                kind="standard_record_manifest_v1",
            )
        return result

    def run_all(self) -> Path:
        for name in STANDARD_RECORD_ORDER:
            self.run_record(name)
        records = {
            name: {
                "manifest_path": str(result.manifest_path),
                "manifest_sha256": sha256_file(result.manifest_path),
            }
            for name, result in self.results.items()
        }
        bundle = {
            "schema": "standard_record_bundle_v1",
            "selection_splits": list(self.context.selection_splits),
            "observation_splits": list(self.context.observation_splits),
            "records": records,
        }
        bundle["bundle_sha256"] = canonical_json_hash(bundle)
        target = self.records_dir / "bundle_manifest.json"
        target.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if (self.context.experiment_dir / MANIFEST_NAME).is_file():
            record_artifact(
                self.context.experiment_dir,
                name="record_bundle",
                path=target,
                kind="standard_record_bundle_v1",
            )
        return target


def materialize_record_spec(spec_path: str | Path, experiment_dir: str | Path) -> Path:
    """Materialize external analytics/ledger outputs through the standard chain."""

    source = Path(spec_path).resolve()
    spec = json.loads(source.read_text(encoding="utf-8-sig"))
    schema = json.loads(RECORD_SPEC_SCHEMA.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(spec), key=lambda item: list(item.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "<root>"
        raise ValueError(f"record bundle spec error at {location}: {error.message}")

    producers = {}
    for name in STANDARD_RECORD_ORDER:
        record_spec = spec["records"][name]
        payload = dict(record_spec["payload"])
        artifacts = {
            artifact_name: (
                Path(raw_path).resolve()
                if Path(raw_path).is_absolute()
                else (source.parent / raw_path).resolve()
            )
            for artifact_name, raw_path in record_spec["artifacts"].items()
        }

        def producer(context, parents, output_dir, *, _payload=payload, _artifacts=artifacts):
            return RecordBuild(payload=_payload, artifacts=_artifacts)

        producers[name] = producer

    context = RecordContext.from_workflow(experiment_dir, spec["workflow"])
    runner = RecordBundleRunner(context, standard_record_templates(producers))
    return runner.run_all()
