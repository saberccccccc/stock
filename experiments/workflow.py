"""Versioned declarative workflow contract for project-native experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from core.research_protocol import SPLIT_SPECS, assert_forward_parent_frozen, get_split_spec
from backtest.market_data_contract import ExecutionMarketDataContract
from experiments.recording import canonical_json_hash, declared_range, fingerprint_path
from experiments.workflow_schema import normalize_workflow_v2, validate_workflow_v2


WORKFLOW_SCHEMA_VERSION = 1
REQUIRED_SECTIONS = (
    "data",
    "features",
    "labels",
    "windows",
    "model",
    "checkpoint",
    "alpha",
    "strategy",
    "ledger",
    "evaluation",
    "reports",
)
REQUIRED_RANGES = ("feature_warmup", "train", "valid", "signal", "backtest")
ALLOWED_MODEL_ADAPTERS = {
    "frozen_registry_candidate",
    "rolling_lgbm_alpha",
    "torch_strong_alpha",
}
ROLLING_MODEL_ADAPTERS = {"rolling_lgbm_alpha", "torch_strong_alpha"}
ALLOWED_STRATEGY_ADAPTERS = {"retention", "topk_dropout"}
REQUIRED_SELECTION_STRESSES = {"normal", "lag1", "cost2x", "capacity_3pct"}


def load_workflow_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("workflow config must be a JSON object")
    return value


def _required_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"workflow section {name!r} must be an object")
    return value


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _range(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"workflow ranges.{field} must be an object")
    result = declared_range(value.get("start"), value.get("end"))
    if result["start"] in {"None", ""} or result["end"] in {"None", ""}:
        raise ValueError(f"workflow ranges.{field} requires start and end")
    return result


def _validate_workflow_v1(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise ValueError(f"workflow schema_version must be {WORKFLOW_SCHEMA_VERSION}")
    experiment_id = str(config.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ValueError("workflow experiment_id is required")
    missing = [name for name in REQUIRED_SECTIONS if name not in config]
    if missing:
        raise ValueError(f"workflow sections are missing: {missing}")
    for name in REQUIRED_SECTIONS:
        _required_mapping(config, name)

    data = config["data"]
    if not isinstance(data.get("sources"), list) or not data["sources"]:
        raise ValueError("workflow data.sources must contain at least one source")
    for source in data["sources"]:
        if not isinstance(source, Mapping) or not source.get("role") or not source.get("root"):
            raise ValueError("each workflow data source requires role and root")

    features = config["features"]
    if not str(features.get("set", "")).strip():
        raise ValueError("workflow features.set is required")
    labels = config["labels"]
    for field in ("family", "horizon_index", "tail_purge_days"):
        if labels.get(field) in (None, ""):
            raise ValueError(f"workflow labels.{field} is required")
    if int(labels["tail_purge_days"]) <= 0:
        raise ValueError("workflow labels.tail_purge_days must be positive")
    if not str(config["checkpoint"].get("rule", "")).strip():
        raise ValueError("workflow checkpoint.rule is required")

    ranges = _required_mapping(config, "ranges")
    parsed_ranges = {name: _range(ranges.get(name), name) for name in REQUIRED_RANGES}
    max_data_date = str(data.get("max_data_date", "")).strip()
    if not max_data_date:
        raise ValueError("workflow data.max_data_date is required")
    for name, value in parsed_ranges.items():
        if value["end"] > max_data_date:
            raise ValueError(f"workflow ranges.{name}.end exceeds data.max_data_date")

    model_adapter = str(config["model"].get("adapter", ""))
    if model_adapter not in ALLOWED_MODEL_ADAPTERS:
        raise ValueError(f"unsupported model adapter: {model_adapter!r}")
    if model_adapter == "rolling_lgbm_alpha" and not config["model"].get("config"):
        raise ValueError("rolling_lgbm_alpha requires model.config")
    if model_adapter == "torch_strong_alpha" and not isinstance(
        config["model"].get("config"), Mapping
    ):
        raise ValueError("torch_strong_alpha requires model.config object")

    strategy_adapter = str(config["strategy"].get("adapter", ""))
    if strategy_adapter not in ALLOWED_STRATEGY_ADAPTERS:
        raise ValueError(f"unsupported strategy adapter: {strategy_adapter!r}")
    if str(config["ledger"].get("adapter", "")) != "official_open_ledger":
        raise ValueError("formal workflow must use the project official_open_ledger adapter")
    if str(config["ledger"].get("execution_mode", "")) != "realistic":
        raise ValueError("formal workflow requires realistic execution mode")

    capitals = [int(value) for value in config["ledger"].get("capitals", [])]
    if not capitals or any(value not in {500_000, 1_000_000} for value in capitals):
        raise ValueError("workflow capitals must be drawn from 500000 and 1000000")
    stresses = {str(value) for value in config["ledger"].get("stresses", [])}
    if not REQUIRED_SELECTION_STRESSES.issubset(stresses):
        raise ValueError(
            "formal selection workflow requires normal, lag1, cost2x, and capacity_3pct"
        )

    splits = config["evaluation"].get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("workflow evaluation.splits must be non-empty")
    has_forward = False
    for split in splits:
        spec = get_split_spec(str(split))
        has_forward = has_forward or spec.is_forward
    if has_forward:
        assert_forward_parent_frozen(
            config["evaluation"].get("parent_fit_end"),
            config["evaluation"].get("parent_selection_end"),
        )
    return {
        "experiment_id": experiment_id,
        "model_adapter": model_adapter,
        "strategy_adapter": strategy_adapter,
        "splits": list(splits),
        "has_forward": has_forward,
        "ranges": parsed_ranges,
        "capitals": capitals,
    }


def validate_workflow_config(config: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = config.get("schema_version")
    if schema_version == WORKFLOW_SCHEMA_VERSION:
        result = _validate_workflow_v1(config)
        result["source_schema_version"] = WORKFLOW_SCHEMA_VERSION
        return result
    if schema_version == 2:
        v2 = validate_workflow_v2(config)
        normalized = normalize_workflow_v2(config)
        result = _validate_workflow_v1(normalized)
        result["source_schema_version"] = 2
        result["declared_model_adapter"] = v2["model_adapter"]
        return result
    raise ValueError("workflow schema_version must be 1 or 2")


def compile_workflow(
    config: Mapping[str, Any],
    *,
    project_root: str | Path,
    output_dir: str | Path,
    python: str | Path = sys.executable,
) -> dict[str, Any]:
    """Resolve a validated config into provenance scope and existing-runner stages."""

    source_config = dict(config)
    validation = validate_workflow_config(source_config)
    config = normalize_workflow_v2(source_config) if source_config.get("schema_version") == 2 else source_config
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    data_sources = []
    data_paths = {}
    for source in config["data"]["sources"]:
        path = _resolve(root, source["root"])
        data_paths[str(source["role"])] = path
        data_sources.append(
            {
                "role": str(source["role"]),
                "root": str(path),
                "fingerprint": fingerprint_path(path),
            }
        )

    features = config["features"]
    transform_manifest = features.get("transform_manifest")
    if transform_manifest:
        transform_path = _resolve(root, transform_manifest)
        transform_hash = fingerprint_path(transform_path)["value"]
    elif isinstance(features.get("transform_contract"), Mapping):
        transform_path = None
        transform_hash = canonical_json_hash(features["transform_contract"])
    else:
        transform_path = None
        transform_hash = str(features.get("transform_state_sha256", "")).strip()
    if not transform_hash:
        raise ValueError(
            "features requires transform_manifest, transform_contract, or transform_state_sha256"
        )

    split_roles = [
        {
            "split": split,
            "role": SPLIT_SPECS[split].role,
            "selection_eligible": SPLIT_SPECS[split].selection_eligible,
            "forward_used": SPLIT_SPECS[split].is_forward,
        }
        for split in validation["splits"]
    ]
    scope = {
        "stage": "workflow_compile",
        "data_sources": data_sources,
        "ranges": validation["ranges"],
        "max_data_date": str(config["data"]["max_data_date"]),
        "split_roles": split_roles,
        "transform": {
            "state_sha256": transform_hash,
            "fit_range": validation["ranges"]["train"],
            "manifest": str(transform_path) if transform_path else None,
        },
        "lineage": {
            "parent_fit_end": config["evaluation"].get("parent_fit_end"),
            "parent_selection_end": config["evaluation"].get("parent_selection_end"),
        },
    }

    stages = []
    model = config["model"]
    if validation["model_adapter"] in ROLLING_MODEL_ADAPTERS:
        if validation["model_adapter"] == "rolling_lgbm_alpha":
            model_config = _resolve(root, model["config"])
            if not model_config.is_file():
                raise FileNotFoundError(model_config)
            model_command = [
                str(Path(python)),
                "run/rolling_lgbm_alpha.py",
                "--config",
                str(model_config),
                "--output-dir",
                str(output / "model_signal"),
                "--experiment-id",
                f"{validation['experiment_id']}:model_signal",
            ]
        else:
            strong = model["config"]
            profile_path = _resolve(root, strong["profile_path"])
            schedule_path = _resolve(root, strong["schedule_path"])
            cache_meta = _resolve(root, strong["cache_meta"])
            for source in (profile_path, schedule_path, cache_meta):
                if not source.is_file():
                    raise FileNotFoundError(source)
            model_command = [
                str(Path(python)),
                "run/rolling_strong_staged_pilot.py",
                "--profile",
                str(profile_path),
                "--schedule",
                str(schedule_path),
                "--cache-meta",
                str(cache_meta),
                "--output-dir",
                str(output / "model_signal"),
                "--device",
                str(strong.get("device", "cuda")),
                "--transition",
                str(strong.get("transition", "exact")),
            ]
            windows = strong.get("windows")
            if windows:
                model_command.extend(["--windows", ",".join(map(str, windows))])
        stages.append(
            {
                "name": "model_signal",
                "adapter": validation["model_adapter"],
                "depends_on": [],
                "command": model_command,
            }
        )
        candidate_id = str(config["alpha"].get("candidate_id", "")).strip()
        if not candidate_id:
            raise ValueError("rolling workflow requires alpha.candidate_id")
        local_candidates = output / "workflow_candidates.csv"
        registration_command = [
            str(Path(python)),
            "run/materialize_workflow_candidates.py",
            "--rolling-experiment-dir",
            str(output / "model_signal"),
            "--candidate-id",
            candidate_id,
            "--output",
            str(local_candidates),
        ]
        for comparison_id in config["alpha"].get("comparison_candidate_ids", []):
            registration_command.extend(["--comparison-candidate-id", str(comparison_id)])
        stages.append(
            {
                "name": "candidate_registry",
                "adapter": "workflow_candidate_registry",
                "depends_on": ["model_signal"],
                "command": registration_command,
            }
        )
    else:
        candidate_ids = list(model.get("candidate_ids", []))
        if not candidate_ids:
            raise ValueError("frozen_registry_candidate requires model.candidate_ids")
        frozen_command = [
            str(Path(python)),
            "run/materialize_frozen_predictions.py",
            "--candidates-csv",
            str(root / "registry" / "candidates.csv"),
            "--output-dir",
            str(output / "model_signal"),
            "--workflow-dir",
            str(output),
        ]
        for candidate_id in candidate_ids:
            frozen_command.extend(["--candidate-id", str(candidate_id)])
        for split in validation["splits"]:
            frozen_command.extend(["--split", str(split)])
        stages.append(
            {
                "name": "model_signal",
                "adapter": "frozen_dated_predictions",
                "depends_on": [],
                "candidate_ids": candidate_ids,
                "command": frozen_command,
            }
        )

    ledger = config["ledger"]
    candidate_ids = list(config["alpha"].get("candidate_ids") or model.get("candidate_ids") or [])
    ledger_dependencies = ["model_signal"]
    if validation["model_adapter"] in ROLLING_MODEL_ADAPTERS:
        candidate_ids = [str(config["alpha"]["candidate_id"])] + [
            str(value) for value in config["alpha"].get("comparison_candidate_ids", [])
        ]
        ledger_dependencies = ["candidate_registry"]
    ledger_command = None
    ledger_status = "ready"
    if candidate_ids:
        ledger_command = [
            str(Path(python)),
            "run/official_backtest_from_registry.py",
            "--output-root",
            str(output / "ledger"),
            "--run-id",
            validation["experiment_id"],
            "--stresses",
            ",".join(map(str, ledger.get("stresses", []))),
            "--portfolio-values",
            ",".join(map(str, validation["capitals"])),
            "--execution-mode",
            "realistic",
            "--reports-csv",
            str(output / "workflow_reports.csv"),
        ]
        if validation["model_adapter"] in ROLLING_MODEL_ADAPTERS:
            ledger_command.extend(["--candidates-csv", str(output / "workflow_candidates.csv")])
        research_data = data_paths.get("research_market") or data_paths.get("market")
        forward_data = data_paths.get("forward_market")
        if research_data:
            ledger_command.extend(["--research-data-dir", str(research_data)])
        if forward_data:
            ledger_command.extend(["--forward-data-dir", str(forward_data)])
        parameter_flags = {
            "target_fracs": "--target-fracs",
            "hold_fracs": "--hold-fracs",
            "rebalance_bands": "--rebalance-bands",
            "max_new_names": "--max-new-names-list",
            "exit_hold_fracs": "--exit-hold-fracs",
            "switch_gap_fracs": "--switch-gap-fracs",
        }
        strategy = config["strategy"]
        for field, flag in parameter_flags.items():
            value = strategy.get(field)
            if value not in (None, ""):
                if isinstance(value, list):
                    value = ",".join(map(str, value))
                ledger_command.extend([flag, str(value)])
        market_data = ExecutionMarketDataContract(
            backend=ledger.get("ohlc_backend", "legacy"),
            market_daily_store_root=ledger.get(
                "market_daily_store_root", "data/market_daily_candidate_v2"
            ),
            monthly_cache_root=ledger.get(
                "ohlc_monthly_cache_dir", "cache/ohlcv_monthly_v3_candidate"
            ),
        )
        ledger_command.extend(market_data.cli_args())
        for candidate_id in candidate_ids:
            ledger_command.extend(["--candidate-id", str(candidate_id)])
        for split in validation["splits"]:
            ledger_command.extend(["--split", split])
        if validation["has_forward"]:
            ledger_command.extend(
                [
                    "--parent-fit-end",
                    str(config["evaluation"]["parent_fit_end"]),
                    "--parent-selection-end",
                    str(config["evaluation"]["parent_selection_end"]),
                ]
            )
        # Every workflow owns an isolated reports CSV consumed by its scorecard.
        # This never writes the project-global registry because --reports-csv
        # above is rooted inside the workflow output directory.
        ledger_command.append("--append-registry")
    else:
        ledger_status = "awaiting_alpha_registration_adapter"
    stages.append(
        {
            "name": "realistic_ledger",
            "adapter": "official_open_ledger",
            "depends_on": ledger_dependencies,
            "status": ledger_status,
            "command": ledger_command,
        }
    )

    stages.append(
        {
            "name": "scorecard",
            "adapter": "registry_scorecard",
            "depends_on": ["realistic_ledger"],
            "command": [
                str(Path(python)),
                "run/scorecard_from_registry.py",
                "--output-dir",
                str(output / "scorecard"),
                "--reports-csv",
                str(output / "workflow_reports.csv"),
            ] + (
                [
                    "--candidates-csv",
                    str(output / "workflow_candidates.csv"),
                ]
                if validation["model_adapter"] in ROLLING_MODEL_ADAPTERS
                else []
            ) + [item for candidate_id in candidate_ids for item in ("--candidate-id", str(candidate_id))],
        }
    )
    for split in validation["splits"]:
        stages[-1]["command"].extend(["--expected-split", split])
    if config.get("reports", {}).get("record_templates"):
        stages.append(
            {
                "name": "standard_records",
                "adapter": "workflow_standard_records",
                "depends_on": ["scorecard"],
                "command": [
                    str(Path(python)),
                    "run/materialize_workflow_records.py",
                    "--workflow-dir",
                    str(output),
                ],
            }
        )
    return {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "workflow_schema_version": validation["source_schema_version"],
        "experiment_id": validation["experiment_id"],
        "config_sha256": canonical_json_hash(source_config),
        "project_root": str(root),
        "output_dir": str(output),
        "scope": scope,
        "stages": stages,
    }
