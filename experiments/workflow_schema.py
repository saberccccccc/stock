"""Workflow v2 schema validation and compatibility normalization."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_V2_SCHEMA_PATH = ROOT / "schemas" / "workflow_v2.schema.json"
SELECTION_SPLITS = ("val_2024", "test_2025")
OBSERVATION_SPLITS = ("forward_2026",)
V2_RUNTIME_ADAPTERS = {"rolling_lgbm_alpha", "torch_strong_alpha", "frozen_artifact"}


def _load_schema() -> dict[str, Any]:
    return json.loads(WORKFLOW_V2_SCHEMA_PATH.read_text(encoding="utf-8"))


def _date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"workflow v2 {field} must be an ISO date") from exc


def validate_workflow_v2(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the draft schema plus project-specific cross-field invariants."""

    schema = _load_schema()
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(dict(config)), key=lambda item: list(item.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "<root>"
        raise ValueError(f"workflow v2 schema error at {location}: {error.message}")

    governance = config["governance"]
    evaluation = config["evaluation"]
    if list(governance["selection_splits"]) != list(evaluation["selection_splits"]):
        raise ValueError("workflow v2 governance/evaluation selection_splits disagree")
    if list(governance["observation_splits"]) != list(evaluation["observation_splits"]):
        raise ValueError("workflow v2 governance/evaluation observation_splits disagree")
    if tuple(governance["selection_splits"]) != SELECTION_SPLITS:
        raise ValueError("formal workflow v2 selection_splits must be val_2024 and test_2025")
    if any(split not in OBSERVATION_SPLITS for split in governance["observation_splits"]):
        raise ValueError("workflow v2 contains an unknown observation split")

    max_data_date = _date(config["data"]["max_data_date"], "data.max_data_date")
    for name, value in config["ranges"].items():
        start = _date(value["start"], f"ranges.{name}.start")
        end = _date(value["end"], f"ranges.{name}.end")
        if start > end:
            raise ValueError(f"workflow v2 ranges.{name}.start exceeds end")
        if end > max_data_date:
            raise ValueError(f"workflow v2 ranges.{name}.end exceeds data.max_data_date")

    segments = config["dataset"]["segments"]
    for name, value in segments.items():
        if _date(value["start"], f"dataset.segments.{name}.start") > _date(
            value["end"], f"dataset.segments.{name}.end"
        ):
            raise ValueError(f"workflow v2 dataset segment {name} is reversed")

    if governance["observation_splits"]:
        for field in ("parent_fit_end", "parent_selection_end"):
            if field not in governance:
                raise ValueError(f"workflow v2 governance.{field} is required for Forward")
        if _date(governance["parent_fit_end"], "governance.parent_fit_end") > date(2025, 12, 31):
            raise ValueError("workflow v2 Forward parent_fit_end must not exceed 2025-12-31")
        if _date(governance["parent_selection_end"], "governance.parent_selection_end") > date(2025, 12, 31):
            raise ValueError("workflow v2 Forward parent_selection_end must not exceed 2025-12-31")

    model = config["model"]
    adapter = str(model["adapter"])
    model_config = model["config"]
    state_policy = str(config["processors"]["state_policy"])
    if adapter == "frozen_artifact" and state_policy != "frozen_parent_no_refit":
        raise ValueError(
            "workflow v2 frozen_artifact requires processors.state_policy="
            "frozen_parent_no_refit"
        )
    if adapter != "frozen_artifact" and state_policy != "fit_train_freeze_elsewhere":
        raise ValueError(
            "workflow v2 learnable adapters require processors.state_policy="
            "fit_train_freeze_elsewhere"
        )
    if adapter == "rolling_lgbm_alpha" and not str(model_config.get("path", "")).strip():
        raise ValueError("workflow v2 rolling_lgbm_alpha requires model.config.path")
    if adapter == "torch_strong_alpha":
        missing = [
            field
            for field in ("profile_path", "schedule_path", "cache_meta")
            if not str(model_config.get(field, "")).strip()
        ]
        if missing:
            raise ValueError(
                f"workflow v2 torch_strong_alpha missing model.config fields: {missing}"
            )
        if model_config.get("transition", "exact") not in {"exact", "selected"}:
            raise ValueError("workflow v2 torch_strong_alpha transition must be exact or selected")
    if adapter == "frozen_artifact":
        candidate_ids = [str(value).strip() for value in model_config.get("candidate_ids", [])]
        if not candidate_ids or any(not value for value in candidate_ids):
            raise ValueError("workflow v2 frozen_artifact requires model.config.candidate_ids")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("workflow v2 frozen_artifact candidate_ids must be unique")
        main_candidate = str(config["signal"].get("candidate_id", "")).strip()
        if main_candidate and main_candidate not in candidate_ids:
            raise ValueError(
                "workflow v2 frozen_artifact signal.candidate_id must be present "
                "in model.config.candidate_ids"
            )

    ledger = config["ledger"]
    shadow_backend = ledger.get("ohlc_shadow_backend")
    shadow_report = ledger.get("ohlc_shadow_report")
    if bool(shadow_backend) != bool(shadow_report):
        raise ValueError(
            "workflow v2 ledger dual-read requires both ohlc_shadow_backend "
            "and ohlc_shadow_report"
        )
    if shadow_backend and {
        str(ledger.get("ohlc_backend", "legacy")),
        str(shadow_backend),
    } != {"csv", "monthly"}:
        raise ValueError(
            "workflow v2 ledger dual-read requires one csv and one monthly backend"
        )

    return {
        "experiment_id": str(config["experiment_id"]),
        "model_adapter": adapter,
        "selection_splits": list(governance["selection_splits"]),
        "observation_splits": list(governance["observation_splits"]),
    }


def _list_value(params: Mapping[str, Any], singular: str, plural: str):
    value = params.get(plural, params.get(singular))
    if value is None:
        return None
    return list(value) if isinstance(value, list) else [value]


def normalize_workflow_v2(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate v2 into the proven v1 compiler input without hiding values."""

    validate_workflow_v2(config)
    source = deepcopy(dict(config))
    model = source["model"]
    adapter = str(model["adapter"])
    if adapter not in V2_RUNTIME_ADAPTERS:
        raise NotImplementedError(
            f"workflow v2 model adapter {adapter!r} is declared for Q3 but is not executable in Q1"
        )

    if adapter == "rolling_lgbm_alpha":
        runtime_model = {
            "adapter": "rolling_lgbm_alpha",
            "config": str(model["config"]["path"]),
        }
    elif adapter == "torch_strong_alpha":
        runtime_model = {
            "adapter": "torch_strong_alpha",
            "config": deepcopy(model["config"]),
        }
    else:
        runtime_model = {
            "adapter": "frozen_registry_candidate",
            "candidate_ids": list(model["config"]["candidate_ids"]),
        }

    signal = source["signal"]
    alpha = {
        "transform": str(signal["transform"]),
        "candidate_id": str(signal.get("candidate_id", "")).strip(),
        "comparison_candidate_ids": list(signal.get("comparison_candidate_ids", [])),
    }
    if adapter == "frozen_artifact":
        alpha["candidate_ids"] = list(model["config"]["candidate_ids"])

    params = source["strategy"]["params"]
    strategy = {"adapter": source["strategy"]["adapter"]}
    for singular, plural in (
        ("target_frac", "target_fracs"),
        ("hold_frac", "hold_fracs"),
        ("rebalance_band", "rebalance_bands"),
        ("max_new_names", "max_new_names"),
        ("exit_hold_frac", "exit_hold_fracs"),
        ("switch_gap_frac", "switch_gap_fracs"),
    ):
        value = _list_value(params, singular, plural)
        if value is not None:
            strategy[plural] = value

    governance = source["governance"]
    evaluation_splits = list(governance["selection_splits"]) + list(governance["observation_splits"])
    return {
        "schema_version": 1,
        "experiment_id": source["experiment_id"],
        "data": {
            "sources": [
                {"role": item["role"], "root": item["root"]}
                for item in source["data"]["sources"]
            ],
            "max_data_date": source["data"]["max_data_date"],
        },
        "ranges": deepcopy(source["ranges"]),
        "features": {
            "set": source["dataset"]["feature_set"],
            "transform_contract": {
                "type": "workflow_v2_processor_chain",
                "dataset_adapter": source["dataset"]["adapter"],
                "processors": deepcopy(source["processors"]),
            },
        },
        "labels": deepcopy(source["dataset"]["label"]),
        "windows": {
            "type": str(model["config"].get("window_type", "workflow_v2_declared")),
            "selection_splits": list(governance["selection_splits"]),
        },
        "model": runtime_model,
        "checkpoint": {
            "rule": source["checkpoint"]["rule"],
            "forward_selection_allowed": False,
        },
        "alpha": alpha,
        "strategy": strategy,
        "ledger": deepcopy(source["ledger"]),
        "evaluation": {
            "splits": evaluation_splits,
            "parent_fit_end": governance.get("parent_fit_end"),
            "parent_selection_end": governance.get("parent_selection_end"),
        },
        "reports": {
            "scorecard": "registry_apm_scorecard",
            "attribution_required": "risk_attribution" in source["records"],
            "append_registry": False,
            "record_templates": list(source["records"]),
        },
    }
