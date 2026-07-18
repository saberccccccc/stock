"""Contracts and append-only records for bounded research tuning."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from experiments.recording import canonical_json_hash


ALLOWED_MODEL_OVERRIDES = frozenset({
    "learning_rate",
    "num_leaves",
    "min_data_in_leaf",
    "feature_fraction",
    "bagging_fraction",
    "lambda_l2",
    "num_boost_round",
    "early_stopping_rounds",
})


def load_tuning_spec(path: str | Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    trials = spec.get("trials", [])
    max_trials = int(spec.get("max_trials", 0))
    if not trials or max_trials <= 0 or len(trials) > max_trials:
        raise ValueError("tuning spec must contain 1..max_trials trials")
    ids = [trial.get("trial_id") for trial in trials]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("trial_id values must be unique and non-empty")
    for trial in trials:
        overrides = trial.get("overrides", {})
        if not overrides or len(overrides) != 1:
            raise ValueError("each first-stage trial must override exactly one model parameter")
        unknown = set(overrides) - ALLOWED_MODEL_OVERRIDES
        if unknown:
            raise ValueError(f"unsupported model overrides: {sorted(unknown)}")
    return spec


def build_trial_config(base_config: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(dict(base_config))
    model = dict(config.get("model", {}))
    overrides = dict(trial.get("overrides", {}))
    model.update(overrides)
    config["model"] = model
    config["name"] = f"{config.get('name', 'rolling_lgbm')}__{trial['trial_id']}"
    config["tuning"] = {
        "trial_id": trial["trial_id"],
        "overrides": overrides,
        "base_config_hash": canonical_json_hash(base_config),
    }
    return config


def append_trial_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return target


def latest_trial_records(path: str | Path) -> dict[str, dict[str, Any]]:
    target = Path(path)
    latest: dict[str, dict[str, Any]] = {}
    if not target.is_file():
        return latest
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        latest[record["trial_id"]] = record
    return latest
