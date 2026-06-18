"""Utilities for loading reproducible training experiment presets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingExperiment:
    """One concrete train.py invocation expanded from a preset suite."""

    suite: str
    experiment_id: str
    params: dict[str, Any]

    @property
    def output_dir(self) -> str:
        return str(self.params["output_dir"])

    def train_argv(self) -> list[str]:
        return params_to_train_argv(self.params)


@dataclass(frozen=True)
class TrainingSuite:
    """A group of training experiments sharing common parameters."""

    name: str
    common: dict[str, Any]
    experiments: tuple[TrainingExperiment, ...]


def load_training_suite(path: str | Path) -> TrainingSuite:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    common = dict(payload.get("common", {}))
    raw_experiments = payload.get("experiments", [])
    if not raw_experiments:
        raise ValueError(f"Training suite has no experiments: {path}")

    experiments = []
    for raw in raw_experiments:
        if "id" not in raw:
            raise ValueError(f"Training experiment missing id in {path}")
        if "output_dir" not in raw:
            raise ValueError(f"Training experiment {raw['id']} missing output_dir")
        params = {**common, **raw}
        experiment_id = str(params.pop("id"))
        experiments.append(
            TrainingExperiment(
                suite=path.stem,
                experiment_id=experiment_id,
                params=params,
            )
        )
    return TrainingSuite(
        name=path.stem,
        common=common,
        experiments=tuple(experiments),
    )


def load_training_suites(config_dir: str | Path) -> tuple[TrainingSuite, ...]:
    config_dir = Path(config_dir)
    return tuple(
        load_training_suite(path)
        for path in sorted(config_dir.glob("*.json"))
    )


def params_to_train_argv(params: dict[str, Any]) -> list[str]:
    argv = []
    for key, value in params.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])
    return argv
