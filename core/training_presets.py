"""Utilities for loading reproducible training experiment presets."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_TRAIN_PARAM_KEYS = frozenset(
    {
        "model",
        "test_stocks",
        "epochs",
        "lr",
        "device",
        "output_dir",
        "data_dir",
        "batch_size",
        "val_batch_size",
        "accum_steps",
        "memmap_trim_interval",
        "top_focus_loss_weight",
        "top_focus_temperature",
        "top_focus_delay_epochs",
        "downside_loss_weight",
        "downside_temperature",
        "downside_delay_epochs",
        "lag1_loss_weight",
        "lag1_delay_epochs",
        "lag1_top_focus_loss_weight",
        "lag1_top_focus_temperature",
        "lag1_top_focus_delay_epochs",
        "pairwise_top_loss_weight",
        "pairwise_top_frac",
        "pairwise_num_pairs",
        "pairwise_model_top_weight",
        "pairwise_delay_epochs",
        "best_val_metric",
        "eval_top_fracs",
        "horizon_weights",
        "save_every_epoch",
        "early_stop_patience",
        "seed",
        "industry_loss_weight",
        "multi_loss_weight",
        "diversity_loss_weight",
        "spread_loss_weight",
        "spread_delay_epochs",
        "resume_from",
        "reset_optimizer",
        "train_label_end",
        "val_label_end",
    }
)


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

    def train_command(
        self,
        python_exe: str = "python",
        train_script: str = "run/train.py",
    ) -> str:
        return render_train_command(
            self.params,
            python_exe=python_exe,
            train_script=train_script,
        )


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
    validate_training_params(common, context=f"{path}:common")
    raw_experiments = payload.get("experiments", [])
    if not raw_experiments:
        raise ValueError(f"Training suite has no experiments: {path}")

    experiments = []
    for raw in raw_experiments:
        if "id" not in raw:
            raise ValueError(f"Training experiment missing id in {path}")
        if "output_dir" not in raw:
            raise ValueError(f"Training experiment {raw['id']} missing output_dir")
        validate_training_params(
            {key: value for key, value in raw.items() if key != "id"},
            context=f"{path}:{raw['id']}",
        )
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
    paths = []
    for path in sorted(config_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "common" in payload or "experiments" in payload:
            paths.append(path)
    return tuple(
        load_training_suite(path)
        for path in paths
    )


def validate_training_params(params: dict[str, Any], context: str = "training params") -> None:
    unknown = sorted(set(params) - VALID_TRAIN_PARAM_KEYS)
    if unknown:
        allowed = ", ".join(sorted(VALID_TRAIN_PARAM_KEYS))
        raise ValueError(
            f"Unknown train.py parameter(s) in {context}: {', '.join(unknown)}. "
            f"Allowed keys: {allowed}"
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


def render_train_command(
    params: dict[str, Any],
    python_exe: str = "python",
    train_script: str = "run/train.py",
) -> str:
    """Render a shell-safe train.py command without executing it."""

    return subprocess.list2cmdline([python_exe, train_script, *params_to_train_argv(params)])
