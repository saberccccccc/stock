"""Concrete bridge from the governed Torch adapter to the established trainer."""

from __future__ import annotations

import hashlib
import gc
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from backtest.engine import load_v9_checkpoint
from backtest.runtime import build_v9_backtest_config
from data.cache_metadata import load_explicit_cross_section_meta
from data.dataset_runtime import (
    ProcessorChains,
    ProjectDataset,
    build_v14_strong_rolling_dataset,
)
from data.providers import DataView, V14MemmapProvider
from experiments.model_adapters import TorchStrongAlphaAdapter
from experiments.recording import canonical_json_hash, sha256_file
from experiments.rolling import RollingWindow, resolve_window_indices
from experiments.strong_rolling import run_with_memory_guard


STRONG_REQUIRED_FIELDS = (
    "X",
    "y",
    "y_seq",
    "raw_y_seq",
    "lag1_y_seq",
    "lag1_mask",
    "risk",
    "industry_ids",
)


def validate_strong_training_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in STRONG_REQUIRED_FIELDS if name not in sample]
    if missing:
        raise ValueError(f"strong trainer Dataset sample is missing fields: {missing}")
    rows = int(np.asarray(sample["X"]).shape[0])
    if rows <= 0:
        raise ValueError("strong trainer Dataset sample is empty")
    for name in STRONG_REQUIRED_FIELDS[1:]:
        if int(np.asarray(sample[name]).shape[0]) != rows:
            raise ValueError(f"strong trainer Dataset field {name} has mismatched rows")
    return {
        "rows": rows,
        "input_dim": int(np.asarray(sample["X"]).shape[1]),
        "risk_dim": int(np.asarray(sample["risk"]).shape[1]),
    }


def canonical_torch_state_hash(model) -> str:
    """Hash tensor values without depending on torch.save container bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(repr(tuple(values.shape)).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


class ExistingStrongTrainerDelegate:
    """Run the frozen legacy command, then return its model through the Adapter API."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        project_root: str | Path,
        checkpoint_path: str | Path,
        horizon_indices: Sequence[int],
        expected_epoch: int,
        expected_input_dim: int,
        device: str,
        min_free_gib: float = 0.75,
        reuse_existing_checkpoint: bool = False,
    ):
        self.command = [str(value) for value in command]
        self.project_root = Path(project_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.horizon_indices = tuple(int(value) for value in horizon_indices)
        self.expected_epoch = int(expected_epoch)
        self.expected_input_dim = int(expected_input_dim)
        self.device = str(device)
        self.min_free_gib = float(min_free_gib)
        self.reuse_existing_checkpoint = bool(reuse_existing_checkpoint)

    def __call__(
        self, dataset: ProjectDataset, config: Mapping[str, Any]
    ) -> tuple[Any, Mapping[str, Any]]:
        sample = next(iter(dataset.prepare("train", data_key="learn")))
        sample_contract = validate_strong_training_sample(sample)
        if sample_contract["input_dim"] != self.expected_input_dim:
            raise ValueError(
                "strong trainer input width differs from frozen profile: "
                f"expected={self.expected_input_dim} actual={sample_contract['input_dim']}"
            )
        checkpoint_reused = self.checkpoint_path.is_file()
        if checkpoint_reused and not self.reuse_existing_checkpoint:
            raise FileExistsError(f"delegate checkpoint already exists: {self.checkpoint_path}")

        if not checkpoint_reused:
            run_with_memory_guard(
                self.command,
                cwd=self.project_root,
                min_free_gib=self.min_free_gib,
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)

        import torch

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        actual_epoch = int(checkpoint.get("epoch", -1))
        if actual_epoch != self.expected_epoch:
            raise ValueError(
                f"delegate checkpoint epoch mismatch: expected={self.expected_epoch} actual={actual_epoch}"
            )
        actual_dim = int(checkpoint.get("arch_config", {}).get("input_dim", -1))
        if actual_dim != self.expected_input_dim:
            raise ValueError(
                f"delegate checkpoint input width mismatch: expected={self.expected_input_dim} actual={actual_dim}"
            )

        runtime_config = build_v9_backtest_config()
        runtime_config.horizon_indices = self.horizon_indices
        model, _, regime_dim = load_v9_checkpoint(
            str(self.checkpoint_path), [sample], runtime_config, self.device
        )
        if isinstance(config, MutableMapping):
            config["regime_dim"] = int(regime_dim)
        metrics = {
            "delegate": "existing_run_train_v1",
            "trainer_command_launched": not checkpoint_reused,
            "checkpoint_reused": checkpoint_reused,
            "command_sha256": canonical_json_hash({"command": self.command}),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "model_state_sha256": canonical_torch_state_hash(model),
            "epoch": actual_epoch,
            "selection_metric": checkpoint.get("best_metric"),
            "selection_score": checkpoint.get("best_score"),
            "checkpoint_val_loss": checkpoint.get("val_loss"),
            "regime_dim": int(regime_dim),
            "sample_contract": sample_contract,
        }
        return model, metrics


def build_strong_window_dataset(
    *,
    project_root: str | Path,
    cache_meta: str | Path,
    profile: Mapping[str, Any],
    schedule: Mapping[str, Any],
    window: RollingWindow,
) -> tuple[ProjectDataset, dict[str, Any]]:
    """Build one purged strong-model Dataset under the frozen rolling contract."""

    root = Path(project_root).resolve()
    meta_path = Path(cache_meta).resolve()
    inferred = profile["inferred"]
    expected_dim = int(profile["confirmed"]["architecture"]["input_dim"])
    meta = load_explicit_cross_section_meta(
        meta_path,
        project_root=root,
        expected_input_dim=expected_dim,
        required_label_families=(
            inferred["label_family"],
            inferred["lag1_label_family"],
        ),
        logical_end=window.predict_end,
    )
    indices = resolve_window_indices(
        meta["all_dates"],
        window,
        int(schedule["monthly_schedule"]["label_end_offset"]),
    )
    counts = {name: len(values) for name, values in indices.items()}
    expected = next(
        (
            item
            for item in schedule["monthly_schedule"].get("window_counts", ())
            if item["name"] == window.name
        ),
        None,
    )
    if expected is not None:
        for name in ("train", "valid", "predict"):
            if counts[name] != int(expected[name]):
                raise ValueError(
                    f"schedule index count drift for {window.name}.{name}: "
                    f"expected={expected[name]} actual={counts[name]}"
                )

    view = DataView.create(
        name=f"strong_staged_adapter:{window.name}",
        physical_root=root / "data" / "raw",
        feature_warmup_start=window.train_start,
        feature_warmup_end=window.train_end,
        task_start=window.train_start,
        task_end=window.predict_end,
        evaluation_start=window.predict_start,
        evaluation_end=window.predict_end,
        max_data_date=window.predict_end,
    )
    provider = V14MemmapProvider(meta=meta, meta_path=meta_path, data_view=view)
    dataset = build_v14_strong_rolling_dataset(
        provider=provider,
        segment_indices=indices,
        label_family=inferred["label_family"],
        horizon_indices=inferred["horizon_indices"],
        target_horizon_index=max(inferred["horizon_indices"]),
        include_raw_returns=True,
        include_lag1_labels=True,
        lag1_label_family=inferred["lag1_label_family"],
        processors=ProcessorChains(),
    )
    dataset.fit()
    return dataset, {
        "window": window.name,
        "segment_counts": counts,
        "processor_state": dataset.handler.state_payload(),
        "physical_data_start": meta["physical_data_start"],
        "physical_data_end": meta["physical_data_end"],
    }


def fit_strong_stage_through_adapter(
    *,
    dataset: ProjectDataset,
    stage: Mapping[str, Any],
    command: Sequence[str],
    project_root: str | Path,
    horizon_indices: Sequence[int],
    expected_input_dim: int,
    device: str,
    min_free_gib: float,
    model_id: str,
) -> dict[str, Any]:
    """Run one staged trainer command through the governed Torch Adapter."""

    delegate = ExistingStrongTrainerDelegate(
        command=command,
        project_root=project_root,
        checkpoint_path=stage["exact_checkpoint"],
        horizon_indices=horizon_indices,
        expected_epoch=int(stage["target_epoch"]),
        expected_input_dim=expected_input_dim,
        device=device,
        min_free_gib=min_free_gib,
        reuse_existing_checkpoint=True,
    )
    adapter = TorchStrongAlphaAdapter(
        model_id=model_id,
        config={
            "device": device,
            "score_transform": "v9_rank",
            "checkpoint_rule": "internal_rawtopstable_pilot_only_no_formal_selection",
        },
        fit_delegate=delegate,
    )
    fit_result = adapter.fit(dataset)
    metrics = dict(fit_result.metrics)
    metrics["adapter"] = adapter.adapter_name
    metrics["adapter_contract_sha256"] = adapter.manifest()["contract_sha256"]
    adapter.model = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return metrics
