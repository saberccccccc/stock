from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from alpha.io import write_alpha_rows
from data.dataset_runtime import DataHandlerRuntime, ProcessorChains, ProjectDataset
from data.providers import DateRange
from experiments.model_adapters import (
    FrozenArtifactAdapter,
    LegacyReadOnlyAdapter,
    LightGBMModelAdapter,
    ModelAdapterDependencies,
    PredictionFrame,
    TorchStrongAlphaAdapter,
    create_model_adapter,
)
from experiments.strong_adapter_runtime import (
    ExistingStrongTrainerDelegate,
    fit_strong_stage_through_adapter,
)


def _dataset():
    segments = {
        "train": DateRange.create("2023-01-01", "2023-12-31", field="train"),
        "valid": DateRange.create("2024-01-01", "2024-12-31", field="valid"),
        "test": DateRange.create("2025-01-01", "2025-12-31", field="test"),
    }

    def factory(date_range):
        year = date_range.start.year
        for day in range(2):
            x0 = float(year - 2020 + day)
            x = np.array([[x0, 0.0], [x0 + 1.0, 1.0], [x0 + 2.0, 2.0]], dtype=np.float32)
            yield {
                "date": f"{year}-01-0{day + 2}",
                "codes": ["A", "B", "C"],
                "X": x,
                "y": (x[:, 0] * 0.1 + x[:, 1] * 0.2).astype(np.float32),
                "risk": np.zeros((3, 1), dtype=np.float32),
                "industry_ids": np.array([0, 0, 1], dtype=np.int64),
            }

    return ProjectDataset(
        DataHandlerRuntime(sample_factory=factory, segments=segments, processors=ProcessorChains())
    )


def test_prediction_frame_roundtrip_and_future_asof_rejection(tmp_path):
    prediction = PredictionFrame.from_daily_scores(
        [{"date": "2024-01-03", "codes": ["B", "A"], "scores": [0.1, 0.5]}],
        model_id="model",
    )
    path = prediction.write_alpha_jsonl(tmp_path / "alpha.jsonl")

    assert prediction.to_alpha_rows()[0]["codes"] == ["A", "B"]
    assert path.is_file()
    assert prediction.manifest()["rows"] == 2
    invalid = prediction.frame.copy()
    invalid["asof_time"] = pd.Timestamp("2024-01-04")
    with pytest.raises(ValueError, match="after trade_date"):
        PredictionFrame(invalid)


def test_prediction_frame_can_replay_legacy_numpy_tie_order():
    prediction = PredictionFrame.from_daily_scores(
        [{"date": "2024-01-03", "codes": ["A", "B", "C"], "scores": [0.5, 0.5, 0.7]}],
        model_id="model",
    )
    assert prediction.to_alpha_rows()[0]["codes"] == ["C", "A", "B"]
    assert prediction.to_alpha_rows(tie_breaker="legacy_numpy")[0]["codes"] == ["C", "B", "A"]


def test_frozen_and_legacy_adapters_preserve_artifact_provenance(tmp_path):
    alpha = write_alpha_rows(
        tmp_path / "frozen.jsonl",
        [{"date": "2024-01-03", "codes": ["A", "B"], "alpha": [0.4, 0.2]}],
    )
    dataset = _dataset()
    adapter = FrozenArtifactAdapter(model_id="frozen", alpha_paths={"valid": alpha})
    adapter.fit(dataset)
    prediction = adapter.predict(dataset, "valid")
    state = adapter.save_state(tmp_path / "frozen_state.json")

    restored = FrozenArtifactAdapter(model_id="frozen", alpha_paths={})
    restored.resume(state)
    assert restored.predict(dataset, "valid").manifest() == prediction.manifest()

    legacy = LegacyReadOnlyAdapter(model_id="legacy", alpha_paths={"valid": alpha})
    legacy.fit(dataset)
    with pytest.raises(RuntimeError, match="read-only"):
        legacy.save_state(tmp_path / "legacy.json")


def test_lightgbm_adapter_fits_predicts_and_resumes(tmp_path):
    dataset = _dataset()
    adapter = LightGBMModelAdapter(
        model_id="lgb",
        config={"num_boost_round": 12, "early_stopping_rounds": 3, "num_threads": 1},
    )

    result = adapter.fit(dataset)
    prediction = adapter.predict(dataset, "test")
    model_path = adapter.save_state(tmp_path / "model.txt")

    assert result.status == "fitted"
    assert prediction.manifest()["dates"] == 2
    restored = LightGBMModelAdapter(model_id="lgb")
    restored.resume(model_path)
    assert restored.predict(dataset, "test").manifest()["rows"] == 6


def test_lightgbm_adapter_daily_seeded_quota_matches_date_balanced_contract():
    segments = {
        "train": DateRange.create("2023-01-01", "2023-01-02", field="train"),
        "valid": DateRange.create("2024-01-01", "2024-01-02", field="valid"),
        "test": DateRange.create("2025-01-01", "2025-01-02", field="test"),
    }

    def factory(date_range):
        year = date_range.start.year
        for day in range(2):
            values = np.arange(12, dtype=np.float32).reshape(6, 2) + day + year
            yield {
                "time_index": year * 10 + day,
                "date": f"{year}-01-0{day + 1}",
                "codes": [f"S{i}" for i in range(6)],
                "X": values,
                "y": values[:, 0] * 0.1,
            }

    dataset = ProjectDataset(
        DataHandlerRuntime(sample_factory=factory, segments=segments, processors=ProcessorChains())
    )
    adapter = LightGBMModelAdapter(
        model_id="quota",
        config={
            "sampling_mode": "daily_seeded_quota_v1",
            "segment_date_counts": {"train": 2, "valid": 2},
            "max_train_rows": 6,
            "max_valid_rows": 4,
            "seed": 17,
            "num_boost_round": 3,
            "early_stopping_rounds": 1,
            "num_threads": 1,
        },
    )
    result = adapter.fit(dataset)
    assert result.metrics["train_rows"] == 6
    assert result.metrics["valid_rows"] == 4
    assert result.metrics["sampling_mode"] == "daily_seeded_quota_v1"


class TinyCrossSectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)

    def forward(self, x, risk, mask, industry_ids):
        score = self.linear(x).squeeze(-1)
        return score, score.unsqueeze(-1), score.unsqueeze(-1)


def test_torch_strong_adapter_uses_explicit_delegate_and_checkpoint_factory(tmp_path):
    dataset = _dataset()

    def fit_delegate(project_dataset, config):
        model = TinyCrossSectionModel()
        with torch.no_grad():
            model.linear.weight[:] = torch.tensor([[1.0, 0.5]])
        return model, {"valid_metric": 0.1}

    adapter = TorchStrongAlphaAdapter(
        model_id="torch",
        config={"device": "cpu", "checkpoint_rule": "valid_only"},
        fit_delegate=fit_delegate,
        model_factory=lambda checkpoint, sample: TinyCrossSectionModel(),
    )
    result = adapter.fit(dataset)
    prediction = adapter.predict(dataset, "test")
    checkpoint = adapter.save_state(tmp_path / "torch.pt")

    assert result.checkpoint_rule == "valid_only"
    assert prediction.to_alpha_rows()[0]["codes"] == ["C", "B", "A"]
    restored = TorchStrongAlphaAdapter(
        model_id="torch",
        config={"device": "cpu"},
        model_factory=lambda state, sample: TinyCrossSectionModel(),
    )
    restored.resume(checkpoint, dataset)
    assert restored.predict(dataset, "test").manifest()["rows"] == 6


def test_torch_fit_requires_explicit_existing_trainer_delegate():
    adapter = TorchStrongAlphaAdapter(model_id="torch", config={"device": "cpu"}, model=TinyCrossSectionModel())

    with pytest.raises(RuntimeError, match="explicit existing-trainer delegate"):
        adapter.fit(_dataset())


class RiskAwareModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.observed_risk_dim = None

    def forward(self, x, risk, mask, industry_ids):
        self.observed_risk_dim = int(risk.shape[-1])
        score = x[..., 0] + risk[..., 0] * 0.0 + self.anchor
        return score, score.unsqueeze(-1), score.unsqueeze(-1)


def test_torch_adapter_matches_v9_rank_transform_and_slices_risk():
    segments = {
        "train": DateRange.create("2023-01-01", "2023-01-31", field="train"),
        "predict": DateRange.create("2024-01-01", "2024-01-31", field="predict"),
    }

    def factory(date_range):
        yield {
            "date": str(date_range.start.date()),
            "codes": ["A", "B", "C"],
            "X": np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            "y": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "risk": np.ones((3, 4), dtype=np.float32),
            "industry_ids": np.array([0, 1, 2], dtype=np.int64),
        }

    dataset = ProjectDataset(
        DataHandlerRuntime(sample_factory=factory, segments=segments, processors=ProcessorChains())
    )
    model = RiskAwareModel()
    adapter = TorchStrongAlphaAdapter(
        model_id="rank",
        config={"device": "cpu", "regime_dim": 2, "score_transform": "v9_rank"},
        model=model,
    )
    prediction = adapter.predict(dataset, "predict")
    raw = np.array([1.0, 3.0, 2.0], dtype=float)
    expected = np.tanh((raw - raw.mean()) / (raw.std() + 1e-8))

    assert model.observed_risk_dim == 2
    scores = prediction.frame.sort_values("code")["score"].to_numpy()
    assert np.allclose(scores, expected)


def test_existing_strong_delegate_validates_fields_runs_frozen_command_and_loads_checkpoint(
    tmp_path, monkeypatch
):
    segments = {"train": DateRange.create("2023-01-01", "2023-01-31", field="train")}

    def factory(date_range):
        rows = 2
        yield {
            "date": "2023-01-03",
            "codes": ["A", "B"],
            "X": np.ones((rows, 2), dtype=np.float32),
            "y": np.zeros(rows, dtype=np.float32),
            "y_seq": np.zeros((rows, 4), dtype=np.float32),
            "raw_y_seq": np.zeros((rows, 4), dtype=np.float32),
            "lag1_y_seq": np.zeros((rows, 4), dtype=np.float32),
            "lag1_mask": np.ones(rows, dtype=bool),
            "risk": np.zeros((rows, 3), dtype=np.float32),
            "industry_ids": np.array([0, 1], dtype=np.int64),
        }

    dataset = ProjectDataset(
        DataHandlerRuntime(sample_factory=factory, segments=segments, processors=ProcessorChains())
    )
    checkpoint_path = tmp_path / "ultimate_v7_best.pt"
    command = ["python", "run/train.py", "--epochs", "1"]

    def fake_run(actual_command, *, cwd, min_free_gib):
        assert actual_command == command
        torch.save(
            {
                "epoch": 1,
                "arch_config": {"input_dim": 2},
                "best_metric": "pilot",
                "best_score": 0.2,
                "val_loss": 0.3,
            },
            checkpoint_path,
        )

    model = TinyCrossSectionModel()
    monkeypatch.setattr("experiments.strong_adapter_runtime.run_with_memory_guard", fake_run)
    monkeypatch.setattr(
        "experiments.strong_adapter_runtime.load_v9_checkpoint",
        lambda checkpoint, samples, cfg, device: (model, torch.device("cpu"), 2),
    )
    config = {}
    delegate = ExistingStrongTrainerDelegate(
        command=command,
        project_root=tmp_path,
        checkpoint_path=checkpoint_path,
        horizon_indices=(0, 1, 2, 3),
        expected_epoch=1,
        expected_input_dim=2,
        device="cpu",
        min_free_gib=0.5,
    )

    returned, metrics = delegate(dataset, config)

    assert returned is model
    assert config["regime_dim"] == 2
    assert metrics["epoch"] == 1
    assert metrics["sample_contract"] == {"rows": 2, "input_dim": 2, "risk_dim": 3}
    assert len(metrics["model_state_sha256"]) == 64

    monkeypatch.setattr(
        "experiments.strong_adapter_runtime.run_with_memory_guard",
        lambda *args, **kwargs: pytest.fail("a valid recovered exact checkpoint must not retrain"),
    )
    recovered = ExistingStrongTrainerDelegate(
        command=command,
        project_root=tmp_path,
        checkpoint_path=checkpoint_path,
        horizon_indices=(0, 1, 2, 3),
        expected_epoch=1,
        expected_input_dim=2,
        device="cpu",
        reuse_existing_checkpoint=True,
    )
    _, recovered_metrics = recovered(dataset, {})

    assert recovered_metrics["checkpoint_reused"] is True
    assert recovered_metrics["trainer_command_launched"] is False


def test_strong_stage_helper_runs_the_existing_trainer_through_torch_adapter(
    tmp_path, monkeypatch
):
    dataset = _dataset()
    observed = {}

    class FakeDelegate:
        def __call__(self, project_dataset, config):
            observed["dataset"] = project_dataset
            observed["config"] = dict(config)
            return TinyCrossSectionModel(), {
                "epoch": 6,
                "checkpoint_reused": False,
                "trainer_command_launched": True,
                "regime_dim": 2,
            }

    def build_delegate(**kwargs):
        observed["delegate_kwargs"] = kwargs
        return FakeDelegate()

    monkeypatch.setattr(
        "experiments.strong_adapter_runtime.ExistingStrongTrainerDelegate",
        build_delegate,
    )
    metrics = fit_strong_stage_through_adapter(
        dataset=dataset,
        stage={"target_epoch": 6, "exact_checkpoint": str(tmp_path / "epoch_006.pt")},
        command=["python", "run/train.py", "--epochs", "6"],
        project_root=tmp_path,
        horizon_indices=(0, 2, 4, 6),
        expected_input_dim=2,
        device="cpu",
        min_free_gib=0.75,
        model_id="oos_2024_01:base_e6",
    )

    assert observed["dataset"] is dataset
    assert observed["delegate_kwargs"]["reuse_existing_checkpoint"] is True
    assert observed["delegate_kwargs"]["expected_epoch"] == 6
    assert metrics["adapter"] == "torch_strong_alpha"
    assert metrics["trainer_command_launched"] is True
    assert len(metrics["adapter_contract_sha256"]) == 64


def test_one_model_spec_shape_switches_lightgbm_and_torch_adapters():
    common = {
        "config": {"device": "cpu"},
        "seed": 17,
        "resource_limits": {"ram_gb": 16, "gpu_vram_gb": 8, "threads": 4},
    }
    lgbm = create_model_adapter(
        {**common, "adapter": "rolling_lgbm_alpha"},
        model_id="same_workflow_lgbm",
    )
    torch_adapter = create_model_adapter(
        {**common, "adapter": "torch_strong_alpha"},
        model_id="same_workflow_torch",
        dependencies=ModelAdapterDependencies(model=TinyCrossSectionModel()),
    )

    assert isinstance(lgbm, LightGBMModelAdapter)
    assert isinstance(torch_adapter, TorchStrongAlphaAdapter)
    assert lgbm.config["seed"] == torch_adapter.config["seed"] == 17
    assert lgbm.config["resource_limits"] == torch_adapter.config["resource_limits"]


def test_frozen_factory_requires_explicit_runtime_artifact_paths(tmp_path):
    spec = {"adapter": "frozen_artifact", "config": {}, "seed": 1}
    with pytest.raises(ValueError, match="alpha_paths"):
        create_model_adapter(spec, model_id="frozen")

    alpha = write_alpha_rows(
        tmp_path / "factory_alpha.jsonl",
        [{"date": "2025-01-03", "codes": ["A", "B"], "alpha": [0.4, 0.2]}],
    )
    adapter = create_model_adapter(
        spec,
        model_id="frozen",
        dependencies=ModelAdapterDependencies(alpha_paths={"test": alpha}),
    )
    assert isinstance(adapter, FrozenArtifactAdapter)


def test_model_contract_has_no_execution_layer_dependency():
    source = (Path(__file__).resolve().parents[1] / "experiments" / "model_adapters.py").read_text(encoding="utf-8")
    assert "open_ledger" not in source
    assert "from backtest" not in source
    assert "import backtest" not in source
