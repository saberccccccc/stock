import json
from pathlib import Path

import numpy as np
import pytest

from data.dataset_runtime import (
    DataHandlerRuntime,
    ProcessorChains,
    ProjectDataset,
    StreamingFeatureStandardizer,
    build_v14_dataset_from_workflow,
    processor_from_config,
)
from data.providers import DataView, DateRange, V14MemmapProvider


def _segments():
    return {
        "train": DateRange.create("2023-01-01", "2023-12-31", field="train"),
        "valid": DateRange.create("2024-01-01", "2024-12-31", field="valid"),
    }


def _factory(calls):
    def create(date_range):
        calls.append(date_range.to_dict())
        if str(date_range.start.date()).startswith("2023"):
            rows = [
                {"date": "2023-01-03", "codes": ["A", "B"], "X": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), "y": np.array([0.1, 0.2])},
                {"date": "2023-01-04", "codes": ["A", "B"], "X": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), "y": np.array([0.3, 0.4])},
            ]
        else:
            rows = [
                {"date": "2024-01-03", "codes": ["A", "B"], "X": np.array([[5.0, 6.0], [9.0, 10.0]], dtype=np.float32), "y": np.array([0.5, np.nan])}
            ]
        yield from rows
    return create


def _handler(calls):
    return DataHandlerRuntime(
        sample_factory=_factory(calls),
        segments=_segments(),
        processors=ProcessorChains.from_config(
            {
                "shared": [{"name": "feature_standardize", "config": {"minimum_scale": 1e-6}}],
                "infer": [],
                "learn": [{"name": "drop_invalid_label", "config": {}}],
            }
        ),
    )


def test_train_fit_is_streamed_and_valid_only_applies_frozen_state():
    calls = []
    handler = _handler(calls)
    dataset = ProjectDataset(handler)

    dataset.fit()
    valid = list(dataset.prepare("valid", data_key="infer"))

    standardizer = handler.processors.shared[0]
    assert isinstance(standardizer, StreamingFeatureStandardizer)
    assert np.allclose(standardizer.mean, [4.0, 5.0])
    assert calls[0]["start"] == "2023-01-01"
    assert calls[-1]["start"] == "2024-01-01"
    assert np.allclose(valid[0]["X"][0], [(5.0 - 4.0) / np.sqrt(5.0), (6.0 - 5.0) / np.sqrt(5.0)])
    with pytest.raises(RuntimeError, match="already fitted"):
        dataset.fit()


def test_unfitted_processor_cannot_transform_valid():
    handler = _handler([])

    with pytest.raises(RuntimeError, match="no fitted Train state"):
        list(handler.prepare("valid", data_key="infer"))


def test_learn_chain_drops_invalid_labels_but_infer_keeps_rows():
    handler = _handler([])
    handler.fit()

    infer = next(iter(handler.prepare("valid", data_key="infer")))
    learn = next(iter(handler.prepare("valid", data_key="learn")))

    assert infer["codes"] == ["A", "B"]
    assert learn["codes"] == ["A"]
    assert learn["X"].shape[0] == 1


def test_processor_state_roundtrip_is_frozen_and_hash_checked(tmp_path):
    handler = _handler([])
    handler.fit()
    state_path = handler.save_state(tmp_path / "processor_state.json")
    first_payload = json.loads(state_path.read_text(encoding="utf-8"))

    restored = _handler([])
    restored.load_state(state_path)
    second_payload = restored.state_payload()

    assert first_payload["state_sha256"] == second_payload["state_sha256"]
    assert np.allclose(
        next(iter(handler.prepare("valid", data_key="infer")))["X"],
        next(iter(restored.prepare("valid", data_key="infer")))["X"],
    )


def test_dataset_prepare_col_sets_and_processor_factory_errors():
    handler = _handler([])
    handler.fit()

    features = next(iter(handler.prepare("valid", col_set="feature", data_key="infer")))
    labels = next(iter(handler.prepare("valid", col_set="label", data_key="raw")))

    assert "X" in features and "y" not in features
    assert set(labels) == {"date", "codes", "y"}
    with pytest.raises(ValueError, match="unknown processor"):
        processor_from_config({"name": "does_not_exist", "config": {}})
    with pytest.raises(ValueError, match="declares kind"):
        processor_from_config({"name": "feature_standardize", "kind": "daily_cross_section", "config": {}})
    rank = processor_from_config({"name": "daily_cross_section_rank", "kind": "daily_cross_section", "config": {}})
    assert rank.kind == "daily_cross_section"


def test_prepare_is_lazy_and_fit_is_train_only():
    calls = []
    handler = DataHandlerRuntime(
        sample_factory=_factory(calls),
        segments=_segments(),
        processors=ProcessorChains(),
    )

    stream = handler.prepare("valid", data_key="raw")
    assert calls == []
    first = next(iter(stream))
    assert first["date"] == "2024-01-03"
    assert len(calls) == 1
    with pytest.raises(ValueError, match="only fit on the train"):
        handler.fit("valid")


def test_empty_processor_chain_does_not_copy_provider_arrays():
    values = np.array([[1.0, 2.0]], dtype=np.float32)

    def factory(date_range):
        yield {
            "date": "2023-01-03",
            "codes": ["A"],
            "X": values,
            "y": np.array([0.1], dtype=np.float32),
        }

    handler = DataHandlerRuntime(
        sample_factory=factory,
        segments={"train": DateRange.create("2023-01-01", "2023-12-31", field="train")},
        processors=ProcessorChains(),
    )

    sample = next(iter(handler.prepare("train", data_key="raw")))

    assert sample["X"] is values


def _write_memmap(path, values, dtype):
    out = np.memmap(path, dtype=dtype, mode="w+", shape=values.shape)
    out[:] = values
    out.flush()
    out._mmap.close()


def test_workflow_v2_builds_streaming_dataset_over_real_v14_provider(tmp_path):
    sentinel = np.int16(-32768)
    x = np.full((2, 3, 2), sentinel, dtype=np.int16)
    x[:, :, :] = np.array([100, 200], dtype=np.int16)
    risk = np.zeros((2, 3, 1), dtype=np.int16)
    labels = np.full((2, 3, 1), sentinel, dtype=np.int16)
    labels[:, 0, 0] = [10, 20]
    labels[:, 1, 0] = [30, 40]
    x_path, risk_path, label_path = tmp_path / "x.dat", tmp_path / "risk.dat", tmp_path / "y.dat"
    _write_memmap(x_path, x, np.int16)
    _write_memmap(risk_path, risk, np.int16)
    _write_memmap(label_path, labels, np.int16)
    raw = tmp_path / "raw"
    raw.mkdir()
    meta_path = tmp_path / "meta.bin"
    meta_path.write_bytes(b"v14-test")
    meta = {
        "all_codes": ["A", "B"],
        "all_dates": ["2023-01-03", "2024-01-03", "2025-01-03"],
        "x_dim": 2,
        "risk_full_dim": 1,
        "max_horizon": 1,
        "min_stocks": 1,
        "x_norm_path": x_path,
        "risk_full_path": risk_path,
        "industry_array": np.zeros((2, 3), dtype=np.int16),
        "label_families": {"oo_lag1": {"norm_path": label_path, "date_shift": 0}},
    }
    view = DataView.create(
        name="q2-test",
        physical_root=raw,
        feature_warmup_start="2023-01-03",
        feature_warmup_end="2023-01-03",
        task_start="2023-01-03",
        task_end="2024-12-31",
        evaluation_start="2024-01-01",
        evaluation_end="2024-12-31",
        max_data_date="2024-12-31",
    )
    provider = V14MemmapProvider(meta=meta, meta_path=meta_path, data_view=view)
    config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "workflow_v2_golden.json").read_text(encoding="utf-8"))
    config["dataset"]["segments"] = {
        "train": {"start": "2023-01-03", "end": "2023-12-31"},
        "valid": {"start": "2024-01-01", "end": "2024-12-31"},
    }
    config["dataset"]["label"] = {"family": "oo_lag1", "horizon_index": 0, "tail_purge_days": 1}
    config["processors"] = {
        "shared": [{"name": "identity", "kind": "pit_alignment", "config": {}}],
        "infer": [],
        "learn": [],
        "state_policy": "fit_train_freeze_elsewhere",
    }

    dataset = build_v14_dataset_from_workflow(config, provider=provider)
    dataset.fit()
    valid = list(dataset.prepare("valid", data_key="infer"))

    assert len(valid) == 1
    assert valid[0]["date"] == "2024-01-03"
    assert np.allclose(valid[0]["X"], 0.1 * np.array([[1, 2], [1, 2]], dtype=np.float32))
    assert np.allclose(valid[0]["y"], [0.03, 0.04])
