import numpy as np
import pytest

from core.train_utils import PrecomputedMemmapDataset
from data.dataset_runtime import build_v14_rolling_dataset, build_v14_strong_rolling_dataset
from data.pipeline import _open_memmap
from data.providers import DataView, V14MemmapProvider
from data.rolling_samples import iter_rolling_samples
from run.rolling_lgbm_alpha import (
    collect_rows,
    predict_rows,
    resolve_dataset_runtime,
    resolve_model_runtime,
)


def _write_memmap(path, values, dtype=np.int16):
    out = np.memmap(path, dtype=dtype, mode="w+", shape=values.shape)
    out[:] = values
    out.flush()
    out._mmap.close()


def _provider(tmp_path):
    stocks, dates, features = 5, 6, 3
    x = np.empty((stocks, dates, features), dtype=np.int16)
    for stock in range(stocks):
        for day in range(dates):
            x[stock, day] = [100 + stock + day, 200 + 2 * stock + day, 300 + stock + 2 * day]
    risk = np.zeros((stocks, dates, 1), dtype=np.int16)
    labels = np.empty((stocks, dates, 3), dtype=np.int16)
    for stock in range(stocks):
        for horizon in range(3):
            labels[stock, :, horizon] = 10 + stock + np.arange(dates) + horizon * 20
    raw_labels = labels.astype(np.float32) / 100.0
    x_path, risk_path = tmp_path / "x.dat", tmp_path / "risk.dat"
    label_path, raw_label_path, scalar_path = tmp_path / "y.dat", tmp_path / "y_raw.dat", tmp_path / "scalar.dat"
    _write_memmap(x_path, x)
    _write_memmap(risk_path, risk)
    _write_memmap(label_path, labels)
    _write_memmap(raw_label_path, raw_labels, dtype=np.float32)
    _write_memmap(scalar_path, labels[:, :, 0])
    all_dates = [f"2024-01-{day:02d}" for day in range(2, 8)]
    meta_path = tmp_path / "meta.pkl"
    meta_path.write_bytes(b"rolling-runtime-parity")
    meta = {
        "all_codes": [f"S{stock}" for stock in range(stocks)],
        "all_dates": all_dates,
        "x_dim": features,
        "risk_full_dim": 1,
        "max_horizon": 3,
        "min_stocks": 1,
        "x_norm_path": x_path,
        "risk_full_path": risk_path,
        "y_norm_path": scalar_path,
        "industry_array": np.zeros((stocks, dates), dtype=np.int16),
        "label_families": {
            "oo": {"norm_path": label_path, "raw_path": raw_label_path, "date_shift": 0},
            "oo_lag1": {"alias_of": "oo", "date_shift": 1},
        },
    }
    raw = tmp_path / "raw"
    raw.mkdir()
    view = DataView.create(
        name="rolling-runtime-parity",
        physical_root=raw,
        feature_warmup_start=all_dates[0],
        feature_warmup_end=all_dates[0],
        task_start=all_dates[0],
        task_end=all_dates[-1],
        evaluation_start=all_dates[4],
        evaluation_end=all_dates[-1],
        max_data_date=all_dates[-1],
    )
    return meta, V14MemmapProvider(meta=meta, meta_path=meta_path, data_view=view)


def _dataset(meta, provider):
    indices = {"train": [0, 1], "valid": [2, 3], "predict": [4, 5]}
    dataset = build_v14_rolling_dataset(
        provider=provider,
        segment_indices=indices,
        label_family="oo",
        horizon_index=0,
        feature_indices=[2, 0],
    )
    dataset.fit()
    return indices, dataset


def test_project_dataset_stream_matches_legacy_samples_and_sampling(tmp_path):
    meta, provider = _provider(tmp_path)
    indices, dataset = _dataset(meta, provider)

    for segment, segment_indices in indices.items():
        legacy = list(
            iter_rolling_samples(
                meta,
                segment_indices,
                "oo",
                0,
                [2, 0],
                include_risk=False,
                include_industry=False,
            )
        )
        runtime = list(dataset.prepare(segment, data_key="raw"))
        assert len(runtime) == len(legacy)
        for left, right in zip(legacy, runtime):
            assert left["time_index"] == right["time_index"]
            assert left["date"] == right["date"]
            assert left["codes"] == right["codes"]
            assert np.array_equal(left["X"], right["X"])
            assert np.array_equal(left["y"], right["y"])

    legacy_x, legacy_y = collect_rows(meta, indices["train"], "oo", 0, 7, 19, [2, 0])
    runtime_x, runtime_y = collect_rows(
        meta,
        indices["train"],
        "oo",
        0,
        7,
        19,
        [2, 0],
        dataset=dataset,
        segment="train",
    )
    assert np.array_equal(runtime_x, legacy_x)
    assert np.array_equal(runtime_y, legacy_y)


def test_project_dataset_prediction_stream_matches_legacy(tmp_path):
    meta, provider = _provider(tmp_path)
    indices, dataset = _dataset(meta, provider)

    class SumModel:
        best_iteration = 1

        @staticmethod
        def predict(values, num_iteration=None):
            return np.asarray(values).sum(axis=1)

    legacy = predict_rows(meta, indices["predict"], "oo", 0, SumModel(), [2, 0])
    runtime = predict_rows(
        meta,
        indices["predict"],
        "oo",
        0,
        SumModel(),
        [2, 0],
        dataset=dataset,
    )
    assert runtime == legacy


def test_strong_project_dataset_matches_precomputed_memmap_fields(tmp_path):
    meta, provider = _provider(tmp_path)
    indices = {"train": [0, 1], "valid": [2, 3], "predict": [4, 5]}
    dataset = build_v14_strong_rolling_dataset(
        provider=provider,
        segment_indices=indices,
        label_family="oo",
        horizon_indices=(0, 2),
        target_horizon_index=1,
        include_raw_returns=True,
        include_lag1_labels=True,
        lag1_label_family="oo_lag1",
    )
    dataset.fit()
    runtime = list(dataset.prepare("train", data_key="raw"))

    shape = (5, 6, 3)
    x_mm = _open_memmap(meta["x_norm_path"], np.int16, (5, 6, 3))
    risk_mm = _open_memmap(meta["risk_full_path"], np.int16, (5, 6, 1))
    scalar_mm = _open_memmap(meta["y_norm_path"], np.int16, (5, 6))
    labels_mm = _open_memmap(meta["label_families"]["oo"]["norm_path"], np.int16, shape)
    raw_mm = _open_memmap(meta["label_families"]["oo"]["raw_path"], np.float32, shape)
    legacy = PrecomputedMemmapDataset(
        x_mm,
        risk_mm,
        scalar_mm,
        labels_mm,
        meta["industry_array"],
        meta["all_codes"],
        meta["all_dates"],
        indices["train"],
        n_industries=1,
        max_horizon=3,
        min_stocks=1,
        raw_ret_mm=raw_mm,
        include_lag1_labels=True,
        horizon_indices=(0, 2),
        target_horizon_index=1,
        label_date_shift=0,
        lag1_y_seq_norm_mm=labels_mm,
        lag1_label_date_shift=1,
    )
    assert len(runtime) == len(legacy)
    for expected, actual in zip((legacy[i] for i in range(len(legacy))), runtime):
        for key in ("X", "y", "y_seq", "raw_y_seq", "risk", "industry_ids", "lag1_y_seq", "lag1_mask"):
            assert np.array_equal(expected[key].numpy(), actual[key])


def test_rolling_dataset_rejects_noncontiguous_or_unordered_indices(tmp_path):
    _, provider = _provider(tmp_path)
    with pytest.raises(ValueError, match="contiguous"):
        build_v14_rolling_dataset(
            provider=provider,
            segment_indices={"train": [0, 2]},
            label_family="oo",
            horizon_index=0,
        )
    with pytest.raises(ValueError, match="ordered and unique"):
        build_v14_rolling_dataset(
            provider=provider,
            segment_indices={"train": [1, 0]},
            label_family="oo",
            horizon_index=0,
        )


def test_dataset_runtime_requires_explicit_supported_value():
    assert resolve_dataset_runtime({"data": {}}) == "legacy_iter"
    assert resolve_dataset_runtime({"data": {"dataset_runtime": "project_dataset"}}) == "project_dataset"
    with pytest.raises(ValueError, match="unsupported"):
        resolve_dataset_runtime({"data": {"dataset_runtime": "silent_fallback"}})


def test_model_runtime_requires_explicit_supported_value():
    assert resolve_model_runtime({"model": {}}) == "legacy_trainer"
    assert resolve_model_runtime({"model": {"runtime": "project_adapter"}}) == "project_adapter"
    with pytest.raises(ValueError, match="unsupported"):
        resolve_model_runtime({"model": {"runtime": "silent_fallback"}})
