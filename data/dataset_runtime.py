"""Low-memory Dataset/DataHandler/Processor runtime inspired by Qlib.

The runtime operates on one cross-sectional sample at a time. It deliberately
does not materialize a full v14 date range in memory.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from data.providers import DateRange, V14MemmapProvider
from data.rolling_samples import iter_strong_rolling_samples, iter_v14_inference_samples
from experiments.recording import canonical_json_hash


Sample = dict[str, Any]
SampleFactory = Callable[[DateRange], Iterable[Sample]]
VALID_DATA_KEYS = {"raw", "infer", "learn"}
VALID_COL_SETS = {"all", "feature", "label"}


def _copy_sample(sample: Mapping[str, Any]) -> Sample:
    result = dict(sample)
    for key, value in list(result.items()):
        if isinstance(value, np.ndarray):
            result[key] = value.copy()
        elif isinstance(value, list):
            result[key] = list(value)
    return result


class SampleProcessor(ABC):
    """One auditable transformation over a streamed cross section."""

    requires_fit = False
    usable_for_inference = True
    kind = "stateless"

    def __init__(self, *, name: str, config: Mapping[str, Any] | None = None):
        self.name = str(name).strip()
        if not self.name:
            raise ValueError("processor name is required")
        self.config = dict(config or {})
        self._fitted = not self.requires_fit

    @property
    def fitted(self) -> bool:
        return bool(self._fitted)

    def fit(self, samples: Iterable[Sample]) -> None:
        if self.requires_fit:
            raise NotImplementedError(f"processor {self.name} must implement fit")
        self._fitted = True

    @abstractmethod
    def transform(self, sample: Sample) -> Sample:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError(f"stateless processor {self.name} received unexpected state")
        self._fitted = True

    def manifest(self) -> dict[str, Any]:
        state = self.state_dict()
        payload = {
            "class": type(self).__name__,
            "name": self.name,
            "kind": self.kind,
            "config": self.config,
            "requires_fit": self.requires_fit,
            "usable_for_inference": self.usable_for_inference,
            "fitted": self.fitted,
            "state": state,
        }
        payload["state_sha256"] = canonical_json_hash(state)
        return payload


class IdentityProcessor(SampleProcessor):
    kind = "pit_alignment"

    def transform(self, sample: Sample) -> Sample:
        return sample


class CrossSectionRankProcessor(SampleProcessor):
    """Date-local percentile rank that never fits across dates."""

    kind = "daily_cross_section"

    def transform(self, sample: Sample) -> Sample:
        values = np.asarray(sample["X"], dtype=np.float32)
        ranked = np.zeros_like(values, dtype=np.float32)
        for column in range(values.shape[1]):
            vector = values[:, column]
            finite = np.isfinite(vector)
            count = int(finite.sum())
            if count <= 1:
                ranked[finite, column] = 0.0
                continue
            order = np.argsort(vector[finite], kind="mergesort")
            ranks = np.empty(count, dtype=np.float32)
            ranks[order] = np.arange(count, dtype=np.float32)
            ranked[finite, column] = ranks / float(count - 1) - 0.5
        sample["X"] = ranked
        return sample


class FeatureClipProcessor(SampleProcessor):
    kind = "daily_cross_section"

    def transform(self, sample: Sample) -> Sample:
        lower = float(self.config.get("lower", -5.0))
        upper = float(self.config.get("upper", 5.0))
        if lower >= upper:
            raise ValueError("feature clip lower must be below upper")
        sample["X"] = np.clip(np.asarray(sample["X"], dtype=np.float32), lower, upper)
        return sample


class DropInvalidLabelProcessor(SampleProcessor):
    usable_for_inference = False
    kind = "learning_only"

    def transform(self, sample: Sample) -> Sample:
        if "y" not in sample:
            return sample
        labels = np.asarray(sample["y"])
        keep = np.isfinite(labels)
        for key in (
            "X",
            "y",
            "y_seq",
            "raw_y_seq",
            "lag1_y_seq",
            "lag1_mask",
            "risk",
            "industry_ids",
        ):
            if key in sample:
                sample[key] = np.asarray(sample[key])[keep]
        if "codes" in sample:
            sample["codes"] = [code for code, valid in zip(sample["codes"], keep) if valid]
        return sample


class StreamingFeatureStandardizer(SampleProcessor):
    """Fit feature mean/std on streamed Train cross sections only."""

    requires_fit = True
    kind = "train_fitted"

    def __init__(self, *, name: str, config: Mapping[str, Any] | None = None):
        super().__init__(name=name, config=config)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.count: np.ndarray | None = None

    def fit(self, samples: Iterable[Sample]) -> None:
        if self.fitted:
            raise RuntimeError(f"processor {self.name} is already fitted")
        sums = sums_sq = counts = None
        for sample in samples:
            values = np.asarray(sample["X"], dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("feature matrix X must be two-dimensional")
            finite = np.isfinite(values)
            if sums is None:
                width = values.shape[1]
                sums = np.zeros(width, dtype=np.float64)
                sums_sq = np.zeros(width, dtype=np.float64)
                counts = np.zeros(width, dtype=np.int64)
            if values.shape[1] != len(sums):
                raise ValueError("feature width changed during processor fit")
            safe = np.where(finite, values, 0.0)
            sums += safe.sum(axis=0)
            sums_sq += np.square(safe).sum(axis=0)
            counts += finite.sum(axis=0)
        if sums is None or np.any(counts == 0):
            raise ValueError("processor fit received no finite feature data")
        mean = sums / counts
        variance = np.maximum(sums_sq / counts - np.square(mean), 0.0)
        minimum_scale = float(self.config.get("minimum_scale", 1e-6))
        self.mean = mean.astype(np.float32)
        self.scale = np.maximum(np.sqrt(variance), minimum_scale).astype(np.float32)
        self.count = counts
        self._fitted = True

    def transform(self, sample: Sample) -> Sample:
        if not self.fitted or self.mean is None or self.scale is None:
            raise RuntimeError(f"processor {self.name} has no fitted state")
        values = np.asarray(sample["X"], dtype=np.float32)
        if values.shape[1] != len(self.mean):
            raise ValueError("feature width does not match fitted processor state")
        transformed = (values - self.mean) / self.scale
        fill_value = float(self.config.get("nonfinite_fill", 0.0))
        sample["X"] = np.where(np.isfinite(transformed), transformed, fill_value).astype(np.float32)
        return sample

    def state_dict(self) -> dict[str, Any]:
        if not self.fitted:
            return {}
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count.tolist(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.scale = np.asarray(state["scale"], dtype=np.float32)
        self.count = np.asarray(state["count"], dtype=np.int64)
        if not (len(self.mean) == len(self.scale) == len(self.count)):
            raise ValueError("standardizer state dimensions disagree")
        self._fitted = True


def processor_from_config(value: Mapping[str, Any]) -> SampleProcessor:
    name = str(value["name"])
    config = value.get("config", {})
    factories = {
        "pit_availability": IdentityProcessor,
        "identity": IdentityProcessor,
        "daily_cross_section_rank": CrossSectionRankProcessor,
        "feature_clip": FeatureClipProcessor,
        "feature_standardize": StreamingFeatureStandardizer,
        "drop_invalid_label": DropInvalidLabelProcessor,
    }
    try:
        cls = factories[name]
    except KeyError as exc:
        raise ValueError(f"unknown processor: {name}") from exc
    processor = cls(name=name, config=config)
    declared_kind = value.get("kind")
    if declared_kind is not None and str(declared_kind) != processor.kind:
        raise ValueError(
            f"processor {name} declares kind={declared_kind!r}, expected {processor.kind!r}"
        )
    return processor


@dataclass(frozen=True)
class ProcessorChains:
    shared: tuple[SampleProcessor, ...] = ()
    infer: tuple[SampleProcessor, ...] = ()
    learn: tuple[SampleProcessor, ...] = ()

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ProcessorChains":
        return cls(
            shared=tuple(processor_from_config(item) for item in value.get("shared", [])),
            infer=tuple(processor_from_config(item) for item in value.get("infer", [])),
            learn=tuple(processor_from_config(item) for item in value.get("learn", [])),
        )

    def for_key(self, data_key: str) -> tuple[SampleProcessor, ...]:
        if data_key == "raw":
            return ()
        if data_key == "infer":
            chain = self.shared + self.infer
        elif data_key == "learn":
            chain = self.shared + self.infer + self.learn
        else:
            raise ValueError(f"unknown data_key: {data_key}")
        if data_key == "infer":
            invalid = [processor.name for processor in chain if not processor.usable_for_inference]
            if invalid:
                raise ValueError(f"inference chain contains learning-only processors: {invalid}")
        return chain

    def unique_ordered(self) -> tuple[SampleProcessor, ...]:
        result = []
        seen = set()
        for processor in self.shared + self.infer + self.learn:
            if id(processor) not in seen:
                result.append(processor)
                seen.add(id(processor))
        return tuple(result)


class DataHandlerRuntime:
    """Stream segments through frozen shared/infer/learn processor chains."""

    def __init__(
        self,
        *,
        sample_factory: SampleFactory,
        segments: Mapping[str, DateRange],
        processors: ProcessorChains,
    ):
        self.sample_factory = sample_factory
        self.segments = dict(segments)
        self.processors = processors
        if "train" not in self.segments:
            raise ValueError("dataset segments require train")
        self.fit_segment: str | None = None

    def _range(self, segment: str) -> DateRange:
        try:
            return self.segments[str(segment)]
        except KeyError as exc:
            raise ValueError(f"unknown dataset segment: {segment}") from exc

    @staticmethod
    def _apply(sample: Sample, processors: Sequence[SampleProcessor]) -> Sample:
        if not processors:
            # Provider samples already own their decoded arrays. Preserve a
            # separate mapping for col_set projection without copying tens of
            # megabytes of arrays when no processor can mutate them.
            return dict(sample)
        result = _copy_sample(sample)
        for processor in processors:
            if processor.requires_fit and not processor.fitted:
                raise RuntimeError(f"processor {processor.name} has no fitted Train state")
            result = processor.transform(result)
        return result

    def fit(self, segment: str = "train") -> None:
        if segment != "train":
            raise ValueError("train-fitted processors may only fit on the train segment")
        if self.fit_segment is not None:
            raise RuntimeError("handler processors are already fitted")
        date_range = self._range(segment)
        prior: list[SampleProcessor] = []
        for processor in self.processors.unique_ordered():
            if processor.requires_fit:
                def samples(prior_chain=tuple(prior)):
                    for raw in self.sample_factory(date_range):
                        yield self._apply(raw, prior_chain)

                processor.fit(samples())
            prior.append(processor)
        self.fit_segment = segment

    def prepare(self, segment: str, *, col_set: str = "all", data_key: str = "infer") -> Iterable[Sample]:
        if data_key not in VALID_DATA_KEYS:
            raise ValueError(f"unknown data_key: {data_key}")
        if col_set not in VALID_COL_SETS:
            raise ValueError(f"unknown col_set: {col_set}")
        chain = self.processors.for_key(data_key)
        date_range = self._range(segment)
        for raw in self.sample_factory(date_range):
            sample = self._apply(raw, chain)
            if col_set == "feature":
                sample.pop("y", None)
            elif col_set == "label":
                sample = {key: value for key, value in sample.items() if key in {"time_index", "date", "codes", "y"}}
            yield sample

    def state_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "fit_segment": self.fit_segment,
            "fit_range": self.segments["train"].to_dict() if self.fit_segment else None,
            "processors": [processor.manifest() for processor in self.processors.unique_ordered()],
            "chains": {
                "shared": [processor.name for processor in self.processors.shared],
                "infer": [processor.name for processor in self.processors.infer],
                "learn": [processor.name for processor in self.processors.learn],
            },
        }
        payload["state_sha256"] = canonical_json_hash(payload)
        return payload

    def save_state(self, path: str | Path) -> Path:
        if any(processor.requires_fit and not processor.fitted for processor in self.processors.unique_ordered()):
            raise RuntimeError("cannot save an unfitted processor chain")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.state_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target

    def load_state(self, path: str | Path) -> None:
        if self.fit_segment is not None:
            raise RuntimeError("handler state is already initialized")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        processors = self.processors.unique_ordered()
        saved = payload.get("processors", [])
        if [processor.name for processor in processors] != [item.get("name") for item in saved]:
            raise ValueError("processor state order does not match handler configuration")
        for processor, item in zip(processors, saved):
            if item.get("class") != type(processor).__name__ or item.get("config") != processor.config:
                raise ValueError(f"processor state contract differs for {processor.name}")
            processor.load_state_dict(item.get("state", {}))
        self.fit_segment = str(payload.get("fit_segment") or "train")
        expected = payload.get("state_sha256")
        actual_payload = dict(payload)
        actual_payload.pop("state_sha256", None)
        if expected != canonical_json_hash(actual_payload):
            raise ValueError("processor state hash mismatch")


class ProjectDataset:
    """DatasetH-like named segment facade."""

    def __init__(self, handler: DataHandlerRuntime):
        self.handler = handler

    def fit(self) -> None:
        self.handler.fit("train")

    def prepare(self, segment: str, *, col_set: str = "all", data_key: str = "infer") -> Iterable[Sample]:
        return self.handler.prepare(segment, col_set=col_set, data_key=data_key)


def build_v14_dataset(
    *,
    provider: V14MemmapProvider,
    segments: Mapping[str, DateRange],
    processors: ProcessorChains,
    label_family: str,
    horizon_index: int,
    feature_indices=None,
    include_risk: bool = True,
    include_industry: bool = True,
) -> ProjectDataset:
    def sample_factory(date_range: DateRange):
        return provider.iter_samples(
            start_date=date_range.start,
            end_date=date_range.end,
            label_family=label_family,
            horizon_index=horizon_index,
            feature_indices=feature_indices,
            include_risk=include_risk,
            include_industry=include_industry,
        )

    return ProjectDataset(
        DataHandlerRuntime(sample_factory=sample_factory, segments=segments, processors=processors)
    )


def build_v14_rolling_dataset(
    *,
    provider: V14MemmapProvider,
    segment_indices: Mapping[str, Sequence[int]],
    label_family: str,
    horizon_index: int,
    feature_indices=None,
    processors: ProcessorChains | None = None,
    include_risk: bool = False,
    include_industry: bool = False,
) -> ProjectDataset:
    """Build named Dataset segments from already purged rolling indices.

    Rolling owns label-tail purge. This bridge refuses to infer broader date
    ranges when the supplied indices are not one contiguous calendar slice,
    preventing Dataset construction from silently restoring purged tail dates.
    """

    dates = provider.dates
    segments: dict[str, DateRange] = {}
    for name, raw_indices in segment_indices.items():
        indices = [int(value) for value in raw_indices]
        if not indices:
            raise ValueError(f"rolling Dataset segment {name} is empty")
        if indices != sorted(set(indices)):
            raise ValueError(f"rolling Dataset segment {name} indices must be ordered and unique")
        if indices[0] < 0 or indices[-1] >= len(dates):
            raise ValueError(f"rolling Dataset segment {name} indices exceed provider coverage")
        start, end = dates[indices[0]], dates[indices[-1]]
        provider_indices = provider.date_indices(start, end)
        if provider_indices != indices:
            raise ValueError(
                f"rolling Dataset segment {name} is not a contiguous provider calendar slice"
            )
        segments[str(name)] = DateRange.create(start, end, field=f"rolling.{name}")
    return build_v14_dataset(
        provider=provider,
        segments=segments,
        processors=processors or ProcessorChains(),
        label_family=label_family,
        horizon_index=horizon_index,
        feature_indices=feature_indices,
        include_risk=include_risk,
        include_industry=include_industry,
    )


def build_v14_strong_rolling_dataset(
    *,
    provider: V14MemmapProvider,
    segment_indices: Mapping[str, Sequence[int]],
    label_family: str,
    horizon_indices: Sequence[int],
    target_horizon_index: int,
    include_raw_returns: bool,
    include_lag1_labels: bool,
    lag1_label_family: str | None = None,
    processors: ProcessorChains | None = None,
    inference_segments: Sequence[str] = ("predict",),
) -> ProjectDataset:
    """Build strong-model Dataset segments without changing legacy sample fields."""

    dates = provider.dates
    segments: dict[str, DateRange] = {}
    for name, raw_indices in segment_indices.items():
        indices = [int(value) for value in raw_indices]
        if not indices:
            raise ValueError(f"strong rolling Dataset segment {name} is empty")
        if indices != sorted(set(indices)):
            raise ValueError(f"strong rolling Dataset segment {name} indices must be ordered and unique")
        start, end = dates[indices[0]], dates[indices[-1]]
        if provider.date_indices(start, end) != indices:
            raise ValueError(f"strong rolling Dataset segment {name} is not contiguous")
        segments[str(name)] = DateRange.create(start, end, field=f"strong_rolling.{name}")

    inference_names = {str(value) for value in inference_segments}
    range_modes = {
        (value.start, value.end): name in inference_names for name, value in segments.items()
    }

    def sample_factory(date_range: DateRange):
        indices = provider.date_indices(date_range.start, date_range.end)
        if range_modes.get((date_range.start, date_range.end), False):
            return iter_v14_inference_samples(provider.meta, indices)
        return iter_strong_rolling_samples(
            provider.meta,
            indices,
            label_family,
            horizon_indices=horizon_indices,
            target_horizon_index=target_horizon_index,
            include_raw_returns=include_raw_returns,
            include_lag1_labels=include_lag1_labels,
            lag1_label_family=lag1_label_family,
        )

    return ProjectDataset(
        DataHandlerRuntime(
            sample_factory=sample_factory,
            segments=segments,
            processors=processors or ProcessorChains(),
        )
    )


def build_v14_dataset_from_workflow(
    config: Mapping[str, Any],
    *,
    provider: V14MemmapProvider,
    feature_indices=None,
) -> ProjectDataset:
    """Build the Q2 runtime from a validated Workflow v2 mapping."""

    if config.get("schema_version") != 2:
        raise ValueError("workflow-driven Dataset runtime requires schema_version=2")
    dataset = config["dataset"]
    if dataset.get("adapter") != "project_v14_memmap":
        raise ValueError("build_v14_dataset_from_workflow requires project_v14_memmap")
    processor_config = config["processors"]
    if processor_config.get("state_policy") != "fit_train_freeze_elsewhere":
        raise ValueError("workflow processor state_policy must freeze Train state elsewhere")
    segments = {
        name: DateRange.create(value["start"], value["end"], field=f"dataset.segments.{name}")
        for name, value in dataset["segments"].items()
    }
    label = dataset["label"]
    return build_v14_dataset(
        provider=provider,
        segments=segments,
        processors=ProcessorChains.from_config(processor_config),
        label_family=label["family"],
        horizon_index=int(label["horizon_index"]),
        feature_indices=feature_indices,
    )
