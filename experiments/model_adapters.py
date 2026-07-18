"""Project-native Qlib-like Model adapters and one prediction contract."""

from __future__ import annotations

import gc
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from alpha.io import load_alpha_rows, resolve_row_scores, write_alpha_rows
from data.dataset_runtime import ProjectDataset
from experiments.recording import canonical_json_hash, sha256_file


PREDICTION_COLUMNS = ("trade_date", "code", "score", "model_id", "asof_time")
MODEL_ADAPTER_NAMES = (
    "rolling_lgbm_alpha",
    "torch_strong_alpha",
    "frozen_artifact",
    "legacy_read_only",
)


@dataclass(frozen=True)
class FitResult:
    model_id: str
    status: str
    metrics: Mapping[str, Any]
    checkpoint_rule: str


@dataclass(frozen=True)
class ModelAdapterDependencies:
    """Runtime-only dependencies kept out of the immutable Workflow config."""

    alpha_paths: Mapping[str, str | Path] | None = None
    model: Any = None
    fit_delegate: Callable[[ProjectDataset, Mapping[str, Any]], tuple[Any, Mapping[str, Any]]] | None = None
    model_factory: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None = None


class PredictionFrame:
    """Validated long-form model output, independent from portfolio execution."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = self._validate(frame)

    @staticmethod
    def _validate(frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("PredictionFrame requires a pandas DataFrame")
        missing = set(PREDICTION_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"PredictionFrame missing columns: {sorted(missing)}")
        result = frame.loc[:, PREDICTION_COLUMNS].copy()
        if "_source_order" in frame.columns:
            source_order = pd.to_numeric(frame["_source_order"], errors="coerce")
            if source_order.isna().any():
                raise ValueError("PredictionFrame source order must be numeric")
            result["_source_order"] = source_order.astype(np.int64)
        else:
            result["_source_order"] = np.arange(len(result), dtype=np.int64)
        result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
        result["asof_time"] = pd.to_datetime(result["asof_time"], errors="coerce")
        result["code"] = result["code"].astype(str)
        result["model_id"] = result["model_id"].astype(str)
        result["score"] = pd.to_numeric(result["score"], errors="coerce").astype(float)
        if result.empty:
            raise ValueError("PredictionFrame cannot be empty")
        if result[["trade_date", "asof_time", "code", "model_id", "score"]].isna().any().any():
            raise ValueError("PredictionFrame contains missing or invalid values")
        if not np.isfinite(result["score"].to_numpy()).all():
            raise ValueError("PredictionFrame scores must be finite")
        if (result["code"].str.len() == 0).any() or (result["model_id"].str.len() == 0).any():
            raise ValueError("PredictionFrame code/model_id must be non-empty")
        duplicate = result.duplicated(["trade_date", "code", "model_id"])
        if duplicate.any():
            row = result.loc[duplicate, ["trade_date", "code", "model_id"]].iloc[0].to_dict()
            raise ValueError(f"duplicate PredictionFrame key: {row}")
        asof_dates = result["asof_time"].dt.tz_localize(None).dt.normalize()
        if (asof_dates > result["trade_date"]).any():
            raise ValueError("PredictionFrame asof_time cannot be after trade_date")
        return result.sort_values(["trade_date", "model_id", "score", "code"], ascending=[True, True, False, True]).reset_index(drop=True)

    @classmethod
    def from_daily_scores(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        model_id: str,
        asof_time: str | Callable[[Any], Any] = "close",
    ) -> "PredictionFrame":
        records = []
        for row in rows:
            codes = list(row["codes"])
            scores = np.asarray(resolve_row_scores(row), dtype=np.float64)
            if len(codes) != len(scores):
                raise ValueError("daily score row has mismatched codes and scores")
            trade_date = pd.Timestamp(row.get("trade_date", row.get("date"))).normalize()
            if callable(asof_time):
                resolved_asof = asof_time(trade_date)
            elif asof_time == "close":
                resolved_asof = trade_date + pd.Timedelta(hours=15)
            else:
                resolved_asof = asof_time
            for source_order, (code, score) in enumerate(zip(codes, scores)):
                records.append(
                    {
                        "trade_date": trade_date,
                        "code": str(code),
                        "score": float(score),
                        "model_id": str(model_id),
                        "asof_time": resolved_asof,
                        "_source_order": source_order,
                    }
                )
        return cls(pd.DataFrame.from_records(records, columns=PREDICTION_COLUMNS))

    def to_alpha_rows(self, *, tie_breaker: str = "code") -> list[dict[str, Any]]:
        rows = []
        model_ids = self.frame["model_id"].unique()
        if len(model_ids) != 1:
            raise ValueError("alpha JSONL conversion requires exactly one model_id")
        for trade_date, group in self.frame.groupby("trade_date", sort=True):
            if tie_breaker == "code":
                ranked = group.sort_values(["score", "code"], ascending=[False, True])
            elif tie_breaker == "legacy_numpy":
                source = group.sort_values("_source_order")
                ranked = source.iloc[np.argsort(source["score"].to_numpy())[::-1]]
            else:
                raise ValueError(f"unsupported PredictionFrame tie_breaker: {tie_breaker}")
            rows.append(
                {
                    "date": str(pd.Timestamp(trade_date).date()),
                    "codes": ranked["code"].tolist(),
                    "alpha": ranked["score"].astype(float).tolist(),
                    "n_stocks": int(len(ranked)),
                    "model_id": str(model_ids[0]),
                    "asof_time": str(ranked["asof_time"].max()),
                }
            )
        return rows

    def write_alpha_jsonl(self, path: str | Path) -> Path:
        return write_alpha_rows(path, self.to_alpha_rows())

    def manifest(self) -> dict[str, Any]:
        payload = self.frame.loc[:, PREDICTION_COLUMNS].copy()
        payload["trade_date"] = payload["trade_date"].dt.strftime("%Y-%m-%d")
        payload["asof_time"] = payload["asof_time"].astype(str)
        records = payload.to_dict(orient="records")
        return {
            "schema": "prediction_frame_v1",
            "rows": len(records),
            "dates": int(self.frame["trade_date"].nunique()),
            "codes": int(self.frame["code"].nunique()),
            "model_ids": sorted(self.frame["model_id"].unique().tolist()),
            "sha256": canonical_json_hash({"records": records}),
        }


class ModelAdapter(ABC):
    """Minimal lifecycle shared by learnable and frozen models."""

    adapter_name = "abstract"

    def __init__(self, *, model_id: str, config: Mapping[str, Any] | None = None):
        self.model_id = str(model_id).strip()
        if not self.model_id:
            raise ValueError("model_id is required")
        self.config = dict(config or {})
        self.fit_result: FitResult | None = None

    @abstractmethod
    def fit(self, dataset: ProjectDataset) -> FitResult:
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset: ProjectDataset, segment: str) -> PredictionFrame:
        raise NotImplementedError

    @abstractmethod
    def save_state(self, path: str | Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def resume(self, path: str | Path, dataset: ProjectDataset | None = None) -> None:
        raise NotImplementedError

    def select_checkpoint(self) -> Mapping[str, Any]:
        if self.fit_result is None:
            raise RuntimeError(f"adapter {self.model_id} has no fit result")
        return {
            "model_id": self.model_id,
            "rule": self.fit_result.checkpoint_rule,
            "metrics": dict(self.fit_result.metrics),
        }

    def manifest(self) -> dict[str, Any]:
        payload = {
            "adapter": self.adapter_name,
            "model_id": self.model_id,
            "config": self.config,
            "fit_result": None
            if self.fit_result is None
            else {
                "status": self.fit_result.status,
                "metrics": dict(self.fit_result.metrics),
                "checkpoint_rule": self.fit_result.checkpoint_rule,
            },
        }
        payload["contract_sha256"] = canonical_json_hash(payload)
        return payload


def _ensure_dataset_fitted(dataset: ProjectDataset) -> None:
    if dataset.handler.fit_segment is None:
        dataset.fit()


def _collect_bounded(dataset: ProjectDataset, segment: str, *, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    remaining = int(max_rows)
    if remaining <= 0:
        raise ValueError("max_rows must be positive")
    for sample in dataset.prepare(segment, data_key="learn"):
        values = np.asarray(sample["X"], dtype=np.float32)
        labels = np.asarray(sample["y"], dtype=np.float32)
        take = min(remaining, len(labels))
        if take:
            xs.append(values[:take])
            ys.append(labels[:take])
            remaining -= take
        if remaining <= 0:
            break
    if not xs:
        raise ValueError(f"segment {segment} has no model rows")
    return np.vstack(xs), np.concatenate(ys)


def _collect_daily_seeded_quota(
    dataset: ProjectDataset,
    segment: str,
    *,
    max_rows: int,
    seed: int,
    segment_date_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the established rolling trainer's date-balanced sampler exactly."""

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if segment_date_count <= 0:
        raise ValueError("segment_date_count must be positive")
    quota = max(1, int(np.ceil(max_rows / segment_date_count)))
    xs, ys = [], []
    for sample in dataset.prepare(segment, data_key="learn"):
        if "time_index" not in sample:
            raise ValueError("daily seeded quota sampling requires sample time_index")
        values = np.asarray(sample["X"], dtype=np.float32)
        labels = np.asarray(sample["y"], dtype=np.float32)
        take = min(quota, len(labels))
        rng = np.random.default_rng(int(seed) + int(sample["time_index"]))
        selected = rng.choice(len(labels), size=take, replace=False)
        xs.append(values[selected])
        ys.append(labels[selected])
    if not xs:
        raise ValueError(f"segment {segment} has no model rows")
    return np.vstack(xs)[:max_rows], np.concatenate(ys)[:max_rows]


class LightGBMModelAdapter(ModelAdapter):
    adapter_name = "rolling_lgbm_alpha"

    def __init__(self, *, model_id: str, config: Mapping[str, Any] | None = None):
        super().__init__(model_id=model_id, config=config)
        self.model = None

    def fit(self, dataset: ProjectDataset) -> FitResult:
        import lightgbm as lgb

        _ensure_dataset_fitted(dataset)
        sampling_mode = str(self.config.get("sampling_mode", "sequential_bounded_v1"))
        if sampling_mode == "daily_seeded_quota_v1":
            counts = self.config.get("segment_date_counts", {})
            train_x, train_y = _collect_daily_seeded_quota(
                dataset,
                "train",
                max_rows=int(self.config.get("max_train_rows", 100_000)),
                seed=int(self.config.get("seed", 42)),
                segment_date_count=int(counts.get("train", 0)),
            )
            valid_x, valid_y = _collect_daily_seeded_quota(
                dataset,
                "valid",
                max_rows=int(self.config.get("max_valid_rows", 30_000)),
                seed=int(self.config.get("seed", 42)) + 1,
                segment_date_count=int(counts.get("valid", 0)),
            )
        elif sampling_mode == "sequential_bounded_v1":
            train_x, train_y = _collect_bounded(
                dataset, "train", max_rows=int(self.config.get("max_train_rows", 100_000))
            )
            valid_x, valid_y = _collect_bounded(
                dataset, "valid", max_rows=int(self.config.get("max_valid_rows", 30_000))
            )
        else:
            raise ValueError(f"unsupported LightGBM sampling_mode: {sampling_mode}")
        params = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "force_col_wise": True,
            "num_threads": int(self.config.get("num_threads", 4)),
            "seed": int(self.config.get("seed", 42)),
            "learning_rate": float(self.config.get("learning_rate", 0.03)),
            "num_leaves": int(self.config.get("num_leaves", 31)),
            "min_data_in_leaf": int(self.config.get("min_data_in_leaf", 20)),
            "lambda_l2": float(self.config.get("lambda_l2", 1.0)),
            "feature_fraction": float(self.config.get("feature_fraction", 1.0)),
            "bagging_fraction": float(self.config.get("bagging_fraction", 1.0)),
            "bagging_freq": int(self.config.get("bagging_freq", 0)),
        }
        train = lgb.Dataset(train_x, label=train_y, free_raw_data=True)
        valid = lgb.Dataset(valid_x, label=valid_y, reference=train, free_raw_data=True)
        train.construct()
        valid.construct()
        train_rows, valid_rows = len(train_y), len(valid_y)
        del train_x, train_y, valid_x, valid_y
        gc.collect()
        self.model = lgb.train(
            params,
            train,
            num_boost_round=int(self.config.get("num_boost_round", 200)),
            valid_sets=[valid],
            callbacks=[lgb.early_stopping(int(self.config.get("early_stopping_rounds", 20)), verbose=False)],
        )
        self.fit_result = FitResult(
            self.model_id,
            "fitted",
            {
                "train_rows": train_rows,
                "valid_rows": valid_rows,
                "best_iteration": int(self.model.best_iteration),
                "sampling_mode": sampling_mode,
            },
            "valid_l2_early_stopping",
        )
        return self.fit_result

    def predict(self, dataset: ProjectDataset, segment: str) -> PredictionFrame:
        if self.model is None:
            raise RuntimeError("LightGBM adapter has no fitted or resumed model")
        rows = []
        for sample in dataset.prepare(segment, data_key="infer"):
            scores = self.model.predict(np.asarray(sample["X"], dtype=np.float32), num_iteration=self.model.best_iteration)
            rows.append({"date": sample["date"], "codes": sample["codes"], "scores": scores})
        return PredictionFrame.from_daily_scores(rows, model_id=self.model_id)

    def save_state(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("LightGBM adapter has no model to save")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(target), num_iteration=self.model.best_iteration)
        return target

    def resume(self, path: str | Path, dataset: ProjectDataset | None = None) -> None:
        import lightgbm as lgb

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        self.model = lgb.Booster(model_file=str(source))
        self.fit_result = FitResult(
            self.model_id,
            "resumed",
            {"best_iteration": int(self.model.best_iteration)},
            "frozen_lightgbm_artifact",
        )


class FrozenArtifactAdapter(ModelAdapter):
    adapter_name = "frozen_artifact"

    def __init__(self, *, model_id: str, alpha_paths: Mapping[str, str | Path], config: Mapping[str, Any] | None = None):
        super().__init__(model_id=model_id, config=config)
        self.alpha_paths = {str(key): Path(value).resolve() for key, value in alpha_paths.items()}

    def fit(self, dataset: ProjectDataset) -> FitResult:
        self.fit_result = FitResult(self.model_id, "frozen", {}, "frozen_artifact_no_selection")
        return self.fit_result

    def predict(self, dataset: ProjectDataset, segment: str) -> PredictionFrame:
        try:
            path = self.alpha_paths[str(segment)]
        except KeyError as exc:
            raise ValueError(f"frozen artifact has no alpha for segment {segment}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return PredictionFrame.from_daily_scores(load_alpha_rows(path), model_id=self.model_id)

    def save_state(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "adapter": self.adapter_name,
            "model_id": self.model_id,
            "alpha_paths": {
                key: {"path": str(value), "sha256": sha256_file(value)}
                for key, value in self.alpha_paths.items()
            },
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target

    def resume(self, path: str | Path, dataset: ProjectDataset | None = None) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("adapter") not in {"frozen_artifact", "legacy_read_only"}:
            raise ValueError("frozen adapter state has an incompatible adapter")
        resolved = {}
        for segment, item in payload["alpha_paths"].items():
            source = Path(item["path"])
            if not source.is_file() or sha256_file(source) != item["sha256"]:
                raise ValueError(f"frozen alpha artifact mismatch: {source}")
            resolved[segment] = source
        self.alpha_paths = resolved
        self.fit_result = FitResult(self.model_id, "resumed", {}, "frozen_artifact_no_selection")


class LegacyReadOnlyAdapter(FrozenArtifactAdapter):
    adapter_name = "legacy_read_only"

    def fit(self, dataset: ProjectDataset) -> FitResult:
        self.fit_result = FitResult(self.model_id, "legacy_read_only", {}, "legacy_unresolved")
        return self.fit_result

    def save_state(self, path: str | Path) -> Path:
        raise RuntimeError("legacy artifacts are read-only and cannot be re-saved as governed models")


class TorchStrongAlphaAdapter(ModelAdapter):
    """Strong-alpha lifecycle with explicit existing-trainer delegation."""

    adapter_name = "torch_strong_alpha"

    def __init__(
        self,
        *,
        model_id: str,
        config: Mapping[str, Any] | None = None,
        model=None,
        fit_delegate: Callable[[ProjectDataset, Mapping[str, Any]], tuple[Any, Mapping[str, Any]]] | None = None,
        model_factory: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None = None,
    ):
        super().__init__(model_id=model_id, config=config)
        self.model = model
        self.fit_delegate = fit_delegate
        self.model_factory = model_factory
        self.device = None

    def _resolve_device(self):
        import torch

        requested = str(self.config.get("device", "auto"))
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Torch adapter requested CUDA but CUDA is unavailable")
        self.device = torch.device(requested)
        return self.device

    def fit(self, dataset: ProjectDataset) -> FitResult:
        if self.fit_delegate is None:
            raise RuntimeError("Torch strong-alpha fit requires an explicit existing-trainer delegate")
        _ensure_dataset_fitted(dataset)
        self.model, metrics = self.fit_delegate(dataset, self.config)
        if "regime_dim" in metrics:
            self.config["regime_dim"] = int(metrics["regime_dim"])
        self.model.to(self._resolve_device()).eval()
        self.fit_result = FitResult(
            self.model_id,
            "fitted",
            dict(metrics),
            str(self.config.get("checkpoint_rule", "valid_only_predeclared_metric")),
        )
        return self.fit_result

    def predict(self, dataset: ProjectDataset, segment: str) -> PredictionFrame:
        import torch

        if self.model is None:
            raise RuntimeError("Torch adapter has no fitted or resumed model")
        device = self.device or self._resolve_device()
        self.model.to(device).eval()
        rows = []
        with torch.inference_mode():
            for sample in dataset.prepare(segment, data_key="infer"):
                x = torch.from_numpy(np.asarray(sample["X"], dtype=np.float32)).unsqueeze(0).to(device)
                risk_values = np.asarray(sample.get("risk", np.zeros((len(sample["codes"]), 1))), dtype=np.float32)
                risk = torch.from_numpy(risk_values).unsqueeze(0).to(device)
                regime_dim = int(self.config.get("regime_dim", risk.shape[-1]))
                if regime_dim <= 0 or regime_dim > risk.shape[-1]:
                    raise ValueError("Torch adapter regime_dim is outside risk feature width")
                risk = risk[..., :regime_dim]
                industries = torch.from_numpy(np.asarray(sample.get("industry_ids", np.full(len(sample["codes"]), -1)), dtype=np.int64)).unsqueeze(0).to(device)
                mask = torch.ones((1, len(sample["codes"])), dtype=torch.bool, device=device)
                output = self.model(x, risk, mask, industries)
                scores = output[0] if isinstance(output, (tuple, list)) else output
                scores = scores[0].detach().cpu().numpy()
                score_transform = str(self.config.get("score_transform", "raw"))
                if score_transform == "v9_rank":
                    scores = np.nan_to_num(
                        np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    if scores.size > 1:
                        scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
                    scores = np.tanh(scores)
                elif score_transform != "raw":
                    raise ValueError(f"unsupported Torch score_transform: {score_transform}")
                rows.append({"date": sample["date"], "codes": sample["codes"], "scores": scores})
        return PredictionFrame.from_daily_scores(rows, model_id=self.model_id)

    def save_state(self, path: str | Path) -> Path:
        import torch

        if self.model is None:
            raise RuntimeError("Torch adapter has no model to save")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "adapter_config": self.config,
                "model_id": self.model_id,
                "fit_result": None if self.fit_result is None else dict(self.fit_result.metrics),
            },
            target,
        )
        return target

    def resume(self, path: str | Path, dataset: ProjectDataset | None = None) -> None:
        import torch

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if self.model_factory is None:
            raise RuntimeError("Torch checkpoint resume requires model_factory")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        sample = None
        if dataset is not None:
            sample = next(iter(dataset.prepare("train", data_key="infer")))
        self.model = self.model_factory(checkpoint, sample or {})
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self._resolve_device()).eval()
        self.fit_result = FitResult(
            self.model_id,
            "resumed",
            dict(checkpoint.get("fit_result") or {}),
            str(self.config.get("checkpoint_rule", "frozen_torch_checkpoint")),
        )


def create_model_adapter(
    model_spec: Mapping[str, Any],
    *,
    model_id: str,
    dependencies: ModelAdapterDependencies | None = None,
) -> ModelAdapter:
    """Build one governed adapter from a Workflow-v2 model section.

    Callable trainer/model dependencies are injected by the runtime and never
    serialized into the Workflow. Q5 owns the concrete e19 trainer binding.
    """

    adapter_name = str(model_spec.get("adapter", "")).strip()
    if adapter_name not in MODEL_ADAPTER_NAMES:
        raise ValueError(f"unsupported model adapter: {adapter_name!r}")
    raw_config = model_spec.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise TypeError("model.config must be a mapping")
    config = dict(raw_config)
    if "seed" in model_spec:
        config.setdefault("seed", int(model_spec["seed"]))
    if "resource_limits" in model_spec:
        config.setdefault("resource_limits", dict(model_spec["resource_limits"]))
    deps = dependencies or ModelAdapterDependencies()

    if adapter_name == "rolling_lgbm_alpha":
        return LightGBMModelAdapter(model_id=model_id, config=config)
    if adapter_name == "torch_strong_alpha":
        return TorchStrongAlphaAdapter(
            model_id=model_id,
            config=config,
            model=deps.model,
            fit_delegate=deps.fit_delegate,
            model_factory=deps.model_factory,
        )
    if deps.alpha_paths is None:
        raise ValueError(f"{adapter_name} requires runtime alpha_paths")
    adapter_type = FrozenArtifactAdapter if adapter_name == "frozen_artifact" else LegacyReadOnlyAdapter
    return adapter_type(model_id=model_id, alpha_paths=deps.alpha_paths, config=config)
