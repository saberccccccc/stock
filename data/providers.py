"""Project-native data views and cached providers for formal workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from backtest.ohlc_matrix_cache import (
    ensure_ohlc_matrix_cache,
    load_ohlcv_fields_from_matrix_cache,
)
from backtest.execution_coverage import audit_execution_coverage
from data.fundamental_factors import merge_to_daily_akshare
from data.rolling_samples import iter_rolling_samples
from data.transform_contract import build_v14_transform_contract
from experiments.recording import canonical_json_hash, fingerprint_path, sha256_file


def _date(value: Any, field: str) -> pd.Timestamp:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field} is required")
    return pd.Timestamp(value).normalize()


@dataclass(frozen=True)
class DateRange:
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def create(cls, start: Any, end: Any, *, field: str) -> "DateRange":
        value = cls(_date(start, f"{field}.start"), _date(end, f"{field}.end"))
        if value.start > value.end:
            raise ValueError(f"{field}.start exceeds {field}.end")
        return value

    def to_dict(self) -> dict[str, str]:
        return {"start": str(self.start.date()), "end": str(self.end.date())}


@dataclass(frozen=True)
class DataView:
    """Logical task intervals over one physical point-in-time data store."""

    name: str
    physical_root: Path
    feature_warmup: DateRange
    task: DateRange
    evaluation: DateRange
    max_data_date: pd.Timestamp

    @classmethod
    def create(
        cls,
        *,
        name: str,
        physical_root: str | Path,
        feature_warmup_start: Any,
        feature_warmup_end: Any,
        task_start: Any,
        task_end: Any,
        evaluation_start: Any,
        evaluation_end: Any,
        max_data_date: Any,
    ) -> "DataView":
        value = cls(
            name=str(name).strip(),
            physical_root=Path(physical_root).resolve(),
            feature_warmup=DateRange.create(feature_warmup_start, feature_warmup_end, field="feature_warmup"),
            task=DateRange.create(task_start, task_end, field="task"),
            evaluation=DateRange.create(evaluation_start, evaluation_end, field="evaluation"),
            max_data_date=_date(max_data_date, "max_data_date"),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if not self.name:
            raise ValueError("data view name is required")
        if not self.physical_root.exists():
            raise FileNotFoundError(self.physical_root)
        if self.feature_warmup.start > self.task.start:
            raise ValueError("feature warm-up must begin no later than the task")
        if self.task.end > self.max_data_date:
            raise ValueError("task range exceeds max_data_date")
        if self.evaluation.start < self.task.start or self.evaluation.end > self.task.end:
            raise ValueError("evaluation range must be contained in the task range")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "physical_root": str(self.physical_root),
            "physical_fingerprint": fingerprint_path(self.physical_root),
            "feature_warmup": self.feature_warmup.to_dict(),
            "task": self.task.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "max_data_date": str(self.max_data_date.date()),
        }


class ProcessorKind(str, Enum):
    PIT_ALIGNMENT = "pit_alignment"
    DAILY_CROSS_SECTION = "daily_cross_section"
    TRAIN_FITTED = "train_fitted"
    INFERENCE_ONLY = "inference_only"


@dataclass(frozen=True)
class ProcessorContract:
    name: str
    kind: ProcessorKind
    config: Mapping[str, Any]
    fit_range: DateRange | None = None
    state_path: Path | None = None

    def manifest(self, *, task_train_end: Any) -> dict[str, Any]:
        train_end = _date(task_train_end, "task_train_end")
        if self.kind == ProcessorKind.TRAIN_FITTED:
            if self.fit_range is None or self.state_path is None:
                raise ValueError(f"train-fitted processor {self.name} requires fit_range and state_path")
            if self.fit_range.end > train_end:
                raise ValueError(f"processor {self.name} fit range exceeds task train end")
            state = Path(self.state_path).resolve()
            if not state.is_file():
                raise FileNotFoundError(state)
            state_record = {"path": str(state), "sha256": sha256_file(state)}
            fit_range = self.fit_range.to_dict()
        else:
            if self.fit_range is not None or self.state_path is not None:
                raise ValueError(f"processor {self.name} kind={self.kind.value} must not carry fitted state")
            state_record = None
            fit_range = None
        payload = {
            "name": self.name,
            "kind": self.kind.value,
            "config": dict(self.config),
            "fit_range": fit_range,
            "state": state_record,
        }
        payload["contract_sha256"] = canonical_json_hash(payload)
        return payload


class OhlcvMatrixProvider:
    """Arbitrary-range facade over the existing global OHLC memmap cache."""

    def __init__(self, *, data_view: DataView, cache_dir: str | Path):
        self.data_view = data_view
        self.cache_dir = Path(cache_dir).resolve()

    def load(
        self,
        *,
        codes: Sequence[str],
        fields: Sequence[str],
        start_date: Any,
        end_date: Any,
        money_scale: float = 1.0,
        rebuild: bool = False,
    ):
        requested = DateRange.create(start_date, end_date, field="provider_request")
        if requested.start < self.data_view.feature_warmup.start:
            raise ValueError("provider request starts before declared feature warm-up")
        if requested.end > self.data_view.max_data_date:
            raise ValueError("provider request exceeds data view max_data_date")
        return load_ohlcv_fields_from_matrix_cache(
            self.data_view.physical_root,
            self.cache_dir,
            codes,
            money_scale=money_scale,
            start_date=requested.start,
            end_date=requested.end,
            fields=fields,
            rebuild=rebuild,
        )

    def manifest(self) -> dict[str, Any]:
        meta = ensure_ohlc_matrix_cache(self.data_view.physical_root, self.cache_dir)
        return {
            "schema_version": 1,
            "provider": "ohlcv_matrix_v1",
            "data_view": self.data_view.manifest(),
            "cache": {
                "root": str(self.cache_dir),
                "version": meta["version"],
                "source_count": meta["source_count"],
                "source_hash": meta["source_hash"],
                "date_start": meta["dates"][0],
                "date_end": meta["dates"][-1],
                "fields": list(meta["fields"]),
            },
        }


class V14MemmapProvider:
    """Date-bounded facade over an existing v14 feature/label memmap bundle."""

    def __init__(self, *, meta: Mapping[str, Any], meta_path: str | Path, data_view: DataView):
        self.meta = dict(meta)
        self.meta_path = Path(meta_path).resolve()
        self.data_view = data_view
        if not self.meta_path.is_file():
            raise FileNotFoundError(self.meta_path)
        self.dates = pd.DatetimeIndex(pd.to_datetime(self.meta.get("all_dates", []))).normalize()
        if self.dates.empty or not self.dates.is_monotonic_increasing:
            raise ValueError("v14 metadata dates must be non-empty and ordered")
        if self.data_view.feature_warmup.start < self.dates.min():
            raise ValueError("data view feature warm-up precedes v14 physical coverage")
        if self.data_view.max_data_date > self.dates.max():
            raise ValueError("data view max_data_date exceeds v14 physical coverage")

    def date_indices(self, start_date: Any, end_date: Any) -> list[int]:
        requested = DateRange.create(start_date, end_date, field="v14_request")
        if requested.start < self.data_view.feature_warmup.start:
            lead = self.data_view.feature_warmup.start - requested.start
            if lead > pd.Timedelta(days=10):
                raise ValueError("v14 request starts before declared feature warm-up")
            requested = DateRange(self.data_view.feature_warmup.start, requested.end)
        if requested.end > self.data_view.max_data_date:
            raise ValueError("v14 request exceeds data view max_data_date")
        mask = (self.dates >= requested.start) & (self.dates <= requested.end)
        return [int(value) for value in mask.nonzero()[0]]

    def iter_samples(
        self,
        *,
        start_date: Any,
        end_date: Any,
        label_family: str,
        horizon_index: int,
        feature_indices=None,
        include_risk: bool = True,
        include_industry: bool = True,
    ):
        indices = self.date_indices(start_date, end_date)
        return iter_rolling_samples(
            self.meta,
            indices,
            label_family,
            horizon_index,
            feature_indices,
            include_risk=include_risk,
            include_industry=include_industry,
        )

    def manifest(self) -> dict[str, Any]:
        transform = build_v14_transform_contract(self.meta, meta_path=self.meta_path)
        return {
            "schema_version": 1,
            "provider": "v14_memmap_v1",
            "data_view": self.data_view.manifest(),
            "metadata": {
                "path": str(self.meta_path),
                "sha256": sha256_file(self.meta_path),
                "date_start": str(self.dates.min().date()),
                "date_end": str(self.dates.max().date()),
                "codes": len(self.meta.get("all_codes", [])),
            },
            "transform_contract": transform,
            "transform_contract_sha256": canonical_json_hash(transform),
        }


class FundamentalPITProvider:
    """Effective-date fundamental view with explicit availability-quality flags."""

    def __init__(self, *, source_path: str | Path, data_view: DataView):
        self.source_path = Path(source_path).resolve()
        self.data_view = data_view
        if not self.source_path.is_file():
            raise FileNotFoundError(self.source_path)

    def _read(self) -> pd.DataFrame:
        if self.source_path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(self.source_path)
        else:
            frame = pd.read_csv(self.source_path)
        required = {"ts_code", "effective_date", "end_date"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"fundamental PIT source missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce").dt.normalize()
        frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce").dt.normalize()
        if frame["effective_date"].isna().any():
            raise ValueError("fundamental PIT source contains missing effective_date")
        if "notice_is_estimated" not in frame.columns:
            frame["notice_is_estimated"] = False
        return frame.sort_values(["ts_code", "effective_date", "end_date"])

    def daily(self, *, codes: Sequence[str], dates: Sequence[Any]) -> pd.DataFrame:
        index = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
        if index.empty:
            return pd.DataFrame(index=index)
        if index.min() < self.data_view.feature_warmup.start:
            raise ValueError("fundamental request starts before feature warm-up")
        if index.max() > self.data_view.max_data_date:
            raise ValueError("fundamental request exceeds max_data_date")
        frame = self._read()
        frame = frame.loc[frame["effective_date"] <= self.data_view.max_data_date]
        return merge_to_daily_akshare(frame, list(codes), index, include_quality=True)

    def manifest(self) -> dict[str, Any]:
        frame = self._read()
        estimated = frame["notice_is_estimated"].fillna(False).astype(bool)
        return {
            "schema_version": 1,
            "provider": "fundamental_pit_v1",
            "data_view": self.data_view.manifest(),
            "source": {
                "path": str(self.source_path),
                "fingerprint": fingerprint_path(self.source_path),
                "rows": int(len(frame)),
                "codes": int(frame["ts_code"].nunique()),
                "effective_start": str(frame["effective_date"].min().date()) if len(frame) else None,
                "effective_end": str(frame["effective_date"].max().date()) if len(frame) else None,
                "estimated_notice_rows": int(estimated.sum()),
            },
            "quality_flags": [
                "has_value",
                "days_since_effective",
                "is_fresh_quarter",
                "notice_is_estimated",
            ],
            "availability_rule": "effective_date <= feature_date; then forward-fill latest available report",
        }


class ExecutionConstraintProvider:
    """Manifest facade for A-share listing, OHLC, and historical-ST evidence."""

    def __init__(self, *, data_view: DataView, matrix_cache_dir: str | Path, dataset_role: str):
        self.data_view = data_view
        self.matrix_cache_dir = Path(matrix_cache_dir).resolve()
        self.dataset_role = str(dataset_role)

    def manifest(self) -> dict[str, Any]:
        coverage = audit_execution_coverage(
            self.data_view.physical_root,
            self.matrix_cache_dir,
            start_date=self.data_view.evaluation.start,
            end_date=self.data_view.evaluation.end,
            dataset_role=self.dataset_role,
        )
        return {
            "schema_version": 1,
            "provider": "a_share_execution_constraints_v1",
            "data_view": self.data_view.manifest(),
            "coverage": coverage,
            "formal_coverage_complete": coverage["status"] == "coverage_complete",
        }


class ExternalMarketPITProvider:
    """Read prealigned US/HK/global features under the A-share morning cutoff."""

    def __init__(
        self,
        *,
        feature_path: str | Path,
        data_view: DataView,
        summary_path: str | Path | None = None,
    ):
        self.feature_path = Path(feature_path).resolve()
        self.summary_path = Path(summary_path).resolve() if summary_path else None
        self.data_view = data_view
        if not self.feature_path.is_file():
            raise FileNotFoundError(self.feature_path)

    def _read(self) -> pd.DataFrame:
        if self.feature_path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(self.feature_path)
        else:
            frame = pd.read_csv(self.feature_path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            frame = frame.set_index("date")
        else:
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
        if "us_session_date" not in frame.columns:
            raise ValueError("external market features require us_session_date")
        frame["us_session_date"] = pd.to_datetime(frame["us_session_date"], errors="coerce").dt.normalize()
        invalid = frame["us_session_date"].notna() & (frame["us_session_date"] >= frame.index)
        if invalid.any():
            raise ValueError("external market feature uses a session not completed before the A-share date")
        return frame.sort_index()

    def slice(self, *, start_date: Any, end_date: Any) -> pd.DataFrame:
        requested = DateRange.create(start_date, end_date, field="external_market_request")
        if requested.start < self.data_view.feature_warmup.start:
            raise ValueError("external market request starts before feature warm-up")
        if requested.end > self.data_view.max_data_date:
            raise ValueError("external market request exceeds max_data_date")
        frame = self._read()
        return frame.loc[(frame.index >= requested.start) & (frame.index <= requested.end)].copy()

    def manifest(self) -> dict[str, Any]:
        frame = self._read()
        summary = None
        if self.summary_path:
            if not self.summary_path.is_file():
                raise FileNotFoundError(self.summary_path)
            import json

            summary = json.loads(self.summary_path.read_text(encoding="utf-8-sig"))
            if summary.get("alignment_rule") and "strictly before" not in str(summary["alignment_rule"]):
                raise ValueError("external market summary has an unsafe alignment rule")
        return {
            "schema_version": 1,
            "provider": "external_market_pit_v1",
            "data_view": self.data_view.manifest(),
            "features": {
                "path": str(self.feature_path),
                "fingerprint": fingerprint_path(self.feature_path),
                "date_start": str(frame.index.min().date()) if len(frame) else None,
                "date_end": str(frame.index.max().date()) if len(frame) else None,
                "columns": int(len(frame.columns)),
            },
            "summary": {
                "path": str(self.summary_path),
                "fingerprint": fingerprint_path(self.summary_path),
                "alignment_rule": summary.get("alignment_rule"),
            }
            if self.summary_path
            else None,
            "availability_rule": "A-share date D uses only the latest completed external session strictly before D",
        }
