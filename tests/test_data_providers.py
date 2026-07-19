import json

import numpy as np
import pandas as pd
import pytest

from data.providers import (
    DataView,
    DateRange,
    ExecutionConstraintProvider,
    ExternalMarketPITProvider,
    FundamentalPITProvider,
    OhlcvMatrixProvider,
    ProcessorContract,
    ProcessorKind,
    V14MemmapProvider,
)


def _write_stock(path):
    pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03", "2025-01-02"],
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [100, 110, 120],
            "money": [1000, 1100, 1200],
        }
    ).to_csv(path, index=False)


def _view(data):
    return DataView.create(
        name="val",
        physical_root=data,
        feature_warmup_start="2024-01-01",
        feature_warmup_end="2024-01-01",
        task_start="2024-01-01",
        task_end="2024-12-31",
        evaluation_start="2024-01-01",
        evaluation_end="2024-12-31",
        max_data_date="2024-12-31",
    )


def test_same_physical_store_supports_bounded_arbitrary_views(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    _write_stock(data / "000001.SZ.csv")
    provider = OhlcvMatrixProvider(data_view=_view(data), cache_dir=tmp_path / "cache")

    frames = provider.load(
        codes=["000001.SZ"],
        fields=["open", "close", "pct_chg"],
        start_date="2024-01-03",
        end_date="2024-12-31",
    )

    assert frames["open"].index.max() == pd.Timestamp("2024-01-03")
    assert frames["open"].iloc[0, 0] == 11.0
    assert np.isnan(frames["pct_chg"].iloc[0, 0])
    with pytest.raises(ValueError, match="max_data_date"):
        provider.load(
            codes=["000001.SZ"],
            fields=["open"],
            start_date="2025-01-01",
            end_date="2025-01-02",
        )


def test_ohlcv_manifest_can_audit_without_rebuilding_mismatched_cache(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_stock(source / "000001.SZ.csv")
    other = tmp_path / "other"
    other.mkdir()
    _write_stock(other / "000001.SZ.csv")
    cache = tmp_path / "cache"
    OhlcvMatrixProvider(data_view=_view(source), cache_dir=cache).manifest()
    before = (cache / "ohlc_matrix_meta.json").read_bytes()

    manifest = OhlcvMatrixProvider(data_view=_view(other), cache_dir=cache).manifest(
        ensure_cache=False
    )

    assert manifest["cache"]["matches_data_view"] is False
    assert (cache / "ohlc_matrix_meta.json").read_bytes() == before


def test_data_view_manifest_separates_physical_and_logical_ranges(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    _write_stock(data / "000001.SZ.csv")

    manifest = _view(data).manifest()

    assert manifest["physical_fingerprint"]["file_count"] == 1
    assert manifest["evaluation"]["end"] == "2024-12-31"
    assert manifest["max_data_date"] == "2024-12-31"


def test_train_fitted_processor_state_cannot_cross_train_end(tmp_path):
    state = tmp_path / "scaler.json"
    state.write_text(json.dumps({"mean": 1.0}), encoding="utf-8")
    processor = ProcessorContract(
        name="standardize",
        kind=ProcessorKind.TRAIN_FITTED,
        config={"clip": 4},
        fit_range=DateRange.create("2020-01-01", "2024-01-01", field="fit"),
        state_path=state,
    )

    with pytest.raises(ValueError, match="fit range exceeds"):
        processor.manifest(task_train_end="2023-12-31")

    valid = ProcessorContract(
        name="daily_rank",
        kind=ProcessorKind.DAILY_CROSS_SECTION,
        config={"clip": 4},
    ).manifest(task_train_end="2023-12-31")
    assert valid["kind"] == "daily_cross_section"
    assert valid["state"] is None


def test_v14_provider_bounds_logical_dates_and_records_transform(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    meta_path = tmp_path / "meta.pkl"
    meta_path.write_bytes(b"metadata")
    meta = {
        "all_dates": ["2023-01-02", "2024-01-02", "2025-01-02", "2026-01-02"],
        "all_codes": ["000001.SZ"],
        "x_dim": 2,
        "risk_full_dim": 1,
        "feature_cols": ["ret_5d", "ret_20d"],
        "label_families": {"oo": {"norm_path": "oo.dat"}},
    }
    view = DataView.create(
        name="selection",
        physical_root=data,
        feature_warmup_start="2023-01-02",
        feature_warmup_end="2023-12-31",
        task_start="2023-01-02",
        task_end="2025-12-31",
        evaluation_start="2024-01-01",
        evaluation_end="2025-12-31",
        max_data_date="2025-12-31",
    )
    provider = V14MemmapProvider(meta=meta, meta_path=meta_path, data_view=view)

    assert provider.date_indices("2024-01-01", "2025-12-31") == [1, 2]
    assert provider.date_indices("2022-12-31", "2023-01-02") == [0]
    assert provider.manifest()["transform_contract"]["feature_transforms"]["global_train_fitted_scaler"] is False
    with pytest.raises(ValueError, match="max_data_date"):
        provider.date_indices("2026-01-01", "2026-01-31")


def test_fundamental_provider_applies_values_only_after_effective_date(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    source = tmp_path / "fundamentals.csv"
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "effective_date": "2024-04-30",
                "end_date": "2024-03-31",
                "roe": 0.12,
                "revenue_yoy": 0.08,
                "notice_is_estimated": False,
            }
        ]
    ).to_csv(source, index=False)
    provider = FundamentalPITProvider(source_path=source, data_view=_view(data))

    daily = provider.daily(
        codes=["000001.SZ"],
        dates=["2024-04-29", "2024-04-30", "2024-05-06"],
    )

    assert daily.loc["2024-04-29", "000001.SZ_roe"] == 0.0
    assert daily.loc["2024-04-30", "000001.SZ_roe"] == pytest.approx(0.12)
    assert daily.loc["2024-05-06", "000001.SZ_days_since_effective"] == 6
    assert provider.manifest()["source"]["estimated_notice_rows"] == 0


def test_execution_constraint_provider_preserves_historical_st_gap(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "stable_stocks.csv").write_text(
        "ts_code,name,list_date\n000001.SZ,Demo,20100101\n",
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    (matrix / "ohlc_matrix_meta.json").write_text(
        json.dumps(
            {
                "source_count": 1,
                "fields": ["open", "high", "low", "close", "volume", "money"],
                "dates": ["2024-01-01", "2024-12-31"],
            }
        ),
        encoding="utf-8",
    )
    provider = ExecutionConstraintProvider(
        data_view=_view(raw),
        matrix_cache_dir=matrix,
        dataset_role="research",
    )

    manifest = provider.manifest()

    assert manifest["formal_coverage_complete"] is False
    assert "historical_st_status_not_covered" in manifest["coverage"]["gaps"]


def test_external_market_provider_rejects_same_day_session(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    source = tmp_path / "global.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-03"],
            "us_session_date": ["2024-01-02"],
            "global_spy_ret_1d": [0.01],
        }
    ).to_csv(source, index=False)
    provider = ExternalMarketPITProvider(feature_path=source, data_view=_view(data))

    frame = provider.slice(start_date="2024-01-03", end_date="2024-01-03")
    assert frame.iloc[0]["global_spy_ret_1d"] == pytest.approx(0.01)

    pd.DataFrame(
        {
            "date": ["2024-01-03"],
            "us_session_date": ["2024-01-03"],
            "global_spy_ret_1d": [0.01],
        }
    ).to_csv(source, index=False)
    with pytest.raises(ValueError, match="not completed"):
        provider.slice(start_date="2024-01-03", end_date="2024-01-03")
