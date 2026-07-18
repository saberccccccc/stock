import json

import numpy as np
import pandas as pd
import pytest

from backtest.execution_coverage import audit_execution_coverage
from backtest.open_ledger import build_execution_constraint_masks
from run.download_historical_st_events import (
    fetch_all_namechange_events,
    fetch_all_st_events,
    fetch_namechange_intervals,
    fetch_st_events_by_codes,
    validate_dataset_role,
)
from data.st_status import (
    EVENT_COLUMNS,
    derive_is_st,
    derive_is_st_from_name,
    file_sha256,
    load_st_status_events,
    normalize_namechange_events,
    normalize_st_events,
)


def _event_frame(rows):
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _write_manifest(
    raw_dir,
    event_path,
    *,
    coverage_start="1998-04-28",
    coverage_end="2026-05-18",
    source_kind="event_history",
    source_endpoint="tushare.st",
    source_label=None,
    dataset_role=None,
):
    manifest = {
        "schema_version": 1,
        "source_endpoint": source_endpoint,
        "source_kind": source_kind,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "invalid_row_count": 0,
        "output_sha256": file_sha256(event_path),
    }
    if source_label is not None:
        manifest["source_label"] = source_label
    if dataset_role is not None:
        manifest["dataset_role"] = dataset_role
    (raw_dir / "st_status_events_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_derive_is_st_prioritizes_transition_wording():
    assert derive_is_st("ST", "撤销*ST", "") is False
    assert derive_is_st("*ST", "撤销*ST并实行ST", "") is True
    assert derive_is_st("ST", "", "") is True
    assert derive_is_st("", "", "") is None


def test_normalize_st_events_filters_future_rows_and_preserves_effective_date():
    raw = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "name": "Demo", "pub_date": "20240101", "imp_date": "20240102", "st_type": "ST", "st_reason": "", "st_explain": ""},
            {"ts_code": "000001.SZ", "name": "Demo", "pub_date": "20240701", "imp_date": "20240702", "st_type": "ST", "st_reason": "撤销*ST", "st_explain": ""},
            {"ts_code": "000001.SZ", "name": "Demo", "pub_date": "20260519", "imp_date": "20260520", "st_type": "ST", "st_reason": "", "st_explain": ""},
        ]
    )

    normalized = normalize_st_events(raw, as_of_date="2025-12-31")

    assert normalized["ts_code"].tolist() == ["000001.SZ", "000001.SZ"]
    assert normalized["event_date"].tolist() == ["2024-01-02", "2024-07-02"]
    assert normalized["is_st"].tolist() == [True, False]


def test_normalize_namechange_intervals_reconstructs_activation_and_removal():
    raw = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "name": "平安银行", "start_date": "20240101", "end_date": "20240201", "ann_date": "20231231", "change_reason": ""},
            {"ts_code": "000001.SZ", "name": "ST平安", "start_date": "20240202", "end_date": "20240301", "ann_date": "20240201", "change_reason": "风险警示"},
            {"ts_code": "000001.SZ", "name": "平安银行", "start_date": "20240302", "end_date": "", "ann_date": "20240301", "change_reason": "撤销风险警示"},
        ]
    )

    normalized = normalize_namechange_events(raw, as_of_date="2025-12-31")

    assert derive_is_st_from_name("ST平安")
    assert not derive_is_st_from_name("平安银行")
    assert normalized["event_date"].tolist() == ["2024-01-01", "2024-02-02", "2024-03-02"]
    assert normalized["is_st"].tolist() == [False, True, False]
    assert set(normalized["source_endpoint"]) == {"tushare.namechange"}


def test_historical_event_file_drives_open_limit_status(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    event_path = raw_dir / "st_status_events.csv"
    normalized = normalize_st_events(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "Demo", "pub_date": "20250102", "imp_date": "20250103", "st_type": "ST", "st_reason": "", "st_explain": ""},
            ]
        ),
        as_of_date="2025-12-31",
    )
    normalized.to_csv(event_path, index=False, encoding="utf-8-sig")
    _write_manifest(raw_dir, event_path)

    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    columns = ["000001.SZ"]
    open_df = pd.DataFrame([[10.0], [10.5]], index=dates, columns=columns)
    close_df = pd.DataFrame([[10.0], [10.5]], index=dates, columns=columns)
    high_df = open_df.copy()
    low_df = open_df.copy()
    volume_df = pd.DataFrame(1000.0, index=dates, columns=columns)
    money_df = pd.DataFrame(1_000_000.0, index=dates, columns=columns)

    masks = build_execution_constraint_masks(
        open_df,
        close_df,
        high_df,
        low_df,
        volume_df,
        money_df,
        raw_dir,
        no_limit_first_trading_days=0,
        min_buy_listing_days=0,
    )

    assert masks["limit_up_open"].loc[pd.Timestamp("2025-01-03"), "000001.SZ"]


def test_load_st_status_events_rejects_missing_contract(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame({"ts_code": ["000001.SZ"], "imp_date": ["2025-01-03"]}).to_csv(
        raw_dir / "st_status_events.csv", index=False
    )

    try:
        load_st_status_events(raw_dir)
    except ValueError as exc:
        assert "invalid ST event file" in str(exc)
    else:
        raise AssertionError("malformed ST event file was accepted")


def test_execution_coverage_accepts_historical_event_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "stable_stocks.csv").write_text(
        "ts_code,name,list_date\n000001.SZ,Demo,20100101\n", encoding="utf-8"
    )
    event_path = raw_dir / "st_status_events.csv"
    normalized = normalize_st_events(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "Demo", "pub_date": "20240101", "imp_date": "20240102", "st_type": "ST", "st_reason": "", "st_explain": ""},
            ]
        ),
        as_of_date="2026-05-18",
    )
    normalized.to_csv(event_path, index=False, encoding="utf-8-sig")
    _write_manifest(raw_dir, event_path)
    cache = tmp_path / "matrix"
    cache.mkdir()
    (cache / "ohlc_matrix_meta.json").write_text(
        json.dumps({"source_count": 1, "fields": ["open", "high", "low", "close", "volume", "money"], "dates": ["2024-01-02", "2025-12-31"]}),
        encoding="utf-8",
    )

    report = audit_execution_coverage(raw_dir, cache, start_date="2024-01-02", end_date="2025-12-31")

    assert report["status"] == "coverage_complete"
    assert report["historical_st"]["source_type"] == "event_history"
    assert report["historical_st"]["manifest_hash_matches"]
    assert report["historical_st"]["covers_requested_interval"]


def test_execution_coverage_labels_namechange_reconstruction(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "stable_stocks.csv").write_text(
        "ts_code,name,list_date\n000001.SZ,Demo,20100101\n", encoding="utf-8"
    )
    event_path = raw_dir / "st_status_events.csv"
    normalized = normalize_namechange_events(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": "ST Demo",
                    "start_date": "20240101",
                    "end_date": "20240131",
                    "ann_date": "20231231",
                    "change_reason": "risk warning",
                }
            ]
        ),
        as_of_date="2026-05-18",
    )
    normalized.to_csv(event_path, index=False, encoding="utf-8-sig")
    _write_manifest(
        raw_dir,
        event_path,
        coverage_start="2024-01-01",
        source_kind="tushare_namechange_intervals",
        source_endpoint="tushare.namechange",
        source_label="由历史股票名称区间重建",
    )
    cache = tmp_path / "matrix"
    cache.mkdir()
    (cache / "ohlc_matrix_meta.json").write_text(
        json.dumps(
            {
                "source_count": 1,
                "fields": ["open", "high", "low", "close", "volume", "money"],
                "dates": ["2024-01-01", "2025-12-31"],
            }
        ),
        encoding="utf-8",
    )

    report = audit_execution_coverage(
        raw_dir, cache, start_date="2024-01-01", end_date="2025-12-31"
    )

    assert report["status"] == "coverage_complete"
    assert report["historical_st"]["source_type"] == "tushare_namechange_intervals"
    assert report["historical_st"]["source_endpoint"] == "tushare.namechange"
    assert report["historical_st"]["source_label"] == "由历史股票名称区间重建"
    forward_report = audit_execution_coverage(
        raw_dir,
        cache,
        start_date="2024-01-01",
        end_date="2025-12-31",
        dataset_role="forward",
    )
    assert not forward_report["historical_st"]["covers_requested_interval"]
    assert not forward_report["historical_st"]["dataset_role_matches"]


def test_fetch_all_st_events_paginates_until_short_page():
    pages = {
        0: pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]}),
        2: pd.DataFrame({"ts_code": ["000003.SZ"]}),
    }

    class FakePro:
        def st(self, *, offset, limit):
            return pages.get(offset, pd.DataFrame())

    calls = []

    def caller(func, **kwargs):
        calls.append(kwargs)
        return func(**kwargs)

    frame, page_count = fetch_all_st_events(FakePro(), caller, page_size=2, max_pages=5)

    assert page_count == 2
    assert frame["ts_code"].tolist() == ["000001.SZ", "000002.SZ", "000003.SZ"]
    assert calls == [{"offset": 0, "limit": 2}, {"offset": 2, "limit": 2}]


def test_fetch_all_namechange_uses_namechange_endpoint():
    class FakePro:
        def namechange(self, *, offset, limit):
            return pd.DataFrame({"ts_code": ["000001.SZ"]}) if offset == 0 else pd.DataFrame()

    calls = []

    def caller(func, *args, **kwargs):
        calls.append((args, kwargs))
        return func(**kwargs)

    frame, page_count = fetch_all_namechange_events(FakePro(), caller, page_size=2, max_pages=2)

    assert page_count == 1
    assert frame["ts_code"].tolist() == ["000001.SZ"]
    assert calls == [((), {"offset": 0, "limit": 2})]


def test_fetch_all_st_events_can_resume_a_complete_checkpoint(tmp_path):
    pages = {0: pd.DataFrame({"ts_code": ["000001.SZ"]})}
    checkpoint = tmp_path / "pages"

    class FakePro:
        def st(self, *, offset, limit):
            return pages.get(offset, pd.DataFrame())

    def caller(func, **kwargs):
        return func(**kwargs)

    first, first_count = fetch_all_st_events(
        FakePro(), caller, page_size=2, checkpoint_dir=checkpoint
    )
    second, second_count = fetch_all_st_events(
        FakePro(),
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("checkpoint was not reused")),
        page_size=2,
        checkpoint_dir=checkpoint,
        resume=True,
    )

    assert first_count == second_count == 1
    pd.testing.assert_frame_equal(
        first.astype(str), second.astype(str), check_dtype=False
    )


def test_fetch_namechange_intervals_uses_documented_date_range(tmp_path):
    calls = []

    class FakePro:
        def namechange(self, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["ST Demo"],
                    "start_date": ["20240101"],
                    "end_date": ["20240131"],
                }
            )

    frame, count = fetch_namechange_intervals(
        FakePro(),
        lambda func, **kwargs: func(**kwargs),
        start_date="19900101",
        end_date="20260518",
        checkpoint_dir=tmp_path / "pages",
    )

    assert count == 1
    assert frame["ts_code"].tolist() == ["000001.SZ"]
    assert calls == [{"start_date": "19900101", "end_date": "20260518"}]


def test_fetch_st_events_by_code_uses_code_checkpoints_and_resumes(tmp_path):
    calls = []

    class FakePro:
        def st(self, *, ts_code):
            calls.append(ts_code)
            return pd.DataFrame({"ts_code": [ts_code], "imp_date": ["20240101"]})

    codes = ["000001.SZ", "600000.SH"]
    first, first_count = fetch_st_events_by_codes(
        FakePro(),
        lambda func, **kwargs: func(**kwargs),
        ts_codes=codes,
        checkpoint_dir=tmp_path / "pages",
    )
    second, second_count = fetch_st_events_by_codes(
        FakePro(),
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("code checkpoint was not reused")
        ),
        ts_codes=codes,
        checkpoint_dir=tmp_path / "pages",
        resume=True,
    )

    assert first_count == second_count == 2
    assert calls == codes
    pd.testing.assert_frame_equal(
        first.astype(str), second.astype(str), check_dtype=False
    )


def test_fetch_st_events_by_code_resumes_empty_code_checkpoint(tmp_path):
    calls = []

    class FakePro:
        def st(self, *, ts_code):
            calls.append(ts_code)
            if ts_code == "000001.SZ":
                return pd.DataFrame()
            return pd.DataFrame({"ts_code": [ts_code], "imp_date": ["20240101"]})

    codes = ["000001.SZ", "600000.SH"]
    first, first_count = fetch_st_events_by_codes(
        FakePro(),
        lambda func, **kwargs: func(**kwargs),
        ts_codes=codes,
        checkpoint_dir=tmp_path / "pages",
    )
    second, second_count = fetch_st_events_by_codes(
        FakePro(),
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("empty code checkpoint was not reused")
        ),
        ts_codes=codes,
        checkpoint_dir=tmp_path / "pages",
        resume=True,
    )

    assert first_count == second_count == 2
    assert calls == codes
    pd.testing.assert_frame_equal(
        first.astype(str), second.astype(str), check_dtype=False
    )


def test_dataset_role_enforces_the_research_cutoff():
    assert str(validate_dataset_role("2025-12-31", "research").date()) == "2025-12-31"
    assert str(validate_dataset_role("2026-01-01", "forward").date()) == "2026-01-01"
    with pytest.raises(ValueError, match="research dataset cannot exceed"):
        validate_dataset_role("2026-01-01", "research")
    with pytest.raises(ValueError, match="forward dataset must be after"):
        validate_dataset_role("2025-12-31", "forward")
