from pathlib import Path

import pandas as pd

from data.market_daily_incremental_benchmark import (
    benchmark_daily_incremental_refresh,
)
from data.market_daily_store import MarketDailyStore


def _frame(date, close):
    return pd.DataFrame(
        [
            {
                "trade_date": date,
                "code": code,
                "open": close - 0.2 + index,
                "high": close + 0.3 + index,
                "low": close - 0.4 + index,
                "close": close + index,
                "volume": 1000.0 + index,
                "money": 10000.0 + index,
                "factor": 1.0,
            }
            for index, code in enumerate(("000001.SZ", "600000.SH"))
        ]
    )


def test_incremental_benchmark_is_isolated_and_measures_refresh(tmp_path):
    source_root = tmp_path / "source"
    store = MarketDailyStore(source_root)
    for date, close in (
        ("2026-07-01", 10.0),
        ("2026-07-02", 10.2),
        ("2026-07-03", 10.4),
    ):
        store.commit_partition(
            _frame(date, close),
            instrument_type="equity",
            source="test",
        )
    source_identity = store.active_state()

    result = benchmark_daily_incremental_refresh(
        source_root,
        month="2026-07",
        workspace_parent=tmp_path / "workspaces",
    )

    assert result["status"] == "passed"
    assert result["source"]["rows"] == 6
    assert result["daily_commit"]["rows"] == 2
    assert result["daily_commit"]["result"]["status"] == "committed"
    assert result["monthly_cache_refresh"]["result_status"] == "rebuilt"
    assert result["warm_cache_before_commit"]["result_status"] == "already_current"
    assert result["warm_cache_after_refresh"]["result_status"] == "already_current"
    assert result["integrity"]["store"]["status"] == "passed"
    assert result["integrity"]["cache"]["status"] == "passed"
    assert result["workspace"]["formal_store_modified"] is False
    assert result["workspace"]["removed_after_benchmark"] is True
    assert not Path(result["workspace"]["temporary_path"]).exists()
    assert store.active_state() == source_identity


def test_incremental_benchmark_avoids_unsafe_windows_workspace_path(
    tmp_path,
    monkeypatch,
):
    source_root = tmp_path / "source"
    store = MarketDailyStore(source_root)
    for date, close in (("2026-07-01", 10.0), ("2026-07-02", 10.2)):
        store.commit_partition(
            _frame(date, close),
            instrument_type="equity",
            source="test",
        )
    deep_parent = tmp_path.joinpath(*(["long-segment"] * 12))
    monkeypatch.setattr(
        "data.market_daily_incremental_benchmark.os.name",
        "nt",
    )

    result = benchmark_daily_incremental_refresh(
        source_root,
        month="2026-07",
        workspace_parent=deep_parent,
    )

    assert result["status"] == "passed"
    assert result["workspace"]["fallback_reason"] == (
        "requested_parent_would_exceed_windows_safe_path"
    )
