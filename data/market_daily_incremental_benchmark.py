"""Isolated benchmark for one-day market-data commit and cache refresh."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import psutil

from backtest.monthly_ohlcv_cache import MonthlyOhlcvCache
from data.market_daily_store import MarketDailyStore, NUMERIC_FIELDS


def _resource_snapshot() -> dict[str, Any]:
    process = psutil.Process()
    io = process.io_counters()
    memory = psutil.virtual_memory()
    return {
        "rss_bytes": int(process.memory_info().rss),
        "system_available_bytes": int(memory.available),
        "process_io": {
            "read_count": int(io.read_count),
            "write_count": int(io.write_count),
            "read_bytes": int(io.read_bytes),
            "write_bytes": int(io.write_bytes),
        },
    }


def _io_delta(start: dict[str, Any], end: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(end["process_io"][key] - start["process_io"][key])
        for key in start["process_io"]
    }


def _measure(action: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    before = _resource_snapshot()
    rss_samples = [before["rss_bytes"]]
    available_samples = [before["system_available_bytes"]]
    stop = threading.Event()

    def sample_resources() -> None:
        process = psutil.Process()
        while not stop.wait(0.005):
            rss_samples.append(int(process.memory_info().rss))
            available_samples.append(int(psutil.virtual_memory().available))

    sampler = threading.Thread(
        target=sample_resources,
        name="market-data-benchmark-resource-sampler",
        daemon=True,
    )
    sampler.start()
    started = time.perf_counter()
    try:
        value = action()
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        sampler.join()
    after = _resource_snapshot()
    rss_samples.append(after["rss_bytes"])
    available_samples.append(after["system_available_bytes"])
    return value, {
        "elapsed_seconds": elapsed,
        "rss_bytes_before": before["rss_bytes"],
        "rss_bytes_after": after["rss_bytes"],
        "rss_bytes_peak_observed": max(rss_samples),
        "resource_sample_count": len(rss_samples),
        "resource_sample_interval_seconds": 0.005,
        "system_available_bytes_before": before["system_available_bytes"],
        "system_available_bytes_after": after["system_available_bytes"],
        "system_available_bytes_min_observed": min(available_samples),
        "process_io": _io_delta(before, after),
    }


def _tree_stats(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def benchmark_daily_incremental_refresh(
    source_store_root: str | Path,
    *,
    month: str | None = None,
    workspace_parent: str | Path | None = None,
) -> dict[str, Any]:
    """Replay one real month in isolation and measure only incremental work."""
    source_store = MarketDailyStore(source_store_root)
    source_state = source_store.active_state()
    equity_coverage = source_state["coverage"].get("equity")
    if not equity_coverage:
        raise ValueError("source store has no equity coverage")
    selected_month = pd.Period(
        month or pd.Timestamp(equity_coverage["date_end"]).strftime("%Y-%m"),
        freq="M",
    )
    source_month = source_store.month_snapshot(
        instrument_type="equity",
        month=selected_month,
    )

    month_frame, source_load = _measure(
        lambda: source_store.load(
            instrument_type="equity",
            start_date=selected_month.start_time,
            end_date=selected_month.end_time,
            fields=NUMERIC_FIELDS,
        )
    )
    warm_source_frame, source_warm_load = _measure(
        lambda: source_store.load(
            instrument_type="equity",
            start_date=selected_month.start_time,
            end_date=selected_month.end_time,
            fields=NUMERIC_FIELDS,
        )
    )
    source_read_parity = warm_source_frame.equals(month_frame)
    del warm_source_frame
    dates = pd.DatetimeIndex(sorted(month_frame["trade_date"].unique()))
    if len(dates) < 2:
        raise ValueError("benchmark month requires at least two trading dates")
    latest_date = dates[-1]
    setup_frame = month_frame.loc[month_frame["trade_date"] < latest_date]
    latest_frame = month_frame.loc[month_frame["trade_date"] == latest_date]

    requested_parent = (
        Path(workspace_parent).resolve() if workspace_parent else None
    )
    workspace_fallback_reason = None
    # A content-addressed daily partition adds roughly 140 characters below
    # the workspace. Keep headroom for Win32 APIs that still enforce MAX_PATH.
    unsafe_requested_parent = (
        requested_parent is not None
        and os.name == "nt"
        and len(str(requested_parent)) + 160 >= 240
    )
    if unsafe_requested_parent:
        workspace = Path(tempfile.mkdtemp(prefix="mdi-")).resolve()
        workspace_fallback_reason = "requested_parent_would_exceed_windows_safe_path"
    else:
        if requested_parent is not None:
            requested_parent.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(prefix="mdi-", dir=requested_parent)
        ).resolve()
    store_root = workspace / "store"
    cache_root = workspace / "cache"
    result: dict[str, Any] = {}
    try:
        store = MarketDailyStore(store_root)
        cache = MonthlyOhlcvCache(store_root=store_root, cache_root=cache_root)

        setup_started = time.perf_counter()
        for _, frame in setup_frame.groupby("trade_date", sort=True):
            store.commit_partition(
                frame,
                instrument_type="equity",
                source="benchmark_replay",
            )
        initial_cache = cache.ensure_month(selected_month)
        setup_seconds = time.perf_counter() - setup_started

        warm_before, warm_before_metrics = _measure(
            lambda: cache.ensure_month(selected_month)
        )
        commit, commit_metrics = _measure(
            lambda: store.commit_partition(
                latest_frame,
                instrument_type="equity",
                source="benchmark_replay",
            )
        )
        refreshed, refresh_metrics = _measure(
            lambda: cache.ensure_month(selected_month)
        )
        warm_after, warm_after_metrics = _measure(
            lambda: cache.ensure_month(selected_month)
        )
        store_audit, store_audit_metrics = _measure(
            lambda: store.audit(verify_physical_hashes=True)
        )
        cache_audit, cache_audit_metrics = _measure(
            lambda: cache.audit_month(
                selected_month,
                verify_file_hashes=True,
            )
        )

        measured = (
            source_load,
            source_warm_load,
            warm_before_metrics,
            commit_metrics,
            refresh_metrics,
            warm_after_metrics,
            store_audit_metrics,
            cache_audit_metrics,
        )
        result = {
            "schema": "market_daily_incremental_benchmark_v1",
            "status": (
                "passed"
                if commit.status == "committed"
                and initial_cache["status"] == "rebuilt"
                and warm_before["status"] == "already_current"
                and refreshed["status"] == "rebuilt"
                and warm_after["status"] == "already_current"
                and store_audit["status"] == "passed"
                and cache_audit["status"] == "passed"
                else "failed"
            ),
            "source": {
                "store_root": str(source_store.root),
                "manifest_sha256": source_state["manifest_sha256"],
                "month": str(selected_month),
                "date_start": str(dates[0].date()),
                "date_end": str(latest_date.date()),
                "dates": len(dates),
                "rows": len(month_frame),
                "partitions_requested": len(
                    source_month["index"]["partitions"]
                ),
                "partitions_loaded": len(dates),
                "partition_hit_rate": (
                    len(dates) / len(source_month["index"]["partitions"])
                ),
                "arbitrary_range_reads": {
                    "first": {
                        "cache_state": "os_cache_unspecified",
                        **source_load,
                    },
                    "warm": {
                        "cache_state": "same_process_warm",
                        **source_warm_load,
                    },
                    "exact_frame_parity": source_read_parity,
                },
            },
            "network_acquisition": {
                "status": "not_measured_local_replay",
                "reason": (
                    "Provider network latency is external to the local storage "
                    "commit and cache-refresh acceptance gates."
                ),
            },
            "setup": {
                "excluded_from_incremental_timing": True,
                "dates": len(dates) - 1,
                "rows": len(setup_frame),
                "elapsed_seconds": setup_seconds,
                "initial_cache_status": initial_cache["status"],
            },
            "daily_commit": {
                "trade_date": str(latest_date.date()),
                "rows": len(latest_frame),
                "result": asdict(commit),
                **commit_metrics,
            },
            "monthly_cache_refresh": {
                "month": str(selected_month),
                "result_status": refreshed["status"],
                "shape": refreshed["shape"],
                **refresh_metrics,
            },
            "warm_cache_before_commit": {
                "result_status": warm_before["status"],
                **warm_before_metrics,
            },
            "warm_cache_after_refresh": {
                "result_status": warm_after["status"],
                **warm_after_metrics,
            },
            "cache_hit_observation": {
                "hits": int(warm_before["status"] == "already_current")
                + int(warm_after["status"] == "already_current"),
                "requests": 2,
                "hit_rate": (
                    (
                        int(warm_before["status"] == "already_current")
                        + int(warm_after["status"] == "already_current")
                    )
                    / 2
                ),
            },
            "integrity": {
                "store": store_audit,
                "store_measurement": store_audit_metrics,
                "cache": cache_audit,
                "cache_measurement": cache_audit_metrics,
            },
            "workspace": {
                "isolated": True,
                "formal_store_modified": False,
                "temporary_path": str(workspace),
                "requested_parent": (
                    str(requested_parent) if requested_parent is not None else None
                ),
                "fallback_reason": workspace_fallback_reason,
                **_tree_stats(workspace),
            },
            "resources": {
                "rss_bytes_peak_observed": max(
                    item["rss_bytes_peak_observed"] for item in measured
                ),
                "system_available_bytes_min_observed": min(
                    item["system_available_bytes_min_observed"]
                    for item in measured
                ),
            },
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=False)

    result["workspace"]["removed_after_benchmark"] = not workspace.exists()
    if not result["workspace"]["removed_after_benchmark"]:
        result["status"] = "failed"
    return result
