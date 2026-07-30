"""Evaluate fixed MD8 market-data performance evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MEASURED_SPLITS = ("test_2025", "forward_2026")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def evaluate_market_data_performance(
    evidence_root: str | Path,
    *,
    incremental_evidence: str | Path | None = None,
    min_ohlc_speedup: float = 3.0,
    min_total_speedup: float = 2.0,
    max_monthly_data_fraction: float = 0.20,
    max_rss_mb: float = 1024.0,
    max_daily_commit_seconds: float = 10.0,
    max_cache_refresh_seconds: float = 30.0,
    max_daily_commit_read_count: int = 4999,
) -> dict[str, Any]:
    root = Path(evidence_root).resolve()
    matrix = _load(root / "matrix_status.json")
    comparisons = {item["split"]: item["status"] for item in matrix["comparisons"]}
    parity_passed = all(
        comparisons.get(split) == "passed"
        for split in ("val_2024", "test_2025", "forward_2026")
    )

    rows = []
    for split in MEASURED_SPLITS:
        csv = _load(root / "csv" / split / "performance.json")
        monthly = _load(root / "monthly" / split / "performance.json")
        total_speedup = csv["total_seconds"] / monthly["total_seconds"]
        ohlc_speedup = csv["ohlc_load_seconds"] / monthly["ohlc_load_seconds"]
        monthly_data_fraction = monthly["ohlc_load_seconds"] / monthly["total_seconds"]
        bottleneck_proven = (
            ohlc_speedup >= min_ohlc_speedup
            and monthly_data_fraction <= max_monthly_data_fraction
        )
        rows.append(
            {
                "split": split,
                "csv_total_seconds": csv["total_seconds"],
                "monthly_total_seconds": monthly["total_seconds"],
                "total_speedup": total_speedup,
                "csv_ohlc_seconds": csv["ohlc_load_seconds"],
                "monthly_ohlc_seconds": monthly["ohlc_load_seconds"],
                "ohlc_speedup": ohlc_speedup,
                "monthly_data_fraction": monthly_data_fraction,
                "ledger_bottleneck_proven": bottleneck_proven,
                "runtime_gate_passed": (
                    total_speedup >= min_total_speedup or bottleneck_proven
                ),
                "monthly_rss_mb": monthly["rss_mb"],
                "memory_gate_passed": monthly["rss_mb"] <= max_rss_mb,
                "process_io_recorded": bool(monthly.get("process_io")),
            }
        )

    runtime_passed = all(row["runtime_gate_passed"] for row in rows)
    memory_passed = all(row["memory_gate_passed"] for row in rows)
    io_recorded = all(row["process_io_recorded"] for row in rows)
    incremental = None
    incremental_recorded = incremental_evidence is not None
    incremental_passed = False
    if incremental_recorded:
        incremental_path = Path(incremental_evidence).resolve()
        benchmark = _load(incremental_path)
        commit = benchmark["daily_commit"]
        refresh = benchmark["monthly_cache_refresh"]
        source = benchmark["source"]
        cache_hit = benchmark["cache_hit_observation"]
        integrity = benchmark["integrity"]
        workspace = benchmark["workspace"]
        incremental_gates = {
            "benchmark_passed": (
                benchmark.get("schema") == "market_daily_incremental_benchmark_v1"
                and benchmark.get("status") == "passed"
            ),
            "daily_scale_passed": int(commit["rows"]) >= 5000,
            "daily_commit_time_passed": (
                float(commit["elapsed_seconds"]) <= max_daily_commit_seconds
            ),
            "no_per_stock_scan_passed": (
                int(commit["process_io"]["read_count"])
                <= max_daily_commit_read_count
            ),
            "cache_refresh_time_passed": (
                float(refresh["elapsed_seconds"]) <= max_cache_refresh_seconds
            ),
            "warm_cache_passed": (
                cache_hit["requests"] >= 2
                and cache_hit["hits"] == cache_hit["requests"]
                and benchmark["warm_cache_after_refresh"]["result_status"]
                == "already_current"
            ),
            "partition_coverage_passed": (
                source["partitions_loaded"] == source["partitions_requested"]
                and source["partition_hit_rate"] == 1.0
                and source["arbitrary_range_reads"]["exact_frame_parity"]
            ),
            "integrity_passed": (
                integrity["store"]["status"] == "passed"
                and integrity["cache"]["status"] == "passed"
            ),
            "isolation_passed": (
                workspace["isolated"]
                and not workspace["formal_store_modified"]
                and workspace["removed_after_benchmark"]
            ),
        }
        incremental_passed = all(incremental_gates.values())
        incremental = {
            "path": str(incremental_path),
            "gates": incremental_gates,
            "daily_commit_seconds": commit["elapsed_seconds"],
            "daily_commit_rows": commit["rows"],
            "daily_commit_process_io": commit["process_io"],
            "cache_refresh_seconds": refresh["elapsed_seconds"],
            "cache_shape": refresh["shape"],
            "cache_hit_rate": cache_hit["hit_rate"],
            "rss_mb_peak_observed": (
                benchmark["resources"]["rss_bytes_peak_observed"] / (1024**2)
            ),
            "network_acquisition": benchmark["network_acquisition"],
        }
    status = (
        "passed"
        if (
            parity_passed
            and runtime_passed
            and memory_passed
            and io_recorded
            and incremental_passed
        )
        else "provisional_pass"
        if (
            parity_passed
            and runtime_passed
            and memory_passed
            and (not incremental_recorded or incremental_passed)
        )
        else "failed"
    )
    return {
        "schema": "market_data_performance_acceptance_v1",
        "status": status,
        "evidence_root": str(root),
        "gates": {
            "parity_passed": parity_passed,
            "runtime_passed": runtime_passed,
            "memory_passed": memory_passed,
            "process_io_recorded": io_recorded,
            "incremental_evidence_recorded": incremental_recorded,
            "incremental_passed": incremental_passed,
        },
        "thresholds": {
            "min_ohlc_speedup": min_ohlc_speedup,
            "min_total_speedup": min_total_speedup,
            "max_monthly_data_fraction": max_monthly_data_fraction,
            "max_rss_mb": max_rss_mb,
            "max_daily_commit_seconds": max_daily_commit_seconds,
            "max_cache_refresh_seconds": max_cache_refresh_seconds,
            "max_daily_commit_read_count": max_daily_commit_read_count,
        },
        "splits": rows,
        "incremental": incremental,
    }
