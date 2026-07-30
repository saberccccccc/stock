import json

from data.market_data_performance_acceptance import evaluate_market_data_performance


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _evidence(tmp_path, *, io=True, monthly_ohlc=4.0, monthly_total=30.0):
    root = tmp_path / "evidence"
    _write(
        root / "matrix_status.json",
        {
            "comparisons": [
                {"split": split, "status": "passed"}
                for split in ("val_2024", "test_2025", "forward_2026")
            ]
        },
    )
    for split in ("test_2025", "forward_2026"):
        _write(
            root / "csv" / split / "performance.json",
            {
                "total_seconds": 60.0,
                "ohlc_load_seconds": 30.0,
                "rss_mb": 450.0,
            },
        )
        _write(
            root / "monthly" / split / "performance.json",
            {
                "total_seconds": monthly_total,
                "ohlc_load_seconds": monthly_ohlc,
                "rss_mb": 520.0,
                "process_io": {"read_count": 10} if io else None,
            },
        )
    return root


def _incremental_evidence(tmp_path, *, commit_seconds=0.1):
    path = tmp_path / "incremental.json"
    _write(
        path,
        {
            "schema": "market_daily_incremental_benchmark_v1",
            "status": "passed",
            "source": {
                "partitions_loaded": 21,
                "partitions_requested": 21,
                "partition_hit_rate": 1.0,
                "arbitrary_range_reads": {"exact_frame_parity": True},
            },
            "network_acquisition": {"status": "not_measured_local_replay"},
            "daily_commit": {
                "rows": 5300,
                "elapsed_seconds": commit_seconds,
                "process_io": {"read_count": 20, "read_bytes": 100},
            },
            "monthly_cache_refresh": {
                "elapsed_seconds": 0.2,
                "shape": [21, 5313],
            },
            "warm_cache_after_refresh": {"result_status": "already_current"},
            "cache_hit_observation": {
                "hits": 2,
                "requests": 2,
                "hit_rate": 1.0,
            },
            "integrity": {
                "store": {"status": "passed"},
                "cache": {"status": "passed"},
            },
            "workspace": {
                "isolated": True,
                "formal_store_modified": False,
                "removed_after_benchmark": True,
            },
            "resources": {"rss_bytes_peak_observed": 200 * 1024**2},
        },
    )
    return path


def test_acceptance_passes_with_io_and_fixed_gates(tmp_path):
    result = evaluate_market_data_performance(
        _evidence(tmp_path),
        incremental_evidence=_incremental_evidence(tmp_path),
    )

    assert result["status"] == "passed"
    assert result["gates"] == {
        "parity_passed": True,
        "runtime_passed": True,
        "memory_passed": True,
        "process_io_recorded": True,
        "incremental_evidence_recorded": True,
        "incremental_passed": True,
    }


def test_acceptance_is_provisional_for_old_evidence_without_io(tmp_path):
    result = evaluate_market_data_performance(
        _evidence(tmp_path, io=False),
        incremental_evidence=_incremental_evidence(tmp_path),
    )

    assert result["status"] == "provisional_pass"


def test_acceptance_fails_when_data_layer_remains_the_bottleneck(tmp_path):
    result = evaluate_market_data_performance(
        _evidence(tmp_path, monthly_ohlc=20.0, monthly_total=45.0)
    )

    assert result["status"] == "failed"
    assert result["gates"]["runtime_passed"] is False


def test_acceptance_is_provisional_without_incremental_evidence(tmp_path):
    result = evaluate_market_data_performance(_evidence(tmp_path))

    assert result["status"] == "provisional_pass"
    assert result["gates"]["incremental_evidence_recorded"] is False


def test_acceptance_fails_slow_daily_commit(tmp_path):
    result = evaluate_market_data_performance(
        _evidence(tmp_path),
        incremental_evidence=_incremental_evidence(
            tmp_path,
            commit_seconds=11.0,
        ),
    )

    assert result["status"] == "failed"
    assert result["gates"]["incremental_passed"] is False
    assert result["incremental"]["gates"]["daily_commit_time_passed"] is False
