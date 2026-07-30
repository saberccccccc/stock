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
    min_ohlc_speedup: float = 3.0,
    min_total_speedup: float = 2.0,
    max_monthly_data_fraction: float = 0.20,
    max_rss_mb: float = 1024.0,
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
    status = (
        "passed"
        if parity_passed and runtime_passed and memory_passed and io_recorded
        else "provisional_pass"
        if parity_passed and runtime_passed and memory_passed
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
        },
        "thresholds": {
            "min_ohlc_speedup": min_ohlc_speedup,
            "min_total_speedup": min_total_speedup,
            "max_monthly_data_fraction": max_monthly_data_fraction,
            "max_rss_mb": max_rss_mb,
        },
        "splits": rows,
    }
