from pathlib import Path

import pandas as pd

from run.audit_open_ledger_backend_parity import (
    ARTIFACT_COLUMNS,
    compare_sweep_roots,
)


def _write_sweep(root: Path, *, backend: str, changed_order: bool = False):
    root.mkdir(parents=True)
    summary = pd.DataFrame(
        [
            {
                "ann": 10.0,
                "sharpe": 1.0,
                "ohlc_backend": backend,
                "market_daily_store_root": "store" if backend == "monthly" else "",
                "ohlc_monthly_cache_dir": "cache" if backend == "monthly" else "",
            }
        ]
    )
    summary.to_csv(root / "open_price_ledger_param_sweep_summary.csv", index=False)
    row = {"sweep_key_sha256": "abc"}
    for artifact in ARTIFACT_COLUMNS:
        path = root / f"{artifact}.csv"
        value = 2.0 if changed_order and artifact == "orders" else 1.0
        pd.DataFrame([{"date": "2025-01-02", "value": value}]).to_csv(
            path, index=False
        )
        row[artifact] = str(path.resolve())
    pd.DataFrame([row]).to_csv(root / "path_artifact_index.csv", index=False)


def test_backend_parity_ignores_only_lineage_columns(tmp_path):
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    _write_sweep(oracle, backend="csv")
    _write_sweep(candidate, backend="monthly")

    report = compare_sweep_roots(oracle, candidate)

    assert report["status"] == "passed"
    assert report["artifacts"]["comparison_count"] == 6


def test_backend_parity_detects_order_difference(tmp_path):
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    _write_sweep(oracle, backend="csv")
    _write_sweep(candidate, backend="monthly", changed_order=True)

    report = compare_sweep_roots(oracle, candidate)

    assert report["status"] == "failed"
    assert [item["artifact"] for item in report["artifacts"]["failures"]] == [
        "orders"
    ]
