from types import SimpleNamespace

from pathlib import Path

import pandas as pd

from run.audit_open_ledger_backend_parity import ARTIFACT_COLUMNS
from run.run_open_ledger_backend_parity_matrix import (
    CANDIDATE_ID,
    build_sweep_command,
    parse_args,
    sweep_is_complete,
)


def _args():
    return SimpleNamespace(
        research_data_dir="data/raw",
        forward_data_dir="data/forward_raw",
        market_daily_store_root="data/market_daily_candidate_v2",
        ohlc_monthly_cache_dir="cache/ohlcv_monthly_v3_candidate",
    )


def test_matrix_resource_gate_reserves_task_headroom():
    args = parse_args([])

    assert args.min_free_memory_gib == 3.0
    assert args.estimated_peak_memory_gib == 0.75


def test_matrix_command_freezes_fixed_cells_and_monthly_backend(tmp_path):
    command = build_sweep_command(
        _args(),
        split="test_2025",
        backend="monthly",
        output_dir=tmp_path,
    )

    assert command[0].endswith("python.exe")
    assert command[command.index("--start-date") + 1] == "2025-01-01"
    assert command[command.index("--end-date") + 1] == "2025-12-31"
    assert command[command.index("--stresses") + 1] == (
        "normal,lag1,cost2x,capacity_3pct"
    )
    assert command[command.index("--portfolio-values") + 1] == "500000,1000000"
    assert command[command.index("--ohlc-backend") + 1] == "monthly"
    assert "--save-path-details" in command
    assert "--resume" in command


def test_matrix_command_keeps_forward_observation_to_june_30(tmp_path):
    command = build_sweep_command(
        _args(),
        split="forward_2026",
        backend="csv",
        output_dir=tmp_path,
    )

    assert command[command.index("--data-dir") + 1] == "data/forward_raw"
    assert command[command.index("--end-date") + 1] == "2026-06-30"
    assert command[command.index("--max-data-date") + 1] == "2026-06-30"


def _write_completion_fixture(root: Path, row_count: int):
    root.mkdir()
    rows = []
    index_rows = []
    combinations = [
        (stress, capital)
        for stress in ("normal", "lag1", "cost2x", "capacity_3pct")
        for capital in (500000.0, 1000000.0)
    ]
    for number, (stress, capital) in enumerate(combinations[:row_count]):
        rows.append(
            {
                "alpha_name": CANDIDATE_ID,
                "stress": stress,
                "portfolio_value": capital,
            }
        )
        index_row = {"sweep_key_sha256": f"key-{number}"}
        for artifact in ARTIFACT_COLUMNS:
            path = root / f"{number}-{artifact}.csv"
            pd.DataFrame([{"value": number}]).to_csv(path, index=False)
            index_row[artifact] = str(path)
        index_rows.append(index_row)
    pd.DataFrame(rows).to_csv(
        root / "open_price_ledger_param_sweep_summary.csv", index=False
    )
    pd.DataFrame(index_rows).to_csv(root / "path_artifact_index.csv", index=False)


def test_matrix_completion_rejects_partial_resume_state(tmp_path):
    _write_completion_fixture(tmp_path / "partial", 7)
    _write_completion_fixture(tmp_path / "complete", 8)

    assert sweep_is_complete(tmp_path / "partial") is False
    assert sweep_is_complete(tmp_path / "complete") is True
