import json

import pandas as pd

from run.run_market_data_dual_read_matrix import (
    _filter_alpha_rows,
    _report_passed,
    parse_args,
)


def test_dual_read_matrix_defaults_keep_three_gib_gate():
    args = parse_args([])

    assert args.min_free_memory_gib == 3.0
    assert args.estimated_peak_memory_gib == 0.75
    assert args.research_data_dir == "data/raw"
    assert args.forward_data_dir == "data/forward_raw"


def test_report_passed_requires_explicit_pass(tmp_path):
    path = tmp_path / "report.json"
    assert _report_passed(path) is False

    path.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    assert _report_passed(path) is False

    path.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    assert _report_passed(path) is True


def test_filter_alpha_rows_accepts_string_and_timestamp_dates():
    rows = [
        {"date": "2025-01-01", "codes": []},
        {"date": pd.Timestamp("2025-06-30"), "codes": []},
        {"date": "2026-01-01", "codes": []},
    ]

    selected = _filter_alpha_rows(
        rows,
        start_date="2025-01-01",
        end_date="2025-12-31",
    )

    assert [str(pd.Timestamp(row["date"]).date()) for row in selected] == [
        "2025-01-01",
        "2025-06-30",
    ]
