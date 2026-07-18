import json

from backtest.execution_coverage import audit_execution_coverage
from run.audit_execution_coverage import parse_args, resolve_data_dir


def test_audit_cli_resolves_role_specific_default_cache():
    args = parse_args(["--dataset-role", "forward"])

    assert args.data_dir is None
    assert resolve_data_dir(args.data_dir, args.dataset_role).as_posix() == "data/forward_raw"


def test_audit_cli_keeps_explicit_cache_override():
    args = parse_args(["--dataset-role", "forward", "--data-dir", "custom/cache"])

    assert resolve_data_dir(args.data_dir, args.dataset_role).as_posix() == "custom/cache"


def test_audit_declares_missing_historical_st_coverage(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "stable_stocks.csv").write_text("ts_code,name,list_date\n000001.SZ,Demo,20100101\n", encoding="utf-8")
    (tmp_path / "stock_industry.csv").write_text(
        "updateDate,code,code_name\n2026-04-27,000001.SZ,Demo\n", encoding="utf-8"
    )
    cache = tmp_path / "matrix"
    cache.mkdir()
    (cache / "ohlc_matrix_meta.json").write_text(
        json.dumps({"source_count": 1, "fields": ["open", "high", "low", "close", "volume", "money"], "dates": ["2024-01-02", "2025-12-31"]}),
        encoding="utf-8",
    )

    report = audit_execution_coverage(raw, cache, start_date="2024-01-01", end_date="2025-12-31")

    assert report["status"] == "audited_with_declared_gaps"
    assert "historical_st_status_not_covered" in report["gaps"]
    assert report["ohlc_matrix"]["date_coverage"]["start"] == "2024-01-02"
