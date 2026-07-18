import pytest

from data.boundary_audit import audit_data_root, resolve_data_dir


def _write_stock(path, last_date="2026-06-30"):
    path.write_text(
        "trade_date,ts_code,open,high,low,close,volume,money\n"
        f"2024-01-02,000001.SZ,1,1,1,1,100,100\n"
        f"{last_date},000001.SZ,1,1,1,1,100,100\n",
        encoding="utf-8",
    )


def test_research_audit_distinguishes_physical_superset_from_effective_view(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_stock(raw / "000001.SZ.csv")

    report = audit_data_root(raw, dataset_role="research", effective_end_date="2025-12-31")

    assert report["physical_cache_clean"] is False
    assert report["research_cutoff_violation_count"] == 1
    assert report["effective_view_safe"] is True
    assert report["status"] == "runtime_cutoff_required"


def test_research_audit_rejects_future_effective_end():
    with pytest.raises(ValueError, match="exceeds cutoff"):
        audit_data_root("data/raw", dataset_role="research", effective_end_date="2026-06-30")


def test_role_defaults_are_separate():
    assert resolve_data_dir(None, "research").as_posix() == "data/raw"
    assert resolve_data_dir(None, "forward").as_posix() == "data/forward_raw"
