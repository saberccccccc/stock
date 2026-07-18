import pandas as pd
import pytest

from experiments.leaderboard import (
    build_leaderboard,
    capital_label,
    infer_split,
    rows_from_summary,
    write_leaderboard,
)
from experiments.registry import CandidateSpec, ResultSource, default_registry


def _write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(portfolio_value=500_000, n_return_days=243, ann=10.0, sharpe=1.2):
    return {
        "portfolio_value": portfolio_value,
        "signal_start": "2026-01-05",
        "signal_end": "2026-06-30",
        "backtest_start": "2026-01-06",
        "backtest_end": "2026-06-30",
        "n_return_days": n_return_days,
        "target_frac": 0.006,
        "hold_frac": 0.1,
        "ann": ann,
        "sharpe": sharpe,
        "mdd": 0.12,
        "avg_executed_turnover": 0.34,
        "blocked_buy": 2,
    }


def test_capital_label_formats_wan_units():
    assert capital_label(500_000) == "50w"
    assert capital_label(1_000_000) == "100w"
    assert capital_label(123) == "123"


def test_infer_split_uses_return_day_count():
    assert infer_split({"n_return_days": 243}) == "val"
    assert infer_split({"n_return_days": 328}) == "test"


def test_rows_from_summary_uses_explicit_split_and_stress(tmp_path):
    summary = tmp_path / "summary.csv"
    _write_summary(summary, [_row()])

    rows = rows_from_summary(summary, "demo", split="forward", stress="lag1")

    assert rows == [
        {
            "candidate": "demo",
            "split": "forward",
            "stress": "lag1",
            "signal_start": "2026-01-05",
            "signal_end": "2026-06-30",
            "backtest_start": "2026-01-06",
            "backtest_end": "2026-06-30",
            "capital": "50w",
            "ann": 10.0,
            "sharpe": 1.2,
            "mdd": 0.12,
            "exec_to": 0.34,
            "blocked_buy": 2,
            "evidence_class": "legacy_registered",
            "formal_eligible": False,
        }
    ]


def test_rows_from_summary_prefers_file_split_over_day_count(tmp_path):
    summary = tmp_path / "summary.csv"
    _write_summary(summary, [{**_row(n_return_days=20), "split": "forward"}])

    rows = rows_from_summary(summary, "demo")

    assert rows[0]["split"] == "forward"


def test_rows_from_summary_filters_grid_rows(tmp_path):
    summary = tmp_path / "summary.csv"
    _write_summary(
        summary,
        [
            _row(ann=10.0),
            {**_row(ann=20.0), "hold_frac": 0.06},
            {**_row(ann=30.0), "target_frac": 0.004},
        ],
    )

    rows = rows_from_summary(
        summary,
        "demo",
        target_frac=0.006,
        hold_frac=0.1,
    )

    assert len(rows) == 1
    assert rows[0]["ann"] == 10.0


def test_rows_from_summary_rejects_missing_coverage_columns(tmp_path):
    summary = tmp_path / "summary.csv"
    _write_summary(summary, [{"portfolio_value": 500_000, "n_return_days": 10}])

    with pytest.raises(ValueError, match="missing required coverage columns"):
        rows_from_summary(summary, "demo")


def test_build_leaderboard_records_missing_sources(tmp_path):
    summary = tmp_path / "summary.csv"
    _write_summary(summary, [_row(), _row(portfolio_value=1_000_000, n_return_days=328, ann=20.0)])
    candidate = CandidateSpec(
        name="demo",
        status="research",
        execution_family="test_family",
        description="demo",
        result_sources=(
            ResultSource(path="summary.csv", split=None, stress="normal", target_frac=0.006),
            ResultSource(path="missing.csv", split="val", stress="normal"),
        ),
    )

    frame, missing = build_leaderboard([candidate], root=tmp_path)

    assert len(frame) == 2
    assert frame["split"].tolist() == ["val", "test"]
    assert frame["capital"].tolist() == ["50w", "100w"]
    assert missing == [{"candidate": "demo", "path": str(tmp_path / "missing.csv")}]


def test_write_leaderboard_creates_csv(tmp_path):
    summary = tmp_path / "summary.csv"
    output = tmp_path / "out" / "leaderboard.csv"
    _write_summary(summary, [_row()])
    candidate = CandidateSpec(
        name="demo",
        status="research",
        execution_family="test_family",
        description="demo",
        result_sources=(ResultSource(path="summary.csv"),),
    )

    frame, missing = write_leaderboard(output, [candidate], root=tmp_path)

    assert missing == []
    assert output.exists()
    loaded = pd.read_csv(output)
    assert loaded.to_dict("records") == frame.to_dict("records")


def test_default_registry_contains_current_main_candidates():
    registry = default_registry()

    assert set(registry) >= {
        "official",
        "breadth_m085",
        "edge_r030_100",
        "negfilter_drop3",
        "risk_target_r004",
    }
    assert registry["official"].status == "official"
