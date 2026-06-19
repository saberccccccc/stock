import pandas as pd
import pytest

from alpha.diagnostics import execution_quality_daily, summarize_execution_quality


def test_execution_quality_tracks_chase_share_and_overlap():
    rows = [
        {"date": "2024-01-02", "codes": ["A", "B", "C"]},
        {"date": "2024-01-03", "codes": ["B", "C", "D"]},
    ]
    returns = {
        pd.Timestamp("2024-01-02"): {"A": 0.10, "B": 0.08, "C": 0.0},
        pd.Timestamp("2024-01-03"): {"B": 0.0, "C": 0.10, "D": -0.01},
    }

    daily = execution_quality_daily("candidate", rows, returns, top_n=3)
    summary = summarize_execution_quality(daily).iloc[0]

    assert daily.iloc[0]["share_ge_095"] == pytest.approx(1 / 3)
    assert daily.iloc[1]["overlap_previous"] == pytest.approx(2 / 3)
    assert summary["dates"] == 2
    assert summary["share_ge_070"] == pytest.approx(0.5)


def test_execution_quality_rejects_nonpositive_top_n():
    with pytest.raises(ValueError, match="positive"):
        execution_quality_daily("candidate", [], {}, top_n=0)


def test_summarize_execution_quality_handles_empty_input():
    summary = summarize_execution_quality(pd.DataFrame())

    assert summary.empty
    assert "share_ge_095" in summary.columns
