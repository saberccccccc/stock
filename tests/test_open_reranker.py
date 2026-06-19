import pandas as pd
import pytest

from core.research_protocol import assert_alpha_rows_within_forward
from run.train_open_reranker_current_v9 import add_open_label, load_training_frame


def test_open_label_purges_cross_year_and_tail_rows(tmp_path):
    dates = pd.bdate_range("2023-12-20", "2024-01-10")
    pd.DataFrame(
        {
            "trade_date": dates,
            "open": range(100, 100 + len(dates)),
        }
    ).to_csv(tmp_path / "A.csv", index=False)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-20", "2023-12-27", "2024-01-10"]),
            "code": ["A", "A", "A"],
        }
    )

    labelled = add_open_label(frame, tmp_path, horizon=3)

    assert labelled["date"].dt.strftime("%Y-%m-%d").tolist() == ["2023-12-20"]
    assert labelled["open_h5_label"].notna().all()


def test_training_frame_rejects_rows_after_oof_cutoff(tmp_path, monkeypatch):
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-29", "2024-01-02"]),
            "code": ["A", "B"],
            "candidate_position": [0, 1],
        }
    ).to_parquet(tmp_path / "rows.parquet", index=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="exceed OOF cutoff"):
        load_training_frame("*.parquet", max_rows=100)


def test_forward_rows_reject_historical_dates():
    with pytest.raises(ValueError, match="before forward-test start"):
        assert_alpha_rows_within_forward(
            [{"date": "2026-05-18", "codes": ["A"]}],
            context="open-reranker forward input",
        )
