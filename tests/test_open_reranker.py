import pandas as pd
import pytest

from core.research_protocol import FORWARD_START_DATE

from core.research_protocol import assert_alpha_rows_within_forward
from run.build_reranker_v2_dataset import stock_open_label_rows
from run.train_open_reranker_current_v9 import add_open_label, load_training_frame


def test_open_to_open_executable_label_uses_next_open_entry(tmp_path):
    dates = pd.bdate_range("2023-01-02", periods=15)
    opens = pd.Series(range(100, 115), dtype=float)
    pd.DataFrame(
        {
            "trade_date": dates,
            "open": opens,
            "low": opens * 0.90,
            "close": opens,
        }
    ).to_csv(tmp_path / "A.csv", index=False)

    rows = stock_open_label_rows(
        tmp_path / "A.csv",
        {dates[0]},
        pd.Timestamp("2023-12-31"),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["exec_return_1d"] == pytest.approx(102.0 / 101.0 - 1.0)
    assert row["exec_signal_to_entry"] == pytest.approx(101.0 / 100.0 - 1.0)
    assert row["exec_max_downside"] == pytest.approx(0.10)


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
    with pytest.raises(ValueError, match="before .*forward start"):
        assert_alpha_rows_within_forward(
            [{"date": str((FORWARD_START_DATE - pd.Timedelta(days=1)).date()), "codes": ["A"]}],
            context="open-reranker forward input",
        )
