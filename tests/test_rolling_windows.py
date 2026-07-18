import pandas as pd
import pytest

from experiments.rolling import (
    MonthlyRollingSpec,
    RollingWindow,
    assert_unique_oos_owners,
    build_monthly_windows,
    label_safe_indices,
    resolve_window_indices,
    validate_window,
)


def test_label_safe_indices_purges_incomplete_tail_labels():
    dates = pd.date_range("2024-01-02", periods=5, freq="B")

    assert label_safe_indices(dates, "2024-01-02", "2024-01-08", 2) == [0, 1, 2]


def test_window_indices_keep_prediction_dates_but_purge_training_labels():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    window = RollingWindow(
        "demo", "2020-01-01", "2020-01-10", "2020-01-13", "2020-01-17",
        "2020-01-20", "2020-01-28",
    )

    indices = resolve_window_indices(dates, window, label_end_offset=2)

    assert len(indices["train"]) < len([d for d in dates if d <= pd.Timestamp("2020-01-10")])
    assert len(indices["valid"]) < len([d for d in dates if pd.Timestamp("2020-01-13") <= d <= pd.Timestamp("2020-01-17")])
    assert indices["predict"][-1] == len(dates) - 1


def test_window_rejects_overlap_and_forward_dates():
    overlap = RollingWindow("bad", "2024-01-01", "2024-06-01", "2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01")
    forward = RollingWindow("forward", "2024-01-01", "2024-06-01", "2024-07-01", "2024-08-01", "2026-05-19", "2026-06-01")

    with pytest.raises(ValueError, match="overlapping"):
        validate_window(overlap)
    with pytest.raises(ValueError, match="frozen research"):
        validate_window(forward)


def test_monthly_windows_use_trading_calendar_and_fixed_history():
    dates = pd.date_range("2018-01-01", "2024-08-30", freq="B")

    windows = build_monthly_windows(dates, MonthlyRollingSpec(4, 6, "2024-07-01", "2024-08-30"))

    assert [window.name for window in windows] == ["oos_2024_07", "oos_2024_08"]
    july = windows[0]
    assert july.train_start == "2020-01-01"
    assert july.train_end == "2023-12-29"
    assert july.valid_start == "2024-01-01"
    assert july.valid_end == "2024-06-28"
    assert july.predict_start == "2024-07-01"
    assert july.predict_end == "2024-07-31"


def test_monthly_windows_reject_forward_and_duplicate_oos_owners():
    dates = pd.date_range("2018-01-01", "2026-06-30", freq="B")
    with pytest.raises(ValueError, match="frozen research boundary"):
        build_monthly_windows(dates, MonthlyRollingSpec(4, 6, "2026-05-01", "2026-05-29"))

    overlapping = [
        RollingWindow("a", "2020-01-01", "2023-01-01", "2023-01-02", "2023-06-01", "2024-01-01", "2024-01-31"),
        RollingWindow("b", "2020-02-01", "2023-02-01", "2023-02-02", "2023-07-01", "2024-01-15", "2024-02-15"),
    ]
    with pytest.raises(ValueError, match="duplicate OOS owner"):
        assert_unique_oos_owners(dates, overlapping)
