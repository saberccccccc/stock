import numpy as np
import pandas as pd

from run.compare_alpha_open_returns import ranked_open_return, summarize_daily


def test_ranked_open_return_respects_signal_lag_and_horizon():
    dates = pd.bdate_range("2024-01-02", periods=5)
    opens = pd.DataFrame(
        {"a": [10.0, 11.0, 12.0, 15.0, 18.0], "b": [20.0, 20.0, 22.0, 22.0, 22.0]},
        index=dates,
    )
    row = {"date": dates[0], "codes": ["a", "b"]}

    expected_now = np.mean([12.0 / 11.0 - 1.0, 22.0 / 20.0 - 1.0])
    expected_lag1 = np.mean([15.0 / 12.0 - 1.0, 22.0 / 22.0 - 1.0])
    assert np.isclose(ranked_open_return(row, opens, 2, 1, 0), expected_now)
    assert np.isclose(ranked_open_return(row, opens, 2, 1, 1), expected_lag1)


def test_summarize_daily_reports_month_stability():
    daily = pd.DataFrame({
        "model": ["m"] * 4,
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"]),
        "execution_lag": [0] * 4,
        "horizon": [1] * 4,
        "return": [0.01, 0.02, -0.01, -0.02],
    })
    summary = summarize_daily(daily).iloc[0]
    assert summary["positive_day_rate"] == 0.5
    assert summary["positive_month_rate"] == 0.5
