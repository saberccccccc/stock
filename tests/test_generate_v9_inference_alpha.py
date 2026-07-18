import pandas as pd

from run.generate_v9_inference_alpha import filter_rows_for_emit_start


def test_filter_rows_keeps_warmup_internal_to_predictor():
    rows = [
        {"date": pd.Timestamp("2024-01-30"), "alpha": [1.0]},
        {"date": pd.Timestamp("2024-01-31"), "alpha": [2.0]},
        {"date": pd.Timestamp("2024-02-01"), "alpha": [3.0]},
    ]

    emitted = filter_rows_for_emit_start(rows, "2024-02-01")

    assert [row["date"] for row in emitted] == [pd.Timestamp("2024-02-01")]
