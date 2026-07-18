import numpy as np
import pandas as pd

from run.audit_strong_checkpoint_transfer import evaluate_alpha_rows, metric_transfer_summary


def test_evaluate_alpha_rows_reports_primary_lag1_and_turnover():
    labels = np.zeros((4, 3, 7), dtype=np.float32)
    labels[:, 0, [0, 2, 4, 6]] = np.asarray(
        [[0.04, 0.03, 0.02, 0.01], [0.03, 0.02, 0.01, 0.00], [0.02, 0.01, 0.00, -0.01], [0.01, 0.00, -0.01, -0.02]]
    )
    labels[:, 1, [0, 2, 4, 6]] = labels[:, 0, [0, 2, 4, 6]] * 0.5
    rows = [
        {"date": "2024-01-02", "codes": ["A", "B", "C", "D"], "alpha": [4.0, 3.0, 2.0, 1.0]},
    ]

    result = evaluate_alpha_rows(
        rows,
        labels=labels,
        lag1_shift=1,
        date_to_index={"2024-01-02": 0},
        code_to_index={"A": 0, "B": 1, "C": 2, "D": 3},
    )

    assert result["oos_days"] == 1
    assert result["oos_rank_ic_oo"] == 1.0
    assert result["oos_rank_ic_lag1"] == 1.0
    assert result["oos_top0p6_return_oo"] > result["oos_top0p6_return_lag1"]
    assert result["oos_top0p6_turnover_proxy"] == 0.0


def test_metric_transfer_summary_is_computed_within_each_window():
    frame = pd.DataFrame(
        {
            "window": ["w1"] * 3,
            "rawtopstable_h5_top0p6": [1.0, 2.0, 3.0],
            "rawtopret_h5_top0p6": [1.0, 2.0, 3.0],
            "alpha": [1.0, 2.0, 3.0],
            "oos_rank_ic_oo": [3.0, 2.0, 1.0],
            "oos_rank_ic_lag1": [3.0, 2.0, 1.0],
            "oos_top0p6_return_oo": [3.0, 2.0, 1.0],
            "oos_top0p6_return_lag1": [3.0, 2.0, 1.0],
            "oos_top0p6_turnover_proxy": [1.0, 2.0, 3.0],
        }
    )

    result = metric_transfer_summary(frame)
    row = result[
        (result["validation_metric"] == "rawtopstable_h5_top0p6")
        & (result["oos_metric"] == "oos_rank_ic_oo")
    ].iloc[0]
    assert row["spearman_across_epochs"] == -1.0
