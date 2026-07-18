from types import SimpleNamespace

import numpy as np
import pandas as pd

from run.train_pairwise_replacement_policy_lgbm import (
    feature_columns,
    group_sizes_by_date,
    rank_relevance_labels,
    target_values,
)


def test_target_values_supports_regression_and_binary():
    frame = pd.DataFrame({"ledger_path_utility": [-0.1, 0.0, 0.2]})

    reg = target_values(
        frame,
        SimpleNamespace(
            target_col="ledger_path_utility",
            objective="regression",
            positive_threshold=0.0,
        ),
    )
    binary = target_values(
        frame,
        SimpleNamespace(
            target_col="ledger_path_utility",
            objective="binary",
            positive_threshold=0.0,
        ),
    )

    assert np.allclose(reg, [-0.1, 0.0, 0.2])
    assert binary.tolist() == [0.0, 0.0, 1.0]


def test_feature_columns_excludes_future_label_columns():
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "code": ["000001.SZ"],
            "ledger_path_utility": [0.1],
            "ledger_risk_adjusted_utility": [0.1],
            "ledger_beta_penalty": [0.01],
            "pair_path_raw_edge": [0.2],
            "pair_downside_delta": [0.3],
            "pair_risk_delta": [0.4],
            "cand_ret_20d": [0.5],
        }
    )

    cols = feature_columns(frame)

    assert "ledger_path_utility" not in cols
    assert "ledger_risk_adjusted_utility" not in cols
    assert "ledger_beta_penalty" not in cols
    assert "pair_path_raw_edge" not in cols
    assert "pair_downside_delta" not in cols
    assert "pair_risk_delta" in cols
    assert "cand_ret_20d" in cols


def test_rank_relevance_labels_are_date_local():
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
            "ledger_path_utility": [-0.1, 0.0, 0.2, 10.0, 11.0],
        }
    )

    labels = rank_relevance_labels(
        frame,
        SimpleNamespace(target_col="ledger_path_utility", rank_label_bins=5),
    )

    assert labels[:3].tolist() == [1, 3, 4]
    assert labels[3:].tolist() == [2, 4]


def test_group_sizes_by_date_preserves_date_groups():
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
            "ledger_path_utility": [0.1, 0.2, 0.3],
        }
    )

    assert group_sizes_by_date(frame) == [2, 1]
