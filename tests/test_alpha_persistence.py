import numpy as np
import pandas as pd

from alpha.persistence import rank_alpha_rows, rolling_average_alpha_scores
from backtest.predictors import PersistentPredictor
from run import backtest_v9_retention


class SequencePredictor:
    def predict_alpha(self, sample, valid, regime):
        return np.asarray(sample["scores"], dtype=np.float32)


def score_rows_and_samples():
    rows = [
        {"date": pd.Timestamp("2024-01-02"), "codes": ["B", "A"], "alpha": [2.0, 1.0], "n_stocks": 2},
        {"date": pd.Timestamp("2024-01-03"), "codes": ["A", "B", "C"], "alpha": [3.0, 5.0, 4.0], "n_stocks": 3},
        {"date": pd.Timestamp("2024-01-04"), "codes": ["C", "B", "A"], "alpha": [6.0, 8.0, 7.0], "n_stocks": 3},
    ]
    samples = [{"codes": row["codes"], "scores": row["alpha"]} for row in rows]
    return rows, samples


def test_rolling_average_matches_persistent_predictor_average_mode():
    rows, samples = score_rows_and_samples()
    predictor = PersistentPredictor(SequencePredictor(), window=2, mode="average")
    expected = []
    for sample in samples:
        alpha = predictor.predict_alpha(
            sample,
            np.ones(len(sample["codes"]), dtype=bool),
            regime=None,
        )
        expected.append(dict(zip(sample["codes"], alpha)))

    actual = rolling_average_alpha_scores(rows, window=2)

    for row, expected_by_code in zip(actual, expected):
        actual_by_code = dict(zip(row["codes"], row["alpha"]))
        assert actual_by_code.keys() == expected_by_code.keys()
        for code, value in actual_by_code.items():
            np.testing.assert_allclose(value, expected_by_code[code], rtol=0, atol=1e-7)


def test_rank_alpha_rows_preserves_expected_descending_order():
    ranked = rank_alpha_rows(
        [{"date": "2024-01-02", "codes": ["A", "B", "C"], "alpha": [0.1, 0.9, 0.5]}]
    )

    assert ranked[0]["codes"] == ["B", "C", "A"]
    np.testing.assert_allclose(ranked[0]["alpha"], [0.9, 0.5, 0.1], rtol=0, atol=1e-7)


def test_v9_ranked_rows_remain_a_ranking_of_single_pass_scores(monkeypatch):
    codes = [f"S{i}" for i in range(10)]
    samples = [{"date": "2024-01-02", "codes": codes, "X": np.zeros((10, 1))}]
    predictor = SequencePredictor()
    samples[0]["scores"] = list(range(10))
    monkeypatch.setattr(backtest_v9_retention, "detect_regime", lambda sample: None)

    score_rows = backtest_v9_retention.compute_v9_alpha_scores(samples, predictor)
    ranked_rows = backtest_v9_retention.compute_v9_alpha_rows(samples, predictor)

    assert ranked_rows == rank_alpha_rows(score_rows)
