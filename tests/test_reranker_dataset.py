import numpy as np
import pandas as pd

from run.build_reranker_dataset import (
    percentile_rank_desc,
    relevance_from_rank,
)
from run.build_reranker_v3_dataset import add_state_features
from run.align_reranker_dataset_to_execution import align_candidate_positions


def test_percentile_rank_desc_orders_largest_first():
    values = np.asarray([2.0, 5.0, 1.0, 3.0])
    ranks = percentile_rank_desc(values)
    assert ranks[1] == 0.0
    assert ranks[2] == 1.0


def test_relevance_cutoffs_are_graded():
    rank_pct = np.asarray([0.0, 0.10, 0.20, 0.40, 0.80])
    relevance = relevance_from_rank(rank_pct)
    assert relevance.tolist() == [4, 3, 2, 1, 0]


def test_state_features_use_explicit_target_and_hold_fractions():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    codes = [f"S{i:02d}" for i in range(20)]
    rows = []
    for date, ordered in ((dates[0], codes), (dates[1], list(reversed(codes)))):
        rows.extend(
            {
                "date": date,
                "code": code,
                "candidate_position": position,
            }
            for position, code in enumerate(ordered)
        )
    frame = pd.DataFrame(rows)
    audit = pd.DataFrame({"date": dates, "universe_size": [100, 100]})

    narrow = add_state_features(frame, audit, target_frac=0.10, hold_frac=0.10)
    wide = add_state_features(frame, audit, target_frac=0.10, hold_frac=0.20)

    narrow_day2 = narrow[narrow["date"].eq(dates[1])]["v3_vacancies"].iloc[0]
    wide_day2 = wide[wide["date"].eq(dates[1])]["v3_vacancies"].iloc[0]
    assert narrow_day2 == 10
    assert wide_day2 == 0


def test_candidate_alignment_demotes_chase_names_before_state_building():
    date = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame(
        {
            "date": [date] * 3,
            "code": ["A", "B", "C"],
            "candidate_position": [0, 1, 2],
            "group_size": [3, 3, 3],
        }
    )
    aligned = align_candidate_positions(
        frame,
        {date: {"A": 0.10, "B": 0.01}},
        max_signal_return=0.095,
    )

    assert aligned.sort_values("candidate_position")["code"].tolist() == ["B", "C", "A"]
    assert aligned.set_index("code").loc["A", "execution_demoted"] == 1
