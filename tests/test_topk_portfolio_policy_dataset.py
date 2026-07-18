from types import SimpleNamespace

import pandas as pd

from run.build_topk_portfolio_policy_dataset import build_dataset


def _args():
    return SimpleNamespace(
        split_name="test",
        max_replace=2,
        candidate_end=10,
        round_trip_cost=0.0,
        downside_cost=0.5,
        delayed_cost=0.25,
        blocked_buy_cost=0.1,
    )


def test_topk_dataset_builds_proposal_rows_without_future_features():
    frame = pd.DataFrame(
        [
            {
                "date": "2025-01-02",
                "code": "A",
                "candidate_position": 0,
                "candidate_rank_pct": 0.0,
                "target_n": 2,
                "is_kept": 1,
                "baseline_fill": 0,
                "protected_fill": 0,
                "eligible": 0,
                "label_available": 1,
                "exec_return_1d": 0.01,
                "exec_return_3d": 0.01,
                "exec_return_5d": 0.01,
                "exec_return_10d": 0.01,
                "exec_delayed_return": 0.005,
                "exec_max_downside": 0.01,
            },
            {
                "date": "2025-01-02",
                "code": "B",
                "candidate_position": 1,
                "candidate_rank_pct": 0.1,
                "target_n": 2,
                "is_kept": 0,
                "baseline_fill": 1,
                "protected_fill": 0,
                "eligible": 1,
                "label_available": 1,
                "exec_return_1d": -0.02,
                "exec_return_3d": -0.02,
                "exec_return_5d": -0.02,
                "exec_return_10d": -0.02,
                "exec_delayed_return": -0.02,
                "exec_max_downside": 0.03,
            },
            {
                "date": "2025-01-02",
                "code": "C",
                "candidate_position": 2,
                "candidate_rank_pct": 0.2,
                "target_n": 2,
                "is_kept": 0,
                "baseline_fill": 0,
                "protected_fill": 0,
                "eligible": 1,
                "label_available": 1,
                "exec_return_1d": 0.03,
                "exec_return_3d": 0.03,
                "exec_return_5d": 0.03,
                "exec_return_10d": 0.03,
                "exec_delayed_return": 0.02,
                "exec_max_downside": 0.01,
            },
        ]
    )
    frame["date"] = pd.to_datetime(frame["date"])

    out = build_dataset(frame, _args())

    assert set(out["proposal"]) >= {"baseline", "raw_top"}
    assert "selected_codes" in out.columns
    assert out.loc[out["proposal"].eq("baseline"), "utility_delta_vs_baseline"].iloc[0] == 0
    assert "exec_return_1d" not in out.columns
