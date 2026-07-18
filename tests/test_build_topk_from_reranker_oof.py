from types import SimpleNamespace

import pandas as pd

from run.build_topk_from_reranker_oof import build_day


def test_build_day_creates_oof_topk_proposals_without_future_feature_columns():
    frame = pd.DataFrame(
        [
            {
                "split": "oof",
                "date": "2020-01-02",
                "code": f"C{i}",
                "group_size": 500,
                "candidate_position": i,
                "industry_id": i % 3,
                "future_target": float(i % 5),
                "m0_alpha": 10.0 - i,
                "m0_rank_pct": i / 10.0,
                "m0_alpha_ma3": 9.0 - i,
                "risk_000": float(i),
            }
            for i in range(10)
        ]
    )
    frame["date"] = pd.to_datetime(frame["date"])
    args = SimpleNamespace(
        max_candidates_per_day=10,
        target_frac=0.006,
        min_target_n=3,
        target_col="future_target",
    )

    rows = build_day(frame, args)
    out = pd.DataFrame(rows)

    assert set(out["proposal"]) >= {"baseline", "low_risk", "industry_diverse"}
    assert out["date"].nunique() == 1
    assert "future_target" not in out.columns
    assert "utility_delta_vs_baseline" in out.columns
