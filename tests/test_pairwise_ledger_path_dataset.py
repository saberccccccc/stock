from types import SimpleNamespace

import pandas as pd

from run.build_pairwise_ledger_path_dataset import build_dataset


def _policy_frame():
    rows = [
        {
            "date": "2025-01-02",
            "code": "BASE",
            "eligible": 1,
            "label_available": 1,
            "candidate_position": 1,
            "baseline_fill": 1,
            "exec_return_1d": 0.010,
            "exec_return_3d": 0.020,
            "exec_return_5d": 0.030,
            "exec_return_10d": 0.040,
            "exec_max_downside": -0.020,
            "beta_60d": 1.00,
            "specific_vol_60d": 0.10,
            "candidate_industry_top_share": 0.20,
            "top_industry_share": 0.25,
            "top_industry_hhi": 0.10,
            "ret_5d": 0.010,
            "ret_20d": 0.050,
            "drawdown_20d": 0.010,
        },
        {
            "date": "2025-01-02",
            "code": "CAND",
            "eligible": 1,
            "label_available": 1,
            "candidate_position": 2,
            "baseline_fill": 0,
            "exec_return_1d": 0.020,
            "exec_return_3d": 0.030,
            "exec_return_5d": 0.040,
            "exec_return_10d": 0.050,
            "exec_max_downside": -0.010,
            "beta_60d": 1.20,
            "specific_vol_60d": 0.15,
            "candidate_industry_top_share": 0.30,
            "top_industry_share": 0.35,
            "top_industry_hhi": 0.12,
            "ret_5d": -0.010,
            "ret_20d": 0.200,
            "drawdown_20d": 0.050,
        },
    ]
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _diag_frame():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-03"]),
            "holdings_map": [{"BASE": 0.10}],
            "active_drawdown_trailing_return": [-0.05],
            "global_risk_pressure": [0.02],
        }
    )


def _args(**overrides):
    values = {
        "split_name": "test_2025",
        "max_pairs_per_day": 10,
        "horizon_weights": "0.25,0.25,0.25,0.25",
        "base_cost_bps": 0.0,
        "risk_cost": 0.0,
        "downside_cost": 0.0,
        "quick_fade_cost": 0.0,
        "beta_cost": 0.0,
        "specific_vol_cost": 0.0,
        "industry_concentration_cost": 0.0,
        "active_drawdown_cost": 0.0,
        "lag1_decay_cost": 0.0,
        "momentum_plateau_cost": 0.0,
        "min_baseline_weight": 0.001,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_risk_adjusted_utility_defaults_to_legacy_utility():
    out = build_dataset(_policy_frame(), _diag_frame(), _args())

    assert not out.empty
    assert out["ledger_path_utility"].round(8).tolist() == out[
        "ledger_risk_adjusted_utility"
    ].round(8).tolist()


def test_risk_adjusted_utility_exposes_penalty_components():
    out = build_dataset(
        _policy_frame(),
        _diag_frame(),
        _args(
            beta_cost=0.10,
            specific_vol_cost=0.20,
            industry_concentration_cost=0.30,
            active_drawdown_cost=0.40,
            momentum_plateau_cost=0.50,
        ),
    )
    cand = out[out["code"].eq("CAND")].iloc[0]

    assert cand["ledger_beta_penalty"] > 0
    assert cand["ledger_specific_vol_penalty"] > 0
    assert cand["ledger_industry_concentration_penalty"] > 0
    assert cand["ledger_active_drawdown_penalty"] > 0
    assert cand["ledger_momentum_plateau_penalty"] > 0
    assert cand["ledger_risk_adjusted_utility"] < cand["ledger_path_utility"]
