from types import SimpleNamespace

import pandas as pd

from run.apply_pairwise_replacement_policy_lgbm import apply_concentration_penalty


def test_apply_concentration_penalty_only_penalizes_positive_deltas():
    frame = pd.DataFrame(
        {
            "policy_score": [0.0010, 0.0010, 0.0010],
            "diff_top_industry_hhi": [0.02, -0.05, 0.0],
            "diff_top_industry_share": [0.10, 0.20, -0.30],
            "diff_candidate_industry_top_share": [0.30, -0.40, 0.50],
        }
    )
    args = SimpleNamespace(
        industry_hhi_penalty=0.01,
        top_industry_share_penalty=0.001,
        candidate_industry_share_penalty=0.002,
    )

    out = apply_concentration_penalty(frame, args)

    assert out["raw_policy_score"].tolist() == [0.0010, 0.0010, 0.0010]
    assert out["concentration_penalty"].round(6).tolist() == [0.0009, 0.0002, 0.0010]
    assert out["policy_score"].round(6).tolist() == [0.0001, 0.0008, 0.0]


def test_apply_concentration_penalty_can_be_state_conditioned():
    frame = pd.DataFrame(
        {
            "policy_score": [0.0010, 0.0010, 0.0010],
            "diff_candidate_industry_top_share": [0.50, 0.50, 0.50],
            "diag_portfolio_beta_60d": [1.30, 0.90, 0.80],
            "diag_portfolio_specific_vol_60d": [0.05, 0.09, 0.05],
            "pair_risk_delta": [-0.10, -0.10, 0.20],
        }
    )
    args = SimpleNamespace(
        industry_hhi_penalty=0.0,
        top_industry_share_penalty=0.0,
        candidate_industry_share_penalty=0.002,
        concentration_penalty_condition="fragile_beta_or_vol",
        penalty_beta_threshold=1.2,
        penalty_specific_vol_threshold=0.08,
        penalty_pair_risk_threshold=0.0,
    )

    out = apply_concentration_penalty(frame, args)

    assert out["concentration_penalty_active"].tolist() == [1, 1, 0]
    assert out["concentration_penalty"].round(6).tolist() == [0.001, 0.001, 0.0]
    assert out["policy_score"].round(6).tolist() == [0.0, 0.0, 0.001]


def test_apply_risk_guard_penalizes_vol_and_negative_ret20_under_condition():
    frame = pd.DataFrame(
        {
            "policy_score": [0.0010, 0.0010, 0.0010],
            "diff_candidate_industry_top_share": [-0.10, 0.10, -0.10],
            "diff_specific_vol_60d": [0.20, 0.20, -0.30],
            "diff_ret_20d": [-0.10, -0.10, -0.10],
            "pair_risk_delta": [-0.10, -0.10, -0.10],
        }
    )
    args = SimpleNamespace(
        industry_hhi_penalty=0.0,
        top_industry_share_penalty=0.0,
        candidate_industry_share_penalty=0.0,
        concentration_penalty_condition="always",
        penalty_beta_threshold=1.2,
        penalty_specific_vol_threshold=0.08,
        penalty_pair_risk_threshold=0.0,
        specific_vol_worsen_penalty=0.001,
        ret20_worsen_penalty=0.002,
        pair_risk_worsen_penalty=0.0,
        beta_worsen_penalty=0.0,
        risk_guard_condition="when_decrowding",
    )

    out = apply_concentration_penalty(frame, args)

    assert out["risk_guard_active"].tolist() == [1, 0, 1]
    assert out["risk_guard_penalty"].round(6).tolist() == [0.0004, 0.0, 0.0002]
    assert out["policy_score"].round(6).tolist() == [0.0006, 0.0010, 0.0008]


def test_state_conditioned_risk_guard_uses_active_or_global_pressure():
    frame = pd.DataFrame(
        {
            "policy_score": [0.0010, 0.0010, 0.0010],
            "diff_candidate_industry_top_share": [0.10, 0.10, 0.10],
            "diff_specific_vol_60d": [0.20, 0.20, 0.20],
            "diff_ret_20d": [-0.10, -0.10, -0.10],
            "pair_risk_delta": [-0.10, -0.10, -0.10],
            "diag_active_drawdown_trailing_return": [-0.02, 0.03, 0.03],
            "diag_global_risk_pressure": [0.01, 0.06, 0.01],
        }
    )
    args = SimpleNamespace(
        industry_hhi_penalty=0.0,
        top_industry_share_penalty=0.0,
        candidate_industry_share_penalty=0.0,
        concentration_penalty_condition="always",
        penalty_beta_threshold=1.2,
        penalty_specific_vol_threshold=0.08,
        penalty_pair_risk_threshold=0.0,
        penalty_active_drawdown_threshold=0.0,
        penalty_global_pressure_threshold=0.04,
        specific_vol_worsen_penalty=0.001,
        ret20_worsen_penalty=0.002,
        pair_risk_worsen_penalty=0.0,
        beta_worsen_penalty=0.0,
        risk_guard_condition="active_or_global",
    )

    out = apply_concentration_penalty(frame, args)

    assert out["risk_guard_active"].tolist() == [1, 1, 0]
    assert out["risk_guard_penalty"].round(6).tolist() == [0.0004, 0.0004, 0.0]
    assert out["policy_score"].round(6).tolist() == [0.0006, 0.0006, 0.0010]


def test_risk_guard_penalizes_pair_risk_and_beta_worsening():
    frame = pd.DataFrame(
        {
            "policy_score": [1.0, 1.0],
            "pair_risk_delta": [0.20, -0.10],
            "diff_beta_60d": [0.30, 0.40],
        }
    )
    args = SimpleNamespace(
        industry_hhi_penalty=0.0,
        top_industry_share_penalty=0.0,
        candidate_industry_share_penalty=0.0,
        concentration_penalty_condition="always",
        penalty_beta_threshold=1.2,
        penalty_specific_vol_threshold=0.08,
        penalty_pair_risk_threshold=0.0,
        specific_vol_worsen_penalty=0.0,
        ret20_worsen_penalty=0.0,
        pair_risk_worsen_penalty=0.5,
        beta_worsen_penalty=0.25,
        risk_guard_condition="always",
    )

    out = apply_concentration_penalty(frame, args)

    assert out["risk_guard_penalty"].round(6).tolist() == [0.175, 0.1]
    assert out["policy_score"].round(6).tolist() == [0.825, 0.9]


def test_risk_guard_can_be_conditioned_on_risk_combo():
    frame = pd.DataFrame(
        {
            "policy_score": [1.0, 1.0, 1.0],
            "pair_risk_delta": [0.20, 0.20, -0.10],
            "diff_specific_vol_60d": [0.05, -0.01, 0.05],
            "diff_ret_20d": [-0.03, 0.02, 0.02],
            "diff_candidate_industry_top_share": [0.10, 0.10, -0.10],
            "diff_beta_60d": [0.10, -0.10, -0.10],
        }
    )
    args = SimpleNamespace(
        industry_hhi_penalty=0.0,
        top_industry_share_penalty=0.0,
        candidate_industry_share_penalty=0.0,
        concentration_penalty_condition="always",
        penalty_beta_threshold=1.2,
        penalty_specific_vol_threshold=0.08,
        penalty_pair_risk_threshold=0.0,
        specific_vol_worsen_penalty=1.0,
        ret20_worsen_penalty=1.0,
        pair_risk_worsen_penalty=1.0,
        beta_worsen_penalty=1.0,
        risk_guard_condition="risk_combo_ge",
        risk_combo_min_count=3,
    )

    out = apply_concentration_penalty(frame, args)

    assert out["risk_guard_active"].tolist() == [1, 0, 0]
    assert out["risk_guard_penalty"].round(6).tolist() == [0.38, 0.0, 0.0]
    assert out["policy_score"].round(6).tolist() == [0.62, 1.0, 1.0]


def test_replacement_gate_rejects_only_when_gate_condition_matches():
    frame = pd.DataFrame(
        {
            "policy_score": [0.0010, 0.0010, 0.0010],
            "pair_risk_delta": [0.20, 0.20, -0.10],
            "pair_downside_delta": [0.05, -0.02, 0.05],
            "diff_specific_vol_60d": [0.06, 0.06, 0.06],
            "diff_ret_20d": [-0.10, -0.10, -0.10],
            "diag_active_drawdown_trailing_return": [-0.02, 0.03, -0.02],
            "diag_global_risk_pressure": [0.01, 0.01, 0.01],
        }
    )
    args = SimpleNamespace(
        industry_hhi_penalty=0.0,
        top_industry_share_penalty=0.0,
        candidate_industry_share_penalty=0.0,
        concentration_penalty_condition="always",
        penalty_beta_threshold=1.2,
        penalty_specific_vol_threshold=0.08,
        penalty_pair_risk_threshold=0.0,
        penalty_active_drawdown_threshold=0.0,
        penalty_global_pressure_threshold=0.04,
        specific_vol_worsen_penalty=0.0,
        ret20_worsen_penalty=0.0,
        pair_risk_worsen_penalty=0.0,
        beta_worsen_penalty=0.0,
        risk_guard_condition="always",
        gate_condition="active_drawdown_negative",
        gate_max_pair_risk_delta=0.0,
        gate_max_pair_downside_delta=0.0,
        gate_max_diff_specific_vol_60d=None,
        gate_min_diff_ret20=None,
    )

    out = apply_concentration_penalty(frame, args)

    assert out["replacement_gate_active"].tolist() == [1, 0, 1]
    assert out["replacement_gate_rejected"].tolist() == [1, 0, 1]
    assert out["replacement_gate_reason"].tolist() == [
        "pair_risk_delta;pair_downside_delta",
        "",
        "pair_downside_delta",
    ]
    assert out["policy_score"].replace(-float("inf"), -999).tolist() == [-999, 0.001, -999]


def test_replacement_gate_missing_optional_column_does_not_reject_or_crash():
    frame = pd.DataFrame(
        {
            "policy_score": [0.0010],
            "pair_risk_delta": [-0.10],
        }
    )
    args = SimpleNamespace(
        industry_hhi_penalty=0.0,
        top_industry_share_penalty=0.0,
        candidate_industry_share_penalty=0.0,
        concentration_penalty_condition="always",
        penalty_beta_threshold=1.2,
        penalty_specific_vol_threshold=0.08,
        penalty_pair_risk_threshold=0.0,
        penalty_active_drawdown_threshold=0.0,
        penalty_global_pressure_threshold=0.04,
        specific_vol_worsen_penalty=0.0,
        ret20_worsen_penalty=0.0,
        pair_risk_worsen_penalty=0.0,
        beta_worsen_penalty=0.0,
        risk_guard_condition="always",
        gate_condition="always",
        gate_max_pair_risk_delta=0.0,
        gate_max_pair_downside_delta=0.0,
        gate_max_diff_specific_vol_60d=None,
        gate_min_diff_ret20=None,
    )

    out = apply_concentration_penalty(frame, args)

    assert out["replacement_gate_rejected"].tolist() == [0]
    assert out["policy_score"].tolist() == [0.001]
