from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.execution import apply_open_ledger_constraints, open_limit_trade_mask
from backtest.open_ledger import (
    _load_execution_mask_cache,
    _save_execution_mask_cache,
    active_drawdown_condition_state,
    active_drawdown_observation,
    active_drawdown_throttle_continuous_state,
    active_drawdown_throttle_step_state,
    active_drawdown_throttle_state,
    apply_industry_selection_cap,
    apply_state_aware_selection_rank,
    build_execution_constraint_masks,
    compute_market_multiplier,
    encode_holdings,
    estimate_trailing_stock_risk,
    global_risk_overlay_state,
    infer_ohlc_load_window,
    load_index_returns,
    load_industry_map,
    load_ohlc_money,
    normalize_ts_code,
    parse_active_drawdown_throttle_steps,
    prepare_open_ledger_context,
    recompute_adv,
    run_open_ledger,
    save_stage_breakdown,
    should_apply_defensive_tilt,
    summarize_open_ledger_result,
    summarize_portfolio_risk,
)


def test_open_ledger_keeps_execution_import_compatibility():
    from backtest import open_ledger

    assert open_ledger.apply_open_ledger_constraints is apply_open_ledger_constraints
    assert open_ledger.open_limit_trade_mask is open_limit_trade_mask


def test_execution_mask_cache_roundtrip(tmp_path):
    index = pd.to_datetime(["2025-01-02", "2025-01-03"])
    columns = ["000001.SZ", "600000.SH"]
    masks = {
        name: pd.DataFrame([[False, True], [True, False]], index=index, columns=columns)
        for name in (
            "buy_block",
            "sell_block",
            "no_trade",
            "limit_up_open",
            "limit_down_open",
            "limit_up_touch",
            "limit_down_touch",
            "new_stock_buy_block",
        )
    }
    path = tmp_path / "masks.npz"

    _save_execution_mask_cache(path, masks)
    loaded = _load_execution_mask_cache(path, index, columns)

    assert loaded is not None
    for name, expected in masks.items():
        pd.testing.assert_frame_equal(loaded[name], expected)


def test_execution_mask_cache_rejects_wrong_shape(tmp_path):
    index = pd.to_datetime(["2025-01-02", "2025-01-03"])
    columns = ["000001.SZ"]
    masks = {
        name: pd.DataFrame([[False], [True]], index=index, columns=columns)
        for name in (
            "buy_block",
            "sell_block",
            "no_trade",
            "limit_up_open",
            "limit_down_open",
            "limit_up_touch",
            "limit_down_touch",
            "new_stock_buy_block",
        )
    }
    path = tmp_path / "masks.npz"

    _save_execution_mask_cache(path, masks)

    assert _load_execution_mask_cache(path, index[:1], columns) is None


def test_defensive_tilt_market_multiplier_gate():
    assert should_apply_defensive_tilt(0.15, market_mult=1.0, market_mult_below=1.0)
    assert not should_apply_defensive_tilt(0.15, market_mult=1.0, market_mult_below=0.8)
    assert should_apply_defensive_tilt(0.15, market_mult=0.7, market_mult_below=0.8)
    assert not should_apply_defensive_tilt(0.0, market_mult=0.7, market_mult_below=0.8)


def test_active_drawdown_throttle_state_uses_realized_active_returns():
    scale, trailing, remaining = active_drawdown_throttle_state(
        [-0.02, -0.02],
        lookback_days=2,
        trigger_return=-0.03,
        scale=0.9,
        cooldown_days=5,
        remaining_days=0,
    )

    assert scale == pytest.approx(0.9)
    assert trailing == pytest.approx((0.98 * 0.98) - 1.0)
    assert remaining == 5

    scale, trailing, remaining = active_drawdown_throttle_state(
        [0.01],
        lookback_days=2,
        trigger_return=-0.03,
        scale=0.9,
        cooldown_days=5,
        remaining_days=3,
    )

    assert scale == pytest.approx(0.9)
    assert np.isnan(trailing)
    assert remaining == 3


def test_active_drawdown_observation_does_not_require_throttle():
    trailing = active_drawdown_observation([0.01, -0.02, 0.03], lookback=2)

    assert trailing == pytest.approx((0.98 * 1.03) - 1.0)
    assert np.isnan(active_drawdown_observation([0.01], lookback=2))

    scale, _, remaining = active_drawdown_throttle_state(
        [-0.10],
        lookback_days=1,
        trigger_return=-0.03,
        scale=1.0,
        cooldown_days=5,
        remaining_days=0,
    )

    assert scale == pytest.approx(1.0)
    assert remaining == 0


def test_active_drawdown_throttle_state_respects_condition_gate():
    scale, trailing, remaining = active_drawdown_throttle_state(
        [-0.02, -0.02],
        lookback_days=2,
        trigger_return=-0.03,
        scale=0.9,
        cooldown_days=5,
        remaining_days=0,
        trigger_allowed=False,
    )

    assert scale == pytest.approx(1.0)
    assert trailing == pytest.approx((0.98 * 0.98) - 1.0)
    assert remaining == 0


def test_active_drawdown_throttle_continuous_scales_by_severity():
    mild_scale, mild_trailing, remaining, held = active_drawdown_throttle_continuous_state(
        [-0.015, -0.015],
        lookback_days=2,
        trigger_return=-0.02,
        min_scale=0.7,
        width=0.06,
        cooldown_days=0,
        remaining_days=0,
    )
    deep_scale, deep_trailing, _, _ = active_drawdown_throttle_continuous_state(
        [-0.05, -0.05],
        lookback_days=2,
        trigger_return=-0.02,
        min_scale=0.7,
        width=0.06,
        cooldown_days=0,
        remaining_days=0,
    )

    assert mild_trailing < -0.02
    assert deep_trailing < mild_trailing
    assert 0.7 < mild_scale < 1.0
    assert deep_scale == pytest.approx(0.7)
    assert remaining == 0
    assert held == pytest.approx(1.0)


def test_active_drawdown_throttle_steps_choose_more_defensive_scale():
    steps = parse_active_drawdown_throttle_steps("-0.03:0.85,-0.06:0.65")

    scale, trailing, remaining, held_scale, label = active_drawdown_throttle_step_state(
        active_returns=[-0.02, -0.02, -0.03],
        lookback_days=3,
        steps=steps,
        cooldown_days=5,
        remaining_days=0,
        remaining_scale=1.0,
    )

    assert trailing < -0.06
    assert scale == pytest.approx(0.65)
    assert held_scale == pytest.approx(0.65)
    assert remaining == 5
    assert label == "-0.06:0.65"


def test_active_drawdown_throttle_steps_hold_cooldown_scale():
    steps = parse_active_drawdown_throttle_steps("-0.03:0.85,-0.06:0.65")

    scale, trailing, remaining, held_scale, label = active_drawdown_throttle_step_state(
        active_returns=[0.01, 0.01, 0.01],
        lookback_days=3,
        steps=steps,
        cooldown_days=5,
        remaining_days=3,
        remaining_scale=0.65,
    )

    assert trailing > 0.0
    assert scale == pytest.approx(0.65)
    assert held_scale == pytest.approx(0.65)
    assert remaining == 3
    assert label == ""

    scale, _, remaining = active_drawdown_throttle_state(
        [0.01],
        lookback_days=2,
        trigger_return=-0.03,
        scale=0.9,
        cooldown_days=5,
        remaining_days=3,
        trigger_allowed=False,
    )

    assert scale == pytest.approx(0.9)
    assert remaining == 3


def test_active_drawdown_condition_state_detects_crowding_momentum():
    dates = pd.date_range("2025-01-01", periods=30)
    close_df = pd.DataFrame(
        {
            "A": np.linspace(12.0, 10.0, 30),
            "B": np.linspace(11.0, 10.0, 30),
            "C": np.linspace(10.0, 10.0, 30),
        },
        index=dates,
    )
    close_ret = close_df.pct_change().fillna(0.0).to_numpy()
    args = SimpleNamespace(
        active_drawdown_throttle_condition="crowding_momentum",
        active_drawdown_throttle_min_top_industry_weight=0.50,
        active_drawdown_throttle_min_industry_hhi=0.35,
        active_drawdown_throttle_max_momentum20=0.0,
        active_drawdown_throttle_min_volatility60=0.0,
    )

    allowed, diag = active_drawdown_condition_state(
        args,
        ["A", "B", "C"],
        np.array([0.3, 0.3, 0.1]),
        {"A": "Tech", "B": "Tech", "C": "Bank"},
        close_df,
        close_ret,
        day=25,
    )

    assert allowed
    assert diag["active_drawdown_condition_top_industry"] == "Tech"
    assert diag["active_drawdown_condition_top_industry_weight"] == pytest.approx(0.6)
    assert diag["active_drawdown_condition_industry_hhi"] > 0.7
    assert diag["active_drawdown_condition_momentum20"] < 0.0


def test_active_drawdown_condition_state_accepts_stock_by_date_returns():
    dates = pd.date_range("2025-01-01", periods=30)
    close_df = pd.DataFrame(
        {
            "A": np.linspace(12.0, 10.0, 30),
            "B": np.linspace(11.0, 10.0, 30),
            "C": np.linspace(10.0, 10.0, 30),
        },
        index=dates,
    )
    close_ret = close_df.pct_change().fillna(0.0).to_numpy().T
    args = SimpleNamespace(
        active_drawdown_throttle_condition="crowding_momentum_volatility",
        active_drawdown_throttle_min_top_industry_weight=0.50,
        active_drawdown_throttle_min_industry_hhi=0.35,
        active_drawdown_throttle_max_momentum20=0.0,
        active_drawdown_throttle_min_volatility60=0.0,
    )

    allowed, diag = active_drawdown_condition_state(
        args,
        ["A", "B", "C"],
        np.array([0.3, 0.3, 0.1]),
        {"A": "Tech", "B": "Tech", "C": "Bank"},
        close_df,
        close_ret,
        day=25,
    )

    assert allowed
    assert diag["active_drawdown_condition_volatility60"] >= 0.0


def test_global_risk_overlay_scales_triggered_days():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    features = pd.DataFrame(
        {"global_defensive_pressure": [0.02, 0.08]},
        index=dates,
    )
    args = SimpleNamespace(
        global_risk_overlay_mode="defensive_pressure",
        global_risk_pressure_col="global_defensive_pressure",
        global_risk_pressure_threshold=0.05,
        global_risk_market_scale=0.7,
        global_risk_target_frac=0.004,
    )

    calm = global_risk_overlay_state(args, features, "2025-01-02")
    stressed = global_risk_overlay_state(args, features, "2025-01-03")

    assert calm["global_risk_triggered"] == 0
    assert calm["global_risk_market_scale"] == pytest.approx(1.0)
    assert stressed["global_risk_triggered"] == 1
    assert stressed["global_risk_pressure"] == pytest.approx(0.08)
    assert stressed["global_risk_market_scale"] == pytest.approx(0.7)
    assert stressed["global_risk_target_frac"] == pytest.approx(0.004)


def test_global_risk_overlay_none_records_pressure_without_scaling():
    dates = pd.to_datetime(["2025-01-02"])
    features = pd.DataFrame(
        {"global_defensive_pressure": [0.08]},
        index=dates,
    )
    args = SimpleNamespace(
        global_risk_overlay_mode="none",
        global_risk_pressure_col="global_defensive_pressure",
    )

    state = global_risk_overlay_state(args, features, "2025-01-02")

    assert state["global_risk_triggered"] == 0
    assert state["global_risk_pressure"] == pytest.approx(0.08)
    assert state["global_risk_market_scale"] == pytest.approx(1.0)
    assert np.isnan(state["global_risk_target_frac"])


def test_global_risk_overlay_continuous_scales_with_pressure():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    features = pd.DataFrame(
        {"global_defensive_pressure": [0.04, 0.06, 0.10]},
        index=dates,
    )
    args = SimpleNamespace(
        global_risk_overlay_mode="defensive_pressure_continuous",
        global_risk_pressure_col="global_defensive_pressure",
        global_risk_pressure_threshold=0.05,
        global_risk_pressure_width=0.05,
        global_risk_market_scale=0.75,
        global_risk_target_frac=None,
    )

    calm = global_risk_overlay_state(args, features, "2025-01-02")
    mild = global_risk_overlay_state(args, features, "2025-01-03")
    stressed = global_risk_overlay_state(args, features, "2025-01-06")

    assert calm["global_risk_triggered"] == 0
    assert calm["global_risk_market_scale"] == pytest.approx(1.0)
    assert mild["global_risk_triggered"] == 1
    assert 0.75 < mild["global_risk_market_scale"] < 1.0
    assert stressed["global_risk_market_scale"] == pytest.approx(0.75)


def make_args(**overrides):
    args = SimpleNamespace(
        lot_size=100,
        limit_threshold=0.095,
        min_adv_cny=3_000_000.0,
        adv_participation_cap=0.05,
        commission_rate=0.0001,
        stamp_tax_rate=0.0005,
        slippage_rate=0.0005,
        min_commission_cny=5.0,
        rebalance_band=0.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_frames(open_values, close_values=None, adv_values=None):
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    columns = ["A", "B"]
    open_df = pd.DataFrame(open_values, index=dates, columns=columns)
    close_df = pd.DataFrame(close_values or open_values, index=dates, columns=columns)
    adv_df = pd.DataFrame(
        adv_values
        or [
            [10_000_000.0, 10_000_000.0],
            [10_000_000.0, 10_000_000.0],
        ],
        index=dates,
        columns=columns,
    )
    return open_df, close_df, adv_df


def test_open_limit_trade_mask_blocks_large_open_gaps():
    open_df, close_df, _ = make_frames(
        [[10.0, 20.0], [11.0, 18.0]],
        close_values=[[10.0, 20.0], [11.0, 18.0]],
    )

    buy_block, sell_block = open_limit_trade_mask(open_df, close_df, 1, 0.095)

    assert buy_block.tolist() == [True, False]
    assert sell_block.tolist() == [False, True]


def test_apply_open_ledger_constraints_buys_board_lots_and_costs_cash():
    open_df, close_df, adv_df = make_frames([[10.0, 20.0], [10.0, 20.0]])
    args = make_args()

    shares, cash, executed, info = apply_open_ledger_constraints(
        desired_weights=np.array([0.5, 0.5]),
        current_shares=np.zeros(2),
        cash=500_000.0,
        equity=500_000.0,
        open_df=open_df,
        close_df=close_df,
        adv_df=adv_df,
        day_pos=1,
        args=args,
    )

    assert np.all(shares % 100 == 0)
    assert np.all(executed >= 0)
    assert cash >= 0.0
    assert info["executed_turnover"] > 0.0
    assert info["cost"] > 0.0


def test_execution_trace_is_a_side_channel_and_preserves_fills():
    open_df, close_df, adv_df = make_frames([[10.0, 20.0], [10.0, 20.0]])
    args = make_args()
    common = dict(
        desired_weights=np.array([0.5, 0.5]),
        current_shares=np.zeros(2),
        cash=500_000.0,
        equity=500_000.0,
        open_df=open_df,
        close_df=close_df,
        adv_df=adv_df,
        day_pos=1,
        args=args,
    )

    plain = apply_open_ledger_constraints(**common)
    traced = apply_open_ledger_constraints(**common, capture_trace=True)

    np.testing.assert_array_equal(traced[0], plain[0])
    assert traced[1] == pytest.approx(plain[1])
    np.testing.assert_array_equal(traced[2], plain[2])
    assert traced[3] == plain[3]
    assert len(traced[4]) == 2
    assert {row["status"] for row in traced[4]} == {"filled"}
    assert sum(row["total_cost_cny"] for row in traced[4]) > 0.0


def test_execution_trace_preserves_stock_level_rejection_reason():
    open_df, close_df, adv_df = make_frames(
        [[10.0, 20.0], [11.0, 20.0]],
        close_values=[[10.0, 20.0], [11.0, 20.0]],
    )

    result = apply_open_ledger_constraints(
        desired_weights=np.array([0.5, 0.0]),
        current_shares=np.zeros(2),
        cash=500_000.0,
        equity=500_000.0,
        open_df=open_df,
        close_df=close_df,
        adv_df=adv_df,
        day_pos=1,
        args=make_args(),
        capture_trace=True,
    )

    assert len(result[4]) == 1
    rejection = result[4][0]
    assert rejection["asset_index"] == 0
    assert rejection["side"] == "buy"
    assert rejection["status"] == "rejected"
    assert rejection["reason"] == "limit_up_open"
    assert rejection["target_shares"] == pytest.approx(22700.0)
    assert rejection["executed_shares"] == 0.0


def test_apply_open_ledger_constraints_blocks_limit_up_buy():
    open_df, close_df, adv_df = make_frames(
        [[10.0, 20.0], [11.0, 20.0]],
        close_values=[[10.0, 20.0], [11.0, 20.0]],
    )
    args = make_args()

    shares, _, executed, info = apply_open_ledger_constraints(
        desired_weights=np.array([0.5, 0.0]),
        current_shares=np.zeros(2),
        cash=500_000.0,
        equity=500_000.0,
        open_df=open_df,
        close_df=close_df,
        adv_df=adv_df,
        day_pos=1,
        args=args,
    )

    assert shares[0] == 0
    assert executed[0] == 0
    assert info["blocked_buy"] == 1


def test_apply_open_ledger_constraints_empty_masks_do_not_fall_back_to_proxy():
    open_df, close_df, adv_df = make_frames(
        [[10.0, 20.0], [11.0, 20.0]],
        close_values=[[10.0, 20.0], [11.0, 20.0]],
    )
    args = make_args()

    shares, _, executed, info = apply_open_ledger_constraints(
        desired_weights=np.array([0.5, 0.0]),
        current_shares=np.zeros(2),
        cash=500_000.0,
        equity=500_000.0,
        open_df=open_df,
        close_df=close_df,
        adv_df=adv_df,
        day_pos=1,
        args=args,
        execution_masks={},
    )

    assert shares[0] > 0
    assert executed[0] > 0
    assert info["blocked_buy"] == 0


def test_apply_open_ledger_constraints_blocks_limit_down_sell():
    open_df, close_df, adv_df = make_frames(
        [[10.0, 20.0], [9.0, 20.0]],
        close_values=[[10.0, 20.0], [9.0, 20.0]],
    )
    args = make_args()

    shares, _, executed, info = apply_open_ledger_constraints(
        desired_weights=np.array([0.0, 0.0]),
        current_shares=np.array([1000.0, 0.0]),
        cash=490_000.0,
        equity=499_000.0,
        open_df=open_df,
        close_df=close_df,
        adv_df=adv_df,
        day_pos=1,
        args=args,
    )

    assert shares[0] == 1000
    assert executed[0] == 0
    assert info["blocked_sell"] == 1


def test_apply_open_ledger_constraints_blocks_no_trade_mask():
    open_df, close_df, adv_df = make_frames([[10.0, 20.0], [10.0, 20.0]])
    args = make_args()
    no_trade = pd.DataFrame(
        [[False, False], [True, False]],
        index=open_df.index,
        columns=open_df.columns,
    )

    shares, _, executed, info = apply_open_ledger_constraints(
        desired_weights=np.array([0.5, 0.0]),
        current_shares=np.zeros(2),
        cash=500_000.0,
        equity=500_000.0,
        open_df=open_df,
        close_df=close_df,
        adv_df=adv_df,
        day_pos=1,
        args=args,
        execution_masks={"no_trade": no_trade},
    )

    assert shares[0] == 0
    assert executed[0] == 0
    assert info["no_trade_blocked"] == 1
    assert info["blocked_buy"] == 0


def test_apply_open_ledger_constraints_uses_intraday_limit_touch_mask():
    open_df, close_df, adv_df = make_frames([[10.0, 20.0], [10.2, 20.0]])
    args = make_args()
    touch = pd.DataFrame(
        [[False, False], [True, False]],
        index=open_df.index,
        columns=open_df.columns,
    )

    shares, _, executed, info = apply_open_ledger_constraints(
        desired_weights=np.array([0.5, 0.0]),
        current_shares=np.zeros(2),
        cash=500_000.0,
        equity=500_000.0,
        open_df=open_df,
        close_df=close_df,
        adv_df=adv_df,
        day_pos=1,
        args=args,
        execution_masks={"buy_block": touch, "limit_up_touch": touch},
    )

    assert shares[0] == 0
    assert executed[0] == 0
    assert info["blocked_buy"] == 1
    assert info["limit_up_touch_blocked"] == 1


def test_apply_open_ledger_constraints_skips_rebalance_within_band():
    open_df, close_df, adv_df = make_frames([[10.0, 20.0], [10.0, 20.0]])
    args = make_args(rebalance_band=0.20)

    shares, cash, executed, info = apply_open_ledger_constraints(
        desired_weights=np.array([0.5, 0.5]),
        current_shares=np.array([23_000.0, 13_000.0]),
        cash=10_000.0,
        equity=500_000.0,
        open_df=open_df,
        close_df=close_df,
        adv_df=adv_df,
        day_pos=1,
        args=args,
    )

    assert np.array_equal(executed, np.zeros(2))
    assert np.array_equal(shares, np.array([23_000.0, 13_000.0]))
    assert cash == 10_000.0
    assert info["band_skipped"] == 2


def test_summarize_open_ledger_result_aggregates_diagnostics():
    args = make_args(
        market_timing_mode="legacy",
        max_new_names=5,
        execution_lag=1,
        risk_target_frac=None,
        risk_target_market_mult_below=1.0,
        active_drawdown_throttle_lookback=15,
        active_drawdown_throttle_trigger=-0.03,
        active_drawdown_throttle_scale=0.9,
        active_drawdown_throttle_cooldown=5,
        exit_hold_frac=0.0,
        switch_gap_frac=0.0,
    )
    returns = np.array([0.01, -0.005, 0.002], dtype=float)
    diag_df = pd.DataFrame(
        [
            {
                "turnover": 0.10,
                "executed_turnover": 0.09,
                "unfilled_turnover": 0.01,
                "selected_n": 2,
                "gross_weight": 0.7,
                "market_mult": 0.7,
                "active_drawdown_throttle_scale": 1.0,
                "cost": 0.001,
                "commission": 0.0002,
                "stamp_tax": 0.0003,
                "slippage": 0.0005,
                "blocked_buy": 1,
                "blocked_sell": 0,
                "adv_blocked": 0,
                "missing_adv": 0,
                "no_open": 0,
                "capped": 1,
                "lot_blocked": 0,
                "band_skipped": 2,
                "effective_target_frac": 0.006,
            },
            {
                "turnover": 0.20,
                "executed_turnover": 0.18,
                "unfilled_turnover": 0.02,
                "selected_n": 3,
                "gross_weight": 0.8,
                "market_mult": 1.0,
                "active_drawdown_throttle_scale": 0.9,
                "cost": 0.002,
                "commission": 0.0004,
                "stamp_tax": 0.0006,
                "slippage": 0.0010,
                "blocked_buy": 0,
                "blocked_sell": 1,
                "adv_blocked": 1,
                "missing_adv": 0,
                "no_open": 0,
                "capped": 0,
                "lot_blocked": 1,
                "band_skipped": 0,
                "effective_target_frac": 0.004,
            },
        ]
    )

    row = summarize_open_ledger_result(
        returns,
        diag_df,
        closed_ages=[2, 4],
        target_frac=0.006,
        hold_frac=0.10,
        args=args,
    )

    assert row["target_frac"] == 0.006
    assert row["hold_frac"] == 0.10
    assert row["n_return_days"] == 3
    assert row["avg_turnover"] == pytest.approx(0.15)
    assert row["avg_executed_turnover"] == pytest.approx(0.135)
    assert row["avg_holding_days"] == pytest.approx(3.0)
    assert row["avg_names"] == pytest.approx(2.5)
    assert row["avg_gross_weight"] == pytest.approx(0.75)
    assert row["avg_portfolio_beta_60d"] == 0.0
    assert row["avg_market_mult"] == pytest.approx(0.85)
    assert row["active_drawdown_throttle_lookback"] == 15
    assert row["active_drawdown_throttle_trigger"] == pytest.approx(-0.03)
    assert row["active_drawdown_throttle_scale"] == pytest.approx(0.9)
    assert row["active_drawdown_throttle_cooldown"] == 5
    assert row["avg_active_drawdown_throttle_scale"] == pytest.approx(0.95)
    assert row["active_drawdown_throttle_days"] == 1
    assert row["total_cost"] == pytest.approx(0.003)
    assert row["blocked_buy"] == 1
    assert row["blocked_sell"] == 1
    assert row["adv_blocked"] == 1
    assert row["no_trade_blocked"] == 0
    assert row["limit_up_open_blocked"] == 0
    assert row["limit_down_open_blocked"] == 0
    assert row["limit_up_touch_blocked"] == 0
    assert row["limit_down_touch_blocked"] == 0
    assert row["new_stock_buy_blocked"] == 0
    assert row["capped"] == 1
    assert row["lot_blocked"] == 1
    assert row["band_skipped"] == 2
    assert row["execution_lag"] == 1
    assert row["max_new_names"] == 5
    assert row["avg_effective_target_frac"] == pytest.approx(0.005)


def test_load_index_returns_missing_file_returns_neutral_series(tmp_path):
    dates = pd.date_range("2025-01-01", periods=3)

    close, daily = load_index_returns(tmp_path, "missing.csv", dates)

    assert close.index.equals(dates)
    assert daily.index.equals(dates)
    assert close.isna().all()
    assert daily.tolist() == [0.0, 0.0, 0.0]


def test_load_index_returns_reads_close_and_daily_returns(tmp_path):
    path = tmp_path / "idx.csv"
    path.write_text(
        "date,close\n2025-01-01,100\n2025-01-02,110\n2025-01-03,99\n",
        encoding="utf-8",
    )
    dates = pd.date_range("2025-01-01", periods=3)

    close, daily = load_index_returns(tmp_path, "idx.csv", dates)

    assert close.tolist() == [100.0, 110.0, 99.0]
    assert daily.iloc[0] == 0.0
    assert daily.iloc[1] == pytest.approx(0.10)
    assert daily.iloc[2] == pytest.approx(-0.10)


def test_load_ohlc_money_reads_stock_csvs_and_scales_money(tmp_path):
    (tmp_path / "A.csv").write_text(
        "trade_date,open,close,money\n"
        "2025-01-02,10,11,100\n"
        "2025-01-03,12,13,200\n",
        encoding="utf-8",
    )
    (tmp_path / "B.csv").write_text(
        "trade_date,open,close,money\n"
        "2025-01-03,20,21,300\n",
        encoding="utf-8",
    )

    open_df, close_df, money_df = load_ohlc_money(tmp_path, ["A", "B", "MISSING"], 1000.0, 0)

    assert open_df.columns.tolist() == ["A", "B"]
    assert close_df.loc[pd.Timestamp("2025-01-02"), "A"] == 11.0
    assert pd.isna(open_df.loc[pd.Timestamp("2025-01-02"), "B"])
    assert money_df.loc[pd.Timestamp("2025-01-03"), "B"] == 300_000.0


def test_load_ohlc_money_filters_date_window_and_uses_cache(tmp_path, capsys):
    data_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    (data_dir / "A.csv").write_text(
        "trade_date,open,close,money\n"
        "2024-12-31,9,9,50\n"
        "2025-01-02,10,11,100\n"
        "2025-01-03,12,13,200\n"
        "2025-01-06,14,15,300\n",
        encoding="utf-8",
    )

    open_df, _, money_df = load_ohlc_money(
        data_dir,
        ["A"],
        1000.0,
        0,
        start_date="2025-01-02",
        end_date="2025-01-03",
        cache_dir=cache_dir,
    )

    assert open_df.index.tolist() == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
    ]
    assert money_df.loc[pd.Timestamp("2025-01-03"), "A"] == 200_000.0
    capsys.readouterr()

    cached_open, _, _ = load_ohlc_money(
        data_dir,
        ["A"],
        1000.0,
        0,
        start_date="2025-01-02",
        end_date="2025-01-03",
        cache_dir=cache_dir,
    )

    assert cached_open.equals(open_df)
    assert "loaded OHLC cache" in capsys.readouterr().out


def test_build_execution_constraint_masks_uses_real_limit_rules(tmp_path):
    dates = pd.to_datetime(["2020-08-21", "2020-08-24"])
    columns = ["300001.SZ", "688001.SH", "430001.BJ"]
    open_df = pd.DataFrame(
        [[10.0, 10.0, 10.0], [12.0, 12.0, 13.0]],
        index=dates,
        columns=columns,
    )
    close_df = pd.DataFrame(
        [[10.0, 10.0, 10.0], [12.0, 12.0, 13.0]],
        index=dates,
        columns=columns,
    )
    high_df = pd.DataFrame(
        [[10.0, 10.0, 10.0], [12.0, 12.0, 13.0]],
        index=dates,
        columns=columns,
    )
    low_df = pd.DataFrame(
        [[10.0, 10.0, 10.0], [9.5, 8.0, 7.0]],
        index=dates,
        columns=columns,
    )
    volume_df = pd.DataFrame(1000.0, index=dates, columns=columns)
    money_df = pd.DataFrame(1_000_000.0, index=dates, columns=columns)

    masks = build_execution_constraint_masks(
        open_df,
        close_df,
        high_df,
        low_df,
        volume_df,
        money_df,
        tmp_path,
        block_intraday_limit_touch=True,
        no_limit_first_trading_days=0,
        min_buy_listing_days=0,
    )

    day = pd.Timestamp("2020-08-24")
    assert masks["limit_up_open"].loc[day, "300001.SZ"]
    assert masks["limit_up_open"].loc[day, "688001.SH"]
    assert masks["limit_up_open"].loc[day, "430001.BJ"]
    assert masks["limit_down_touch"].loc[day, "688001.SH"]
    assert masks["limit_down_touch"].loc[day, "430001.BJ"]
    assert masks["buy_block"].loc[day, "300001.SZ"]


def test_build_execution_constraint_masks_rounds_limit_price_to_cent(tmp_path):
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    columns = ["000001.SZ"]
    open_df = pd.DataFrame([[10.01], [11.01]], index=dates, columns=columns)
    close_df = pd.DataFrame([[10.005], [11.01]], index=dates, columns=columns)
    high_df = open_df.copy()
    low_df = open_df.copy()
    volume_df = pd.DataFrame(1000.0, index=dates, columns=columns)
    money_df = pd.DataFrame(1_000_000.0, index=dates, columns=columns)

    masks = build_execution_constraint_masks(
        open_df,
        close_df,
        high_df,
        low_df,
        volume_df,
        money_df,
        tmp_path,
        no_limit_first_trading_days=0,
        min_buy_listing_days=0,
    )

    assert masks["limit_up_open"].loc[pd.Timestamp("2025-01-03"), "000001.SZ"]


def test_build_execution_constraint_masks_disables_limits_for_new_stock_first_days(tmp_path):
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    columns = ["001234.SZ"]
    (tmp_path / "stable_stocks.csv").write_text(
        "ts_code,symbol,name,list_date\n001234.SZ,001234,新股A,20250102\n",
        encoding="utf-8",
    )
    open_df = pd.DataFrame([[10.0], [11.0]], index=dates, columns=columns)
    close_df = pd.DataFrame([[10.0], [11.0]], index=dates, columns=columns)
    high_df = open_df.copy()
    low_df = open_df.copy()
    volume_df = pd.DataFrame(1000.0, index=dates, columns=columns)
    money_df = pd.DataFrame(1_000_000.0, index=dates, columns=columns)

    masks = build_execution_constraint_masks(
        open_df,
        close_df,
        high_df,
        low_df,
        volume_df,
        money_df,
        tmp_path,
        no_limit_first_trading_days=5,
        min_buy_listing_days=0,
    )

    assert not masks["limit_up_open"].loc[pd.Timestamp("2025-01-03"), "001234.SZ"]


def test_build_execution_constraint_masks_blocks_buy_before_min_listing_days(tmp_path):
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    columns = ["001234.SZ"]
    (tmp_path / "stable_stocks.csv").write_text(
        "ts_code,symbol,name,list_date\n001234.SZ,001234,新股A,20250102\n",
        encoding="utf-8",
    )
    open_df = pd.DataFrame([[10.0], [10.1], [10.2]], index=dates, columns=columns)
    close_df = open_df.copy()
    high_df = open_df.copy()
    low_df = open_df.copy()
    volume_df = pd.DataFrame(1000.0, index=dates, columns=columns)
    money_df = pd.DataFrame(1_000_000.0, index=dates, columns=columns)

    masks = build_execution_constraint_masks(
        open_df,
        close_df,
        high_df,
        low_df,
        volume_df,
        money_df,
        tmp_path,
        no_limit_first_trading_days=5,
        min_buy_listing_days=60,
    )

    day = pd.Timestamp("2025-01-06")
    assert masks["new_stock_buy_block"].loc[day, "001234.SZ"]
    assert masks["buy_block"].loc[day, "001234.SZ"]
    assert not masks["sell_block"].loc[day, "001234.SZ"]


def test_infer_ohlc_load_window_uses_signal_range_and_data_cutoff():
    rows = [
        {"date": "2025-01-10"},
        {"date": "2025-02-03"},
    ]

    start, end = infer_ohlc_load_window(
        rows,
        max_data_date="2025-12-31",
        execution_lag=1,
        lookback_days=30,
    )

    assert start == pd.Timestamp("2024-12-11")
    assert end == pd.Timestamp("2025-12-31")


def test_recompute_adv_uses_shifted_rolling_mean():
    money = pd.DataFrame({"A": [100.0, 200.0, 300.0, 400.0]})

    adv = recompute_adv(money, 4)

    assert pd.isna(adv.iloc[0, 0])
    assert pd.isna(adv.iloc[1, 0])
    assert pd.isna(adv.iloc[2, 0])
    assert adv.iloc[3, 0] == pytest.approx(200.0)


def test_run_open_ledger_executes_signal_at_next_open():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    open_df = pd.DataFrame({"A": [9.0, 10.0, 11.0]}, index=dates)
    close_df = pd.DataFrame({"A": [10.0, 10.0, 11.0]}, index=dates)
    adv_df = pd.DataFrame({"A": [1_000_000.0] * 3}, index=dates)
    idx_close = pd.Series([100.0, 100.0, 100.0], index=dates)
    idx_daily = idx_close.pct_change().fillna(0.0)
    args = make_args(
        portfolio_value=100_000.0,
        execution_lag=0,
        market_timing_mode="none",
        market_min_mult=0.2,
        market_max_mult=1.0,
        legacy_bear_mult=0.7,
        legacy_crash_mult=0.3,
        max_weight=1.0,
        max_new_names=0,
        min_adv_cny=1.0,
        adv_participation_cap=1.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        slippage_rate=0.0,
        min_commission_cny=0.0,
    )
    rows = [{"date": pd.Timestamp("2025-01-02"), "codes": ["A"], "alpha": [1.0]}]
    execution_trace = []
    position_trace = []

    summary, returns, diagnostics = run_open_ledger(
        rows,
        open_df,
        close_df,
        adv_df,
        target_frac=1.0,
        hold_frac=1.0,
        args=args,
        idx_close=idx_close,
        idx_daily=idx_daily,
        execution_trace_sink=execution_trace,
        position_trace_sink=position_trace,
    )

    assert diagnostics["date"].tolist() == ["2025-01-03 00:00:00"]
    assert returns["date"].tolist() == [dates[1], dates[2]]
    assert returns["return"].tolist() == pytest.approx([0.0, 0.10])
    assert returns["benchmark_return"].tolist() == pytest.approx([0.0, 0.0])
    assert returns["active_return"].tolist() == pytest.approx([0.0, 0.10])
    assert returns["equity_cny"].tolist() == pytest.approx([100_000.0, 110_000.0])
    assert summary["n_return_days"] == 2
    assert "information_ratio" in summary
    assert "active_ann" in summary
    assert "portfolio_beta_60d" in diagnostics.columns
    assert "portfolio_specific_vol_60d" in diagnostics.columns
    assert diagnostics["holdings"].str.contains("A=").all()
    assert execution_trace[0]["date"] == "2025-01-03"
    assert execution_trace[0]["code"] == "A"
    assert execution_trace[0]["status"] == "filled"
    assert position_trace == [
        {
            "date": "2025-01-03",
            "code": "A",
            "shares": 10000.0,
            "mark_price": 10.0,
            "market_value_cny": 100000.0,
            "weight": 1.0,
        }
    ]


def test_run_open_ledger_prepared_context_matches_uncached_result():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    open_df = pd.DataFrame({"A": [9.0, 10.0, 11.0, 12.0]}, index=dates)
    close_df = pd.DataFrame({"A": [10.0, 10.0, 11.0, 12.0]}, index=dates)
    adv_df = pd.DataFrame({"A": [1_000_000.0] * len(dates)}, index=dates)
    idx_close = pd.Series([100.0] * len(dates), index=dates)
    args = make_args(
        portfolio_value=100_000.0,
        execution_lag=0,
        market_timing_mode="none",
        market_min_mult=0.2,
        market_max_mult=1.0,
        legacy_bear_mult=0.7,
        legacy_crash_mult=0.3,
        max_weight=1.0,
        min_adv_cny=1.0,
        adv_participation_cap=1.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        slippage_rate=0.0,
        min_commission_cny=0.0,
    )
    rows = [{"date": dates[0], "codes": ["A"], "alpha": [1.0]}]
    uncached = run_open_ledger(
        rows, open_df, close_df, adv_df, 1.0, 1.0, args, idx_close, idx_close.pct_change().fillna(0.0)
    )
    context = prepare_open_ledger_context(open_df, close_df)
    cached = run_open_ledger(
        rows, open_df, close_df, adv_df, 1.0, 1.0, args, idx_close, idx_close.pct_change().fillna(0.0),
        prepared_context=context,
    )

    assert cached[0]["ann"] == pytest.approx(uncached[0]["ann"])
    pd.testing.assert_frame_equal(cached[1], uncached[1])
    pd.testing.assert_frame_equal(cached[2], uncached[2])


def test_estimate_trailing_stock_risk_and_portfolio_summary():
    idx_close = pd.Series([100, 101, 102, 103, 104, 105], dtype=float)
    idx_daily = idx_close.pct_change().fillna(0.0)
    idx_ret = idx_daily.to_numpy()
    stock_a = 2.0 * idx_ret[1:] + 0.001
    stock_b = -1.0 * idx_ret[1:] - 0.001
    close_ret_daily = np.vstack([stock_a, stock_b])

    beta, residual_vol, obs = estimate_trailing_stock_risk(
        close_ret_daily,
        idx_daily,
        day_pos=5,
        window=4,
    )
    risk = summarize_portfolio_risk(
        weights=np.array([0.4, 0.2]),
        stock_beta=beta,
        residual_vol=residual_vol,
    )

    assert obs >= 3
    assert beta[0] > 1.0
    assert beta[1] < 0.0
    assert risk["portfolio_beta_60d"] == pytest.approx(0.4 * beta[0] + 0.2 * beta[1])
    assert risk["portfolio_specific_vol_60d"] >= 0.0


def test_encode_holdings_skips_zero_weights():
    encoded = encode_holdings(
        ["A", "B", "C"],
        np.array([0.123456789, 0.0, -0.02]),
    )

    assert encoded == "A=0.1234567890;C=-0.0200000000"


def test_load_industry_map_normalizes_codes(tmp_path):
    path = tmp_path / "industry.csv"
    path.write_text(
        "code,industry\nsh.600000,Bank\nsz.000001,Bank\n",
        encoding="utf-8",
    )

    industry_map = load_industry_map(path)

    assert normalize_ts_code("sh.600000") == "600000.SH"
    assert normalize_ts_code("bj.430001") == "430001.BJ"
    assert industry_map == {"600000.SH": "Bank", "000001.SZ": "Bank"}


def test_apply_industry_selection_cap_refills_from_ranked_candidates():
    selected = ["A", "B", "C", "D"]
    candidates = ["A", "B", "C", "D", "E", "F"]
    industry_map = {
        "A": "Tech",
        "B": "Tech",
        "C": "Tech",
        "D": "Bank",
        "E": "Energy",
        "F": "Health",
    }

    capped, diag = apply_industry_selection_cap(
        selected,
        candidates,
        industry_map,
        max_industry_weight=0.50,
        gross_weight=1.0,
        max_weight=1.0,
        target_n=4,
    )

    assert capped == ["A", "B", "D", "E"]
    assert diag["industry_cap_active"] == 1
    assert diag["industry_cap_max_names"] == 2
    assert diag["industry_cap_removed"] == 1


def test_state_aware_selection_rank_penalizes_fragile_new_names():
    dates = pd.to_datetime([f"2024-01-{day:02d}" for day in range(1, 31)])
    close_df = pd.DataFrame(
        {
            "OLD": np.linspace(10.0, 10.5, len(dates)),
            "MOM": np.linspace(10.0, 18.0, len(dates)),
            "CALM": np.linspace(10.0, 10.6, len(dates)),
        },
        index=dates,
    )
    close_ret_daily = close_df.to_numpy(dtype=float).T[:, 1:] / close_df.to_numpy(dtype=float).T[:, :-1] - 1.0
    idx_daily = pd.Series(np.zeros(len(dates)), index=dates)
    global_frame = pd.DataFrame(
        {"global_us_hk_pressure": [0.10]},
        index=pd.DatetimeIndex([dates[-1]]),
    )
    args = SimpleNamespace(
        state_aware_selection_mode="risk_rank",
        state_aware_selection_pressure_col="global_us_hk_pressure",
        state_aware_selection_pressure_threshold=0.035,
        state_aware_selection_pressure_width=0.055,
        state_aware_selection_min_stress=0.0,
        state_aware_selection_rank_penalty=0.50,
        state_aware_selection_top_frac=0.50,
        state_aware_selection_crowd_scale=0.50,
        state_aware_selection_momentum_weight=1.0,
        state_aware_selection_beta_weight=0.0,
        state_aware_selection_vol_weight=0.0,
        state_aware_selection_industry_weight=0.0,
    )

    selected, diag = apply_state_aware_selection_rank(
        selected=["OLD", "MOM"],
        row={"date": dates[-1], "codes": ["MOM", "CALM", "OLD"]},
        current_selected=["OLD"],
        target_n=2,
        codes=["OLD", "MOM", "CALM"],
        code2idx={"OLD": 0, "MOM": 1, "CALM": 2},
        close_df=close_df,
        close_ret_daily=close_ret_daily,
        idx_daily=idx_daily,
        day=len(dates) - 1,
        stock_beta=np.zeros(3),
        residual_vol=np.zeros(3),
        industry_map={"OLD": "A", "MOM": "B", "CALM": "C"},
        global_risk_frame=global_frame,
        args=args,
    )

    assert selected == ["OLD", "CALM"]
    assert diag["state_aware_selection_active"] == 1
    assert diag["state_aware_selection_changed"] == 1
    assert diag["state_aware_selection_stress"] == pytest.approx(1.0)


def test_state_aware_selection_suppresses_riskier_replacement():
    dates = pd.to_datetime([f"2024-01-{day:02d}" for day in range(1, 31)])
    close_df = pd.DataFrame(
        {
            "OLD": np.linspace(10.0, 10.5, len(dates)),
            "MOM": np.linspace(10.0, 18.0, len(dates)),
            "CALM": np.linspace(10.0, 10.6, len(dates)),
        },
        index=dates,
    )
    close_ret_daily = close_df.to_numpy(dtype=float).T[:, 1:] / close_df.to_numpy(dtype=float).T[:, :-1] - 1.0
    idx_daily = pd.Series(np.zeros(len(dates)), index=dates)
    global_frame = pd.DataFrame(
        {"global_defensive_pressure": [0.10]},
        index=pd.DatetimeIndex([dates[-1]]),
    )
    args = SimpleNamespace(
        state_aware_selection_mode="risk_suppress",
        state_aware_selection_pressure_col="global_defensive_pressure",
        state_aware_selection_pressure_threshold=0.035,
        state_aware_selection_pressure_width=0.055,
        state_aware_selection_min_stress=0.0,
        state_aware_selection_risk_delta_threshold=0.15,
        state_aware_selection_top_frac=0.50,
        state_aware_selection_crowd_scale=0.50,
        state_aware_selection_momentum_weight=1.0,
        state_aware_selection_beta_weight=0.0,
        state_aware_selection_vol_weight=0.0,
        state_aware_selection_industry_weight=0.0,
    )

    selected, diag = apply_state_aware_selection_rank(
        selected=["MOM", "CALM"],
        row={"date": dates[-1], "codes": ["MOM", "CALM", "OLD"]},
        current_selected=["OLD"],
        target_n=2,
        codes=["OLD", "MOM", "CALM"],
        code2idx={"OLD": 0, "MOM": 1, "CALM": 2},
        close_df=close_df,
        close_ret_daily=close_ret_daily,
        idx_daily=idx_daily,
        day=len(dates) - 1,
        stock_beta=np.zeros(3),
        residual_vol=np.zeros(3),
        industry_map={"OLD": "A", "MOM": "B", "CALM": "C"},
        global_risk_frame=global_frame,
        args=args,
    )

    assert selected == ["OLD", "CALM"]
    assert diag["state_aware_selection_suppressed"] == 1
    assert diag["state_aware_selection_mean_suppressed_risk_delta"] >= 0.15


def test_save_stage_breakdown_writes_yearly_and_monthly_files(tmp_path):
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-02-03"]),
            "return": [0.01, -0.005, 0.002],
            "benchmark_return": [0.002, -0.001, 0.001],
            "active_return": [0.008, -0.004, 0.001],
        }
    )

    save_stage_breakdown(tmp_path, {"tiny": returns})

    yearly = pd.read_csv(tmp_path / "yearly_summary.csv")
    monthly = pd.read_csv(tmp_path / "monthly_summary.csv")
    assert set(yearly["period"].astype(str)) == {"2025", "all"}
    assert monthly["month"].tolist() == ["2025-01", "2025-02"]
    assert "information_ratio" in yearly.columns
    assert "active_sum_return" in monthly.columns


def test_compute_market_multiplier_none_and_short_history():
    idx_close = pd.Series(np.linspace(100, 120, 80))
    idx_daily = idx_close.pct_change().fillna(0.0)
    ret_daily = np.zeros((2, 79))

    assert compute_market_multiplier(idx_close, idx_daily, ret_daily, 10, "none", 0.2, 1.0) == 1.0
    assert compute_market_multiplier(idx_close, idx_daily, ret_daily, 10, "legacy", 0.2, 1.0) == 1.0


def test_compute_market_multiplier_legacy_bear_and_crash():
    # Last value is below its 60-day average and 120-day return is below -10%.
    idx_close = pd.Series(np.linspace(130, 100, 130))
    idx_daily = idx_close.pct_change().fillna(0.0)
    ret_daily = np.zeros((2, 129))

    mult = compute_market_multiplier(
        idx_close,
        idx_daily,
        ret_daily,
        129,
        "legacy",
        0.2,
        1.0,
        legacy_bear_mult=0.7,
        legacy_crash_mult=0.3,
    )

    assert mult == pytest.approx(0.3)


def test_compute_market_multiplier_dynamic_is_clipped_to_bounds():
    idx_close = pd.Series(np.linspace(100, 130, 130))
    idx_daily = idx_close.pct_change().fillna(0.0)
    ret_daily = np.ones((5, 129)) * 0.001

    mult = compute_market_multiplier(idx_close, idx_daily, ret_daily, 129, "dynamic", 0.2, 0.8)

    assert 0.2 <= mult <= 0.8


def test_compute_market_multiplier_dynamic_uses_current_signal_day_volatility():
    idx_close = pd.Series(np.linspace(100, 130, 130))
    quiet_daily = pd.Series(np.zeros(130))
    shocked_daily = quiet_daily.copy()
    shocked_daily.iloc[129] = 1.0
    ret_daily = np.ones((5, 129)) * 0.001

    quiet = compute_market_multiplier(
        idx_close, quiet_daily, ret_daily, 129, "dynamic", 0.2, 1.0
    )
    shocked = compute_market_multiplier(
        idx_close, shocked_daily, ret_daily, 129, "dynamic", 0.2, 1.0
    )

    assert shocked < quiet


def test_compute_market_multiplier_rejects_unknown_mode():
    idx_close = pd.Series(np.linspace(100, 130, 80))
    idx_daily = idx_close.pct_change().fillna(0.0)
    ret_daily = np.zeros((2, 79))

    with pytest.raises(ValueError, match="Unknown market_timing_mode"):
        compute_market_multiplier(idx_close, idx_daily, ret_daily, 70, "bad", 0.2, 1.0)
