from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.open_ledger import (
    apply_open_ledger_constraints,
    compute_market_multiplier,
    load_index_returns,
    open_limit_trade_mask,
    summarize_open_ledger_result,
)


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
    assert row["avg_market_mult"] == pytest.approx(0.85)
    assert row["total_cost"] == pytest.approx(0.003)
    assert row["blocked_buy"] == 1
    assert row["blocked_sell"] == 1
    assert row["adv_blocked"] == 1
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


def test_compute_market_multiplier_rejects_unknown_mode():
    idx_close = pd.Series(np.linspace(100, 130, 80))
    idx_daily = idx_close.pct_change().fillna(0.0)
    ret_daily = np.zeros((2, 79))

    with pytest.raises(ValueError, match="Unknown market_timing_mode"):
        compute_market_multiplier(idx_close, idx_daily, ret_daily, 70, "bad", 0.2, 1.0)
