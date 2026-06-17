from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest.open_ledger import apply_open_ledger_constraints, open_limit_trade_mask


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
