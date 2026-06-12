import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from run.backtest_retention_execution_constraints import apply_execution_constraints


def make_args():
    return SimpleNamespace(
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


def main():
    dates = pd.to_datetime(["2026-05-15", "2026-05-18"])
    close = pd.DataFrame(
        {"000001.SZ": [10.0, 10.0], "000002.SZ": [20.0, 20.0]},
        index=dates,
    )
    adv = pd.DataFrame(
        {"000001.SZ": [10_000_000.0, 10_000_000.0], "000002.SZ": [10_000_000.0, 10_000_000.0]},
        index=dates,
    )
    args = make_args()

    shares, cash, executed, info = apply_execution_constraints(
        desired_weights=np.array([0.5, 0.5]),
        current_shares=np.zeros(2),
        cash=500_000.0,
        equity=500_000.0,
        close_df=close,
        adv_df=adv,
        day_pos=1,
        args=args,
    )
    assert np.all(shares % 100 == 0)
    assert np.all(executed % 100 == 0)
    assert cash >= 0.0
    assert info["executed_turnover"] > 0.0

    band_args = make_args()
    band_args.rebalance_band = 0.20
    band_shares, band_cash, band_executed, band_info = apply_execution_constraints(
        desired_weights=np.array([0.5, 0.5]),
        current_shares=np.array([23_000.0, 13_000.0]),
        cash=10_000.0,
        equity=500_000.0,
        close_df=close,
        adv_df=adv,
        day_pos=1,
        args=band_args,
    )
    assert np.array_equal(band_executed, np.zeros(2))
    assert np.array_equal(band_shares, np.array([23_000.0, 13_000.0]))
    assert band_cash == 10_000.0
    assert band_info["band_skipped"] == 2

    shares, cash, executed, info = apply_execution_constraints(
        desired_weights=np.zeros(2),
        current_shares=shares,
        cash=cash,
        equity=cash + float(np.dot(shares, close.iloc[1].to_numpy(float))),
        close_df=close,
        adv_df=adv,
        day_pos=1,
        args=args,
    )
    assert np.array_equal(shares, np.zeros(2))
    assert np.all(executed <= 0)
    assert cash > 0.0
    assert info["unfilled_turnover"] == 0.0
    print("test_execution_constraints.py: ALL PASSED")


if __name__ == "__main__":
    main()
