# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def test_calc_metrics_known_input():
    import numpy as np
    from backtest.reports import calc_metrics, calc_extended_metrics

    returns = np.array([0.01, -0.005, 0.02, -0.01, 0.005])
    ann, sharpe, mdd = calc_metrics(returns)
    assert isinstance(ann, float), f"expected float, got {type(ann)}"
    assert isinstance(sharpe, float)
    assert isinstance(mdd, float)
    assert 0 <= mdd <= 1, f"mdd should be 0-1, got {mdd}"
    print(f"  calc_metrics: ann={ann:.2f}, sharpe={sharpe:.2f}, mdd={mdd:.4f}")

    ext = calc_extended_metrics(returns)
    required_keys = ['ann_return', 'sharpe', 'max_drawdown', 'calmar', 'sortino', 'win_rate', 'profit_loss_ratio', 'total_days']
    for k in required_keys:
        assert k in ext, f"missing key: {k}"
    print(f"  calc_extended_metrics: {len(ext)} keys, all required present")


def test_calc_metrics_empty():
    from backtest.reports import calc_metrics
    a, s, m = calc_metrics([])
    assert a == 0 and s == 0 and m == 0, f"empty should return zeros, got {(a,s,m)}"
    print("  empty returns -> (0,0,0): OK")


if __name__ == "__main__":
    test_calc_metrics_known_input()
    test_calc_metrics_empty()
    print("test_metrics.py: ALL PASSED")
