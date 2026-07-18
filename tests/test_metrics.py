# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import numpy as np

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


def test_max_drawdown_includes_loss_from_initial_capital():
    from backtest.reports import calc_metrics

    _, _, mdd = calc_metrics([-0.10, 0.0])

    assert np.isclose(mdd, 0.10)


def test_sortino_uses_downside_deviation_over_all_days():
    from backtest.reports import calc_extended_metrics

    returns = np.array([0.02, -0.01, 0.01, 0.0])
    expected_downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    expected = returns.mean() / (expected_downside + 1e-8) * np.sqrt(252)

    assert np.isclose(calc_extended_metrics(returns)["sortino"], expected)


def test_active_management_metrics_against_benchmark():
    from backtest.reports import calc_active_management_metrics

    strategy = np.array([0.02, -0.01, 0.01, 0.0])
    benchmark = np.array([0.01, -0.005, 0.0, 0.002])
    metrics = calc_active_management_metrics(strategy, benchmark)

    active = strategy - benchmark
    expected_te = np.std(active) * np.sqrt(252)
    expected_ir = active.mean() / (active.std() + 1e-8) * np.sqrt(252)

    assert metrics["tracking_error"] == np.float64(expected_te)
    assert metrics["information_ratio"] == np.float64(expected_ir)
    assert metrics["active_ann"] != metrics["benchmark_ann"]
    assert "beta_to_benchmark" in metrics


def test_training_selection_metrics():
    from core.train_utils import _mean_std_score, _top_frac_tag

    assert _top_frac_tag(0.006) == "0p6"
    assert _top_frac_tag(0.01) == "1"
    assert _top_frac_tag(0.025) == "2p5"
    assert _mean_std_score([1.0, 2.0, 3.0]) > 0.0
    assert _mean_std_score([1.0]) == 0.0
    print("  training selection metric tags/stability: OK")


if __name__ == "__main__":
    test_calc_metrics_known_input()
    test_calc_metrics_empty()
    test_training_selection_metrics()
    print("test_metrics.py: ALL PASSED")
