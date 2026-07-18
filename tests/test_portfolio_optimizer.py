import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backtest.portfolio_optimizer import portfolio_weights

def test_basic_convergence():
    n = 20
    alpha = np.random.randn(n) * 0.01
    X = np.column_stack([np.abs(np.random.randn(n)) + 0.8, np.random.randn(n) * 0.3])
    spec = np.random.rand(n) * 0.02 + 0.01
    w, ok, _ = portfolio_weights(alpha, X, spec, risk_aversion=5, gross_target=0.90)
    assert ok and np.all(w >= 0) and np.sum(w) > 0.70

def test_max_weight():
    n = 30
    alpha = np.zeros(n); alpha[0] = 0.05
    X = np.ones((n, 2)); X[:, 0] = 1.0
    spec = np.ones(n) * 0.02
    w, ok, _ = portfolio_weights(alpha, X, spec, risk_aversion=0, gross_target=0.90, max_weight=0.05)
    assert ok and w.max() <= 0.051

def test_risk_penalty():
    n = 30; np.random.seed(42)
    alpha = np.random.randn(n) * 0.01
    X = np.column_stack([np.abs(np.random.randn(n))*0.5+0.8, np.random.randn(n)*0.3])
    X[0, 0] = 2.5
    spec = np.random.rand(n)*0.02+0.01
    wL, _, _ = portfolio_weights(alpha, X, spec, risk_aversion=1, gross_target=0.90)
    wH, _, _ = portfolio_weights(alpha, X, spec, risk_aversion=100, gross_target=0.90)
    assert wH[0] <= wL[0] + 0.01

def test_empty():
    w, ok, msg = portfolio_weights(np.array([]), np.empty((0,2)), np.array([]))
    assert not ok

def test_industry():
    n = 20; np.random.seed(1)
    alpha = np.random.randn(n)*0.01
    X = np.column_stack([np.abs(np.random.randn(n))+0.8, np.random.randn(n)*0.3])
    spec = np.random.rand(n)*0.02+0.01
    ind = np.repeat([0,1,2,3], 5)
    w, ok, _ = portfolio_weights(alpha, X, spec, risk_aversion=10, gross_target=0.90, industry_codes=ind, max_industry_weight=0.30)
    assert ok
    for i in range(4):
        assert np.sum(w[ind==i]) <= 0.31
