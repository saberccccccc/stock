"""Tests for the Phase 3 factor risk model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.risk_model import (
    normalize_code,
    load_industry_map,
    compute_stock_beta,
    compute_factor_exposures,
    portfolio_risk_decomposition,
)
import numpy as np
import pandas as pd


def test_normalize_code_sh_prefix():
    assert normalize_code("sh.600000") == "600000.SH"


def test_normalize_code_sz_prefix():
    assert normalize_code("sz.000001") == "000001.SZ"


def test_normalize_code_passthrough():
    assert normalize_code("600000.SH") == "600000.SH"


def test_normalize_code_empty_returns_none():
    assert normalize_code("") is None
    assert normalize_code(None) is None


def test_load_industry_map_returns_valid_data():
    ind_map, ind_codes, n_inds = load_industry_map("data/stock_industry.csv")
    assert len(ind_map) > 4000
    assert n_inds > 30
    sample = list(ind_map.keys())[0]
    assert "." in sample  # normalized code format


def test_compute_stock_beta_identity():
    r_i = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    r_m = r_i.copy()
    beta = compute_stock_beta(r_i, r_m)
    assert abs(beta - 1.0) < 0.01


def test_compute_stock_beta_high():
    r_m = np.array([0.005, -0.01, 0.015, -0.005, 0.01])
    r_i = r_m * 2.0
    beta = compute_stock_beta(r_i, r_m)
    assert abs(beta - 2.0) < 0.5


def test_compute_stock_beta_insufficient_data():
    beta = compute_stock_beta(np.array([0.01]), np.array([0.01]))
    assert np.isnan(beta)


def test_compute_stock_beta_zero_market_var():
    beta = compute_stock_beta(np.array([0.01, 0.02]), np.array([0.0, 0.0]))
    assert np.isnan(beta)


def test_portfolio_risk_decomposition_empty():
    result = portfolio_risk_decomposition(pd.DataFrame(), pd.DataFrame())
    assert np.isnan(result["total_var"])


def test_portfolio_risk_decomposition_simple():
    holdings = pd.DataFrame({"code": ["600000.SH", "600004.SH"], "weight": [0.5, 0.5]})
    exposures = pd.DataFrame({
        "code": ["600000.SH", "600004.SH"],
        "beta": [1.0, 1.5],
        "industry_code": [0, 1],
        "log_mkt_cap": [22.0, 21.0],
        "momentum_20d": [0.01, -0.02],
        "volatility_60d": [0.02, 0.03],
        "specific_vol": [0.015, 0.020],
        "liquidity_ratio": [0.1, 0.05],
    })
    result = portfolio_risk_decomposition(holdings, exposures)
    assert result["total_var"] >= 0
    assert result["factor_share"] >= 0
    assert result["specific_share"] >= 0
    assert abs(result["factor_share"] + result["specific_share"] - 1.0) < 1e-6
    assert len(result["top_contributors"]) > 0
