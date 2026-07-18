"""Portfolio optimizer using SciPy SLSQP.

Converts alpha predictions, factor risk model, and constraints
into optimal long-only target weights.

Formulation (minimize):
  -gain = -(alpha^T w - risk_aversion * w^T Sigma w - turnover_penalty * ||w - w0||_1)
  subject to: gross_target_lower <= sum(w) <= gross_target_upper
              0 <= w_i <= max_weight
              industry constraints (optional)
"""
import numpy as np
from scipy.optimize import minimize


def portfolio_weights(
    alpha,
    factor_exposures,
    specific_vol,
    risk_aversion=10.0,
    max_weight=0.05,
    gross_target=0.90,
    factor_cov=None,
    turnover_penalty=0.0,
    prev_weights=None,
    industry_codes=None,
    max_industry_weight=0.0,
    max_iter=1000,
):
    """Compute optimal portfolio weights.

    Parameters
    ----------
    alpha : np.ndarray, shape (n,)
        Daily alpha predictions for each stock.
    factor_exposures : np.ndarray, shape (n, n_factors)
        Stock-level factor exposures (e.g. from risk_model.compute_factor_exposures).
    specific_vol : np.ndarray, shape (n,)
        Stock-specific residual volatility.
    risk_aversion : float
        Risk aversion coefficient (higher = more risk penalty).
    max_weight : float
        Maximum weight per stock.
    gross_target : float
        Target gross exposure (sum of weights).
    factor_cov : np.ndarray, shape (n_factors, n_factors)
        Factor covariance matrix. If None, identity assumed.
    turnover_penalty : float
        Penalty for deviation from previous weights.
    prev_weights : np.ndarray, shape (n,)
        Previous period weights (for turnover penalty).
    industry_codes : np.ndarray, shape (n,)
        Industry code for each stock.
    max_industry_weight : float
        Maximum total weight per industry (0 = no constraint).
    max_iter : int
        Maximum optimizer iterations.

    Returns
    -------
    weights : np.ndarray, shape (n,)
        Optimal weights (zero for stocks excluded).
    success : bool
        Whether optimizer converged.
    message : str
        Optimizer status message.
    """
    n = len(alpha)
    if n == 0:
        return np.array([]), False, "empty universe"

    # Default cov
    if factor_cov is None:
        nf = factor_exposures.shape[1]
        factor_cov = np.eye(nf)

    # Objective
    def objective(w):
        port_factor = factor_exposures.T @ w
        factor_var = port_factor @ factor_cov @ port_factor
        specific_var = np.sum(w**2 * specific_vol**2)
        total_var = factor_var + specific_var

        alpha_term = -alpha @ w  # negative = maximize
        risk_term = risk_aversion * total_var
        turnover = 0.0
        if turnover_penalty > 0 and prev_weights is not None:
            turnover = turnover_penalty * np.sum(np.abs(w - prev_weights))

        return alpha_term + risk_term + turnover

    # Constraints
    constraints = [
        # sum(w) >= gross_target * 0.9 (min investment)
        {"type": "ineq", "fun": lambda w: np.sum(w) - gross_target * 0.9},
        # sum(w) <= gross_target * 1.1 (max investment)
        {"type": "ineq", "fun": lambda w: gross_target * 1.1 - np.sum(w)},
    ]

    # Industry constraint
    if max_industry_weight > 0 and industry_codes is not None:
        ind_codes = np.unique(industry_codes[industry_codes >= 0])
        for ind in ind_codes:
            mask = industry_codes == ind
            if mask.sum() > 1:
                constraints.append(
                    {"type": "ineq", "fun": lambda w, m=mask: max_industry_weight - np.sum(w[m])}
                )

    # Bounds
    bounds = [(0, max_weight) for _ in range(n)]

    # Initial guess: equal weight
    w0 = np.ones(n) * (gross_target / n)

    result = minimize(
        objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-9},
    )

    return result.x, result.success, result.message
