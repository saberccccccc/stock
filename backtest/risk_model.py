"""
Factor risk model for small-account long-only portfolios.

Computes stock-level factor exposures and portfolio-level risk decomposition.
Diagnostics only -- does not change trades.

Factors:
  - market beta (60d rolling, from index returns)
  - industry exposure (from stock_industry.csv)
  - log market cap (size)
  - liquidity (ADV / avg market cap or turnover)
  - short-term momentum (trailing 20d return)
  - volatility (trailing 60d daily return vol)
  - stock-specific residual volatility
"""
from pathlib import Path
import numpy as np
import pandas as pd

def normalize_code(code):
    """Normalize stock code between sh/sz prefix and suffix formats.

    sh.600000 -> 600000.SH
    600000.SH -> 600000.SH (pass through)
    sz.000001 -> 000001.SZ
    """
    if code is None:
        return None
    code = str(code).strip()
    if not code:
        return None
    if "." in code:
        prefix, suffix = code.split(".", 1)
        if prefix.lower() in ("sh", "sz"):
            code = "{}.{}".format(suffix.zfill(6), prefix.upper())
    return code

def load_industry_map(industry_csv):
    """Load industry classification and build mapping."""
    df = pd.read_csv(industry_csv)
    df = df[df["industry"].notna()].copy()
    df["code"] = df["code"].apply(normalize_code)
    df = df[df["code"].notna()].copy()
    industries = sorted(df["industry"].dropna().unique())
    ind_to_code = {ind: i for i, ind in enumerate(industries)}
    return df.set_index("code")["industry"].to_dict(), ind_to_code, len(industries)

def compute_stock_beta(returns_60d, market_returns_60d):
    """Beta = cov(r_i, r_m) / var(r_m) over 60d window."""
    if len(returns_60d) < 5 or len(market_returns_60d) < 5:
        return np.nan
    cov = np.nanmean((returns_60d - np.nanmean(returns_60d)) * (market_returns_60d - np.nanmean(market_returns_60d)))
    var_m = np.nanvar(market_returns_60d)
    if var_m < 1e-12:
        return np.nan
    return cov / var_m

FACTOR_COLS = ["beta", "momentum_20d", "volatility_60d", "log_mkt_cap", "liquidity_ratio"]

def portfolio_risk_decomposition(holdings_df, factor_exposures, factor_cov=None):
    """Decompose portfolio variance into factor + specific components."""
    if holdings_df.empty or factor_exposures.empty:
        return {"factor_var": np.nan, "specific_var": np.nan, "total_var": np.nan, "top_contributors": []}

    merged = holdings_df.merge(factor_exposures, on="code", how="inner")
    if merged.empty:
        return {"factor_var": np.nan, "specific_var": np.nan, "total_var": np.nan, "top_contributors": []}

    w = merged["weight"].values
    n_cov = factor_cov.shape[0] if factor_cov is not None else len(FACTOR_COLS)
    cols = FACTOR_COLS[:n_cov]

    E_raw = merged[cols].fillna(0).values.T
    if factor_cov is None:
        E_z = (E_raw - E_raw.mean(axis=1, keepdims=True)) / np.maximum(E_raw.std(axis=1, keepdims=True), 1e-10)
        E = E_z
        factor_cov = np.eye(E.shape[0])
    else:
        E = E_raw

    port_factor_exp = E @ w
    factor_var = port_factor_exp.T @ factor_cov @ port_factor_exp
    specific_vols = merged["specific_vol"].fillna(0).values
    specific_var = np.sum(w**2 * specific_vols**2)
    total_var = factor_var + specific_var

    factor_contrib = np.diag(E.T @ factor_cov @ E) * w**2
    specific_contrib = w**2 * specific_vols**2
    total_contrib = factor_contrib + specific_contrib
    contrib_df = merged[["code"]].copy()
    contrib_df["total_risk_contrib"] = total_contrib / max(total_var, 1e-12)
    top = contrib_df.sort_values("total_risk_contrib", ascending=False).head(5)
    top_names = [{"code": r["code"], "risk_share": r["total_risk_contrib"]} for _, r in top.iterrows()]

    return {
        "factor_var": factor_var,
        "specific_var": specific_var,
        "total_var": total_var,
        "factor_share": factor_var / max(total_var, 1e-12),
        "specific_share": specific_var / max(total_var, 1e-12),
        'port_beta': float(port_factor_exp[0]) if len(port_factor_exp) > 0 else np.nan,
        "top_contributors": top_names,
    }

def compute_factor_exposures(
    date, universe_codes, open_df, close_df, money_df,
    market_returns, industry_map,
    beta_window=60, momentum_window=20, vol_window=60, min_obs=10,
):
    """Compute stock-level factor exposures for a given date."""
    rows = []
    date = pd.Timestamp(date)
    for code in universe_codes:
        if code not in close_df.columns:
            continue
        close = close_df[code].dropna()
        if len(close) < min_obs + 5:
            continue
        close_before = close[close.index <= date]
        if len(close_before) < min_obs:
            continue
        ret = close_before.pct_change().dropna()
        recent_ret = ret.tail(beta_window)
        mkt_recent = market_returns.reindex(recent_ret.index).dropna()
        aligned_ret = recent_ret.reindex(mkt_recent.index).dropna()
        mkt_aligned = mkt_recent.reindex(aligned_ret.index).dropna()
        beta = compute_stock_beta(aligned_ret.values, mkt_aligned.values)
        close_mom = close_before.tail(momentum_window + 1)
        momentum = np.nan if len(close_mom) < 2 else (close_mom.iloc[-1] / close_mom.iloc[0] - 1)
        ret_v = ret.tail(vol_window)
        vol = np.nanstd(ret_v.values) if len(ret_v) >= 5 else np.nan
        mkt_cap = np.nan
        if money_df is not None and code in money_df.columns:
            money = money_df[code].dropna()
            money_before = money[money.index <= date]
            if len(money_before) > 0 and code in close_df.columns:
                c_before = close_df[code].dropna()
                c_before = c_before[c_before.index <= date]
                if len(c_before) > 0 and len(money_before) > 0:
                    last_c = c_before.iloc[-1]
                    last_m = money_before.iloc[-1]
                    if last_c > 0 and last_m > 0:
                        mkt_cap = np.log(last_m * 1000 / last_c)
        specific_vol = np.nan
        if not np.isnan(beta) and not np.isnan(vol) and len(mkt_aligned) > 0:
            mkt_vol = np.nanstd(mkt_aligned.values)
            specific_vol = np.sqrt(max(0, vol**2 - beta**2 * mkt_vol**2))
        ind_code = industry_map.get(code, -1)
        if ind_code == -1:
            ind_code = len(set(industry_map.values()))
        rows.append({
            "code": code, "beta": beta, "industry_code": ind_code,
            "log_mkt_cap": mkt_cap, "momentum_20d": momentum,
            "volatility_60d": vol, "specific_vol": specific_vol,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for i, row in result.iterrows():
        code = row["code"]
        if code in money_df.columns and code in close_df.columns:
            money = money_df[code].dropna()
            money_before = money[money.index <= date]
            close_c = close_df[code].dropna()
            c_before = close_c[close_c.index <= date]
            if len(money_before) >= 20 and len(c_before) >= 20:
                adv = money_before.tail(20).mean() * 1000
                avg_c = c_before.tail(20).mean()
                if avg_c > 0 and adv > 0 and row["log_mkt_cap"] is not None and not np.isnan(row["log_mkt_cap"]):
                    mkt_cap_val = np.exp(row["log_mkt_cap"])
                    result.at[i, "liquidity_ratio"] = adv / max(mkt_cap_val, 1e6)
    result["liquidity_ratio"] = result.get("liquidity_ratio", np.nan)
    return result

def calibrate_alpha(alpha_scores, specific_vol, ic_estimate=0.05):
    """Calibrate raw alpha scores to expected active return in bps/day.

    Formula: expected_active_return = IC * score_z * sigma_specific
    """
    import numpy as np
    alpha = np.asarray(alpha_scores, dtype=float)
    spec = np.asarray(specific_vol, dtype=float)
    mask = ~np.isnan(alpha)
    if mask.sum() < 2:
        return alpha * 0, alpha * 0
    mean = np.nanmean(alpha)
    std = np.nanmax([np.nanstd(alpha), 1e-10])
    score_z = np.where(mask, (alpha - mean) / std, 0.0)
    med_spec = np.nanmedian(spec) if np.isfinite(np.nanmedian(spec)) else 0.02
    expected_active = ic_estimate * score_z * np.where(~np.isnan(spec), spec, med_spec)
    expected_bps = expected_active * 10000
    return expected_bps, score_z

def breadth_diagnostics(mean_ic, realized_ir, n_stocks, n_days, avg_correlation=0.3):
    """Compute breadth diagnostics from the Fundamental Law (IR = IC * sqrt(BR))."""
    raw_br = n_stocks * n_days
    k = 3.0
    effective_br = raw_br / max(1 + k * avg_correlation * n_stocks**0.5, 1.0)
    predicted_ir = mean_ic * np.sqrt(effective_br)
    implied_br = (realized_ir / max(mean_ic, 1e-6)) ** 2 if mean_ic > 0 else 0
    return {
        'raw_br': int(raw_br), 'effective_br': int(effective_br),
        'predicted_ir': float(predicted_ir), 'implied_br': int(implied_br),
        'mean_ic': float(mean_ic), 'realized_ir': float(realized_ir),
    }
