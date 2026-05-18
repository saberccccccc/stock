# layered_engine.py - true rolling sleeve holdings backtest
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.config import DataConfig
from core.train_utils import get_regime_dim
from backtest.engine import (
    build_simple_weights,
    build_universe_matrix,
    calc_metrics,
    compute_adv_weight_cap,
    compute_risk_model,
    detect_regime,
    ewma_beta,
    execute_order_with_impact,
    optimize_mean_variance,
    optimize_projected_simple_weights,
    optimize_with_risk_budget,
    risk_budget_allocation,
)


def resolve_layer_scale(layer_scale, holding_days, rebalance_freq):
    if layer_scale == "auto":
        return min(1.0, float(rebalance_freq) / float(holding_days))
    value = float(layer_scale)
    if value <= 0:
        raise ValueError(f"layer_scale必须为正数: {layer_scale}")
    return value


def aggregate_layers(active_layers, n_codes, day):
    total = np.zeros(n_codes, dtype=np.float64)
    count = 0
    for layer in active_layers:
        if layer["hold_start"] <= day <= layer["hold_end"]:
            total[layer["idx_full"]] += layer["weights"]
            count += 1
    return total, count


def prune_expired_layers(active_layers, day):
    return [layer for layer in active_layers if layer["hold_end"] >= day]


def write_layered_daily_weights(daily_total_weights, active_layers, n_codes, start_day, end_day):
    for day in range(start_day, end_day + 1):
        daily_total_weights[day], _ = aggregate_layers(active_layers, n_codes, day)


def build_index_returns(config, all_dates):
    idx_path = os.path.join(config.data_dir, "hs300_index.csv")
    if os.path.exists(idx_path):
        idx_df = pd.read_csv(idx_path)
        date_col = "trade_date" if "trade_date" in idx_df.columns else "date"
        idx_df[date_col] = pd.to_datetime(idx_df[date_col])
        idx_df.set_index(date_col, inplace=True)
        idx_close = idx_df["close"].reindex(all_dates)
        idx_daily = idx_close.pct_change().fillna(0)
    else:
        idx_close = pd.Series(np.nan, index=all_dates)
        idx_daily = pd.Series(0.0, index=all_dates)
    return idx_close, idx_daily


def build_date_index(val_samples, all_dates):
    date2idx = {}
    for sample in val_samples:
        dt = sample["date"]
        pos = all_dates.searchsorted(dt, side="right") - 1
        date2idx[dt] = max(pos, 0)
    return date2idx


def build_daily_returns(price_mat):
    ret_daily = np.full((price_mat.shape[0], price_mat.shape[1] - 1), np.nan)
    for i in range(price_mat.shape[0]):
        p = price_mat[i]
        ret_daily[i] = p[1:] / p[:-1] - 1
    ret_daily[~np.isfinite(ret_daily)] = 0.0
    return ret_daily


def simple_long_market_multiplier(portfolio_mode, col_cur, idx_close):
    if portfolio_mode != "simple_long":
        return 1.0
    market_mult = 1.0
    if col_cur >= 60 and np.isfinite(idx_close.iloc[col_cur]):
        idx_ma60 = idx_close.iloc[col_cur - 60:col_cur].mean()
        idx_cur = idx_close.iloc[col_cur]
        if idx_cur < idx_ma60:
            market_mult = 0.7
        if col_cur >= 120:
            idx_ret_6m = idx_close.iloc[col_cur] / idx_close.iloc[col_cur - 120] - 1
            if idx_ret_6m < -0.10:
                market_mult = min(market_mult, 0.3)
    return market_mult


def make_target_weights(alpha, sample, valid, rets, R_mat, idx_daily, hist_start, hist_end,
                        price_hist, vol_mat, idx_full, prev_w, config, regime, market_mult,
                        portfolio_mode, top_frac, max_weight, adv_mode, adv_limit_ratio,
                        portfolio_value, ewma_hl, lambda_t, lambda_b, target_vol,
                        optimizer_base_mode, optimizer_exposure_control,
                        optimizer_beta_limit, optimizer_dollar_neutral,
                        mvo_risk_aversion, mvo_lr, mvo_n_iter, diag):
    if portfolio_mode in ("optimizer", "optimizer_projected", "optimizer_mvo"):
        idx_ret_hist = idx_daily.iloc[hist_start + 1:hist_end + 1].values
        beta_vec = ewma_beta(rets, idx_ret_hist, ewma_hl)

        regime_dim = get_regime_dim(config)
        B_style = np.hstack([
            sample["risk"][valid, :3],
            sample["risk"][valid, regime_dim:],
        ])
        F_cov, D_diag = compute_risk_model(B_style, R_mat)

        max_w_arr, adv_diag = compute_adv_weight_cap(
            max_weight, adv_mode, adv_limit_ratio, vol_mat, idx_full,
            hist_start, hist_end, price_hist, portfolio_value,
        )
        if adv_diag:
            diag["avg_adv_weight_cap"].append(adv_diag["avg_adv_weight_cap"])

        if portfolio_mode == "optimizer":
            risk_budget = risk_budget_allocation(alpha, D_diag, target_vol)
            w_target, opt_diag = optimize_with_risk_budget(
                alpha, B_style, beta_vec, prev_w,
                F_cov, D_diag, risk_budget,
                lambda_t, lambda_b,
                max_weight=max_w_arr, max_leverage=1.0,
                return_diagnostics=True,
            )
            diag["opt_converged"].append(opt_diag["converged"])
            diag["opt_iterations"].append(opt_diag["iterations"])
            diag["risk_budget_match"].append(opt_diag["risk_budget_match"])
            diag["beta_exposure"].append(opt_diag["beta_exposure"])
            return w_target

        if portfolio_mode == "optimizer_projected":
            w_target, project_diag = optimize_projected_simple_weights(
                alpha, base_mode=optimizer_base_mode, beta=beta_vec, prev_w=prev_w,
                max_weight=max_w_arr, max_leverage=1.0, top_frac=top_frac,
                regime=regime, market_mult=market_mult, lambda_t=lambda_t,
                exposure_control=optimizer_exposure_control,
                beta_limit=optimizer_beta_limit,
                dollar_neutral=optimizer_dollar_neutral,
                return_diagnostics=True,
            )
            for key, value in project_diag.items():
                diag[key].append(value)
            return w_target

        w_target, mvo_diag = optimize_mean_variance(
            alpha, B_style, beta_vec, prev_w, F_cov, D_diag,
            max_weight=max_w_arr, max_leverage=1.0,
            base_mode=optimizer_base_mode, top_frac=top_frac,
            regime=regime, market_mult=market_mult,
            lambda_risk=mvo_risk_aversion, lambda_t=lambda_t,
            lambda_b=lambda_b,
            exposure_control=optimizer_exposure_control,
            beta_limit=optimizer_beta_limit,
            dollar_neutral=optimizer_dollar_neutral,
            lr0=mvo_lr, n_iter=mvo_n_iter,
            return_diagnostics=True,
        )
        for key, value in mvo_diag.items():
            diag[key].append(value)
        return w_target

    return build_simple_weights(
        alpha, portfolio_mode, top_frac,
        max_leverage=1.0, regime=regime, market_mult=market_mult,
    )


def trim_active_returns(port_ret, neu_ret, all_dates, daily_total_weights, daily_costs):
    return_dates = all_dates[1:]
    return_costs = daily_costs[1:]
    active = np.array([np.sum(np.abs(w)) > 0 for w in daily_total_weights[1:]], dtype=bool)
    if active.any():
        first_active = int(np.argmax(active))
        last_active = len(active) - int(np.argmax(active[::-1]))
        return (
            port_ret[first_active:last_active],
            np.asarray(neu_ret)[first_active:last_active],
            return_dates[first_active:last_active],
            return_costs[first_active:last_active],
        )
    return port_ret, np.asarray(neu_ret), return_dates, return_costs


def run_backtest_layered_production(
    predictor,
    val_samples,
    price_dict,
    vol_dict,
    holding_days=5,
    rebalance_freq=1,
    hist_window=60,
    ewma_hl=20,
    adv_limit_ratio=0.02,
    adv_mode="execution",
    portfolio_mode="optimizer_projected",
    top_frac=0.10,
    max_weight=0.05,
    lambda_t=0.05,
    lambda_b=0.2,
    target_vol=0.15,
    impact_coeff=0.1,
    config=None,
    portfolio_value=1e8,
    optimizer_base_mode="simple_ls",
    optimizer_exposure_control="beta",
    optimizer_beta_limit=0.05,
    optimizer_dollar_neutral=True,
    mvo_risk_aversion=1.0,
    mvo_lr=0.02,
    mvo_n_iter=200,
    layer_scale="auto",
):
    if holding_days <= 0:
        raise ValueError(f"holding_days必须为正数: {holding_days}")
    if rebalance_freq <= 0:
        raise ValueError(f"rebalance_freq必须为正数: {rebalance_freq}")

    config = config or DataConfig()
    layer_scale_value = resolve_layer_scale(layer_scale, holding_days, rebalance_freq)

    print(f"预测器 {predictor.name}")
    print(f"Layered holdings mode: true rolling sleeves")
    print(f"组合模式: {portfolio_mode} | top_frac={top_frac:.2%}")
    print(f"holding_days={holding_days} | rebalance_freq={rebalance_freq} | layer_scale={layer_scale_value:.4f}")
    print(f"ADV模式: {adv_mode}")
    print("return metric: next_close_to_next_close")

    all_codes = sorted(set(code for sample in val_samples for code in sample["codes"]))
    price_mat, vol_mat, all_dates, code2idx = build_universe_matrix(price_dict, vol_dict, all_codes)
    T_total = len(all_dates)
    idx_close, idx_daily = build_index_returns(config, all_dates)
    date2idx = build_date_index(val_samples, all_dates)
    ret_daily = build_daily_returns(price_mat)

    active_layers = []
    daily_total_weights = [np.zeros(len(all_codes), dtype=np.float64) for _ in range(T_total)]
    daily_costs = np.zeros(T_total)
    last_signal_idx = -1
    total_cost = 0.0
    diag = defaultdict(list)
    diag_counts = defaultdict(int)

    for t_idx, sample in enumerate(tqdm(val_samples, desc=f"分层回测-{predictor.name}")):
        dt = sample["date"]
        col_cur = date2idx[dt]
        if col_cur < hist_window + holding_days:
            continue
        if t_idx - last_signal_idx < rebalance_freq:
            continue

        diag_counts["rebalance_attempts"] += 1
        before_prune = len(active_layers)
        active_layers = prune_expired_layers(active_layers, col_cur)
        expired_count = before_prune - len(active_layers)
        diag_counts["layers_expired"] += expired_count

        codes = sample["codes"]
        idx_full = [code2idx[code] for code in codes]
        n_before_valid = len(codes)
        hist_start = col_cur - hist_window
        hist_end = col_cur - 1
        if hist_start < 0:
            continue

        price_hist = price_mat[idx_full, hist_start:hist_end + 1]
        valid = np.sum(~np.isnan(price_hist), axis=1) >= 0.7 * hist_window
        if not np.any(valid):
            continue

        price_hist = price_hist[valid]
        codes = [codes[i] for i in range(n_before_valid) if valid[i]]
        idx_full = [idx_full[i] for i in range(n_before_valid) if valid[i]]
        n_codes = len(codes)

        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.log(price_hist[:, 1:] / price_hist[:, :-1])
            rets[~np.isfinite(rets)] = 0
        R_mat = rets.T

        regime = detect_regime(sample)
        alpha = predictor.predict_alpha(sample, valid, regime)
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        if alpha.shape[0] != n_codes:
            raise ValueError(f"预测长度不匹配 alpha={alpha.shape[0]}, N={n_codes}")
        diag["valid_names"].append(n_codes)
        diag["alpha_std"].append(float(np.std(alpha)))

        market_mult = simple_long_market_multiplier(portfolio_mode, col_cur, idx_close)
        entry_day = col_cur + 1
        exit_day = min(entry_day + holding_days, T_total - 1)
        hold_start = entry_day + 1
        hold_end = exit_day
        if entry_day >= T_total or hold_start > hold_end:
            continue

        prev_total, active_count = aggregate_layers(active_layers, len(all_codes), entry_day)
        prev_w = prev_total[idx_full]
        w_target = make_target_weights(
            alpha, sample, valid, rets, R_mat, idx_daily, hist_start, hist_end,
            price_hist, vol_mat, idx_full, prev_w, config, regime, market_mult,
            portfolio_mode, top_frac, max_weight, adv_mode, adv_limit_ratio,
            portfolio_value, ewma_hl, lambda_t, lambda_b, target_vol,
            optimizer_base_mode, optimizer_exposure_control,
            optimizer_beta_limit, optimizer_dollar_neutral,
            mvo_risk_aversion, mvo_lr, mvo_n_iter, diag,
        )
        if not np.all(np.isfinite(w_target)):
            continue

        w_layer_target = w_target * layer_scale_value
        target_total = prev_w + w_layer_target
        diag["target_leverage"].append(float(np.sum(np.abs(w_layer_target))))

        price_next = price_mat[idx_full, entry_day]
        vol_next = vol_mat[idx_full, entry_day] if vol_mat is not None else np.ones(n_codes) * 1e9
        tradable = np.isfinite(price_next) & (vol_next > 0)
        target_total_tradable = np.where(tradable, target_total, prev_w)

        exec_adv_ratio = adv_limit_ratio if adv_mode in ("execution", "both") else 1e9
        w_total_filled, impact_cost, fill_ratio, trade_exec = execute_order_with_impact(
            target_total_tradable, prev_w, price_next, vol_next,
            adv_ratio=exec_adv_ratio, impact_coeff=impact_coeff,
            portfolio_value=portfolio_value, regime=regime,
        )
        w_layer_filled = w_total_filled - prev_w
        w_layer_filled[np.abs(w_layer_filled) < 1e-8] = 0.0
        if np.sum(np.abs(w_layer_filled)) <= 1e-12:
            continue

        new_layer = {
            "signal_date": dt,
            "entry_day": entry_day,
            "hold_start": hold_start,
            "hold_end": hold_end,
            "idx_full": np.asarray(idx_full, dtype=np.int64),
            "weights": w_layer_filled.astype(np.float64),
            "codes": codes,
        }
        active_layers.append(new_layer)
        write_layered_daily_weights(daily_total_weights, active_layers, len(all_codes), hold_start, hold_end)

        total_cost += impact_cost
        daily_costs[entry_day] += impact_cost
        diag_counts["successful_rebalances"] += 1
        diag_counts["layers_created"] += 1
        diag["filled_leverage"].append(float(np.sum(np.abs(w_layer_filled))))
        diag["turnover"].append(float(np.sum(np.abs(trade_exec))))
        active_trade = np.abs(target_total_tradable - prev_w) > 1e-8
        if np.any(active_trade):
            diag["fill_ratio"].append(float(np.mean(fill_ratio[active_trade])))
        diag["untradable_ratio"].append(float(1.0 - np.mean(tradable)))
        diag["holding_days_written"].append(max(0, hold_end - hold_start + 1))
        diag["layer_holding_days"].append(float(holding_days))
        diag["layer_rebalance_freq"].append(float(rebalance_freq))
        diag["layer_scale"].append(float(layer_scale_value))
        diag["new_layer_gross"].append(float(np.sum(np.abs(w_layer_filled))))
        diag["aggregate_gross_before"].append(float(np.sum(np.abs(prev_total))))
        after_total, overlap_count = aggregate_layers(active_layers, len(all_codes), hold_start)
        diag["active_layers"].append(float(overlap_count))
        diag["aggregate_gross_after"].append(float(np.sum(np.abs(after_total))))
        diag["aggregate_net_after"].append(float(np.sum(after_total)))
        diag["expired_layers_count"].append(float(expired_count))
        diag["layer_overlap_count"].append(float(overlap_count))
        diag["layer_turnover"].append(float(np.sum(np.abs(trade_exec))))
        if np.any(active_trade):
            diag["layer_fill_ratio"].append(float(np.mean(fill_ratio[active_trade])))
        diag["layer_untradable_ratio"].append(float(1.0 - np.mean(tradable)))
        diag["layer_days_written"].append(max(0, hold_end - hold_start + 1))

        last_signal_idx = t_idx

    daily_port_ret = []
    for day in range(1, T_total):
        w_day = daily_total_weights[day].copy()
        stock_ret = ret_daily[:, day - 1]
        valid_ret = np.isfinite(stock_ret)
        w_day[~valid_ret] = 0.0
        stock_ret = np.nan_to_num(stock_ret, nan=0.0)
        lev = np.sum(np.abs(w_day))
        if lev > 1.0:
            w_day = w_day / lev
        daily_port_ret.append(np.dot(w_day, stock_ret) - daily_costs[day])

    port_ret = np.nan_to_num(np.asarray(daily_port_ret), nan=0.0)
    idx_ret_arr = np.nan_to_num(idx_daily.iloc[1:len(daily_port_ret) + 1].values, nan=0.0)

    neu_ret = []
    beta_estimates = []
    for i in range(len(port_ret)):
        start = max(0, i - 60)
        if i - start < 20:
            beta = 0.0
        else:
            cov_ = np.cov(port_ret[start:i], idx_ret_arr[start:i])[0, 1]
            var_ = np.var(idx_ret_arr[start:i])
            beta = cov_ / (var_ + 1e-8) if var_ > 1e-8 else 0.0
            beta_estimates.append(beta)
        neu_ret.append(port_ret[i] - beta * idx_ret_arr[i])

    leverage_values = [np.sum(np.abs(w)) for w in daily_total_weights if np.sum(np.abs(w)) > 0]
    avg_leverage = np.mean(leverage_values) if leverage_values else 0.0
    nonzero_days = len(leverage_values)

    def diag_mean(key):
        return float(np.mean(diag[key])) if diag[key] else 0.0

    print("\n========== 分层持仓回测诊断 ==========")
    print(f"预测器 {predictor.name} | 组合模式: {portfolio_mode}")
    print(f"rebalance: attempts {diag_counts['rebalance_attempts']} | success {diag_counts['successful_rebalances']}")
    print(f"holding_days={holding_days} | rebalance_freq={rebalance_freq} | layer_scale={layer_scale_value:.4f}")
    print(f"平均有效股票数 {diag_mean('valid_names'):.1f}")
    print(f"Alpha标准差 {diag_mean('alpha_std'):.4f}")
    print(f"平均活跃层数 {diag_mean('active_layers'):.2f} | 最大活跃层数 {(max(diag['active_layers']) if diag['active_layers'] else 0):.0f}")
    print(f"新layer平均gross {diag_mean('new_layer_gross'):.3f} | 聚合后gross {diag_mean('aggregate_gross_after'):.3f}")
    print(f"目标新layer杠杆 {diag_mean('target_leverage'):.3f} | 成交新layer杠杆 {diag_mean('filled_leverage'):.3f}")
    print(f"平均换手/调仓 {diag_mean('turnover'):.3f} | 平均填充率 {diag_mean('fill_ratio'):.3f}")
    print(f"不可交易比例 {diag_mean('untradable_ratio'):.3%}")
    if diag.get("project_alpha_retention"):
        print(f"投影Alpha保留率 {diag_mean('project_alpha_retention'):.3f}")
        print(f"投影seed/final权重相关 {diag_mean('project_weight_corr_seed_final'):.3f}")
    if diag.get("mvo_objective"):
        print(f"MVO目标函数 {diag_mean('mvo_objective'):.4f}")
        print(f"MVO Alpha保留率 {diag_mean('mvo_alpha_retention'):.3f}")
    print(f"非零持仓天数: {nonzero_days} / {T_total}")
    print(f"平均杠杆: {avg_leverage:.3f}")
    print(f"平均估计Beta: {(np.mean(beta_estimates) if beta_estimates else 0.0):.3f}")
    print(f"总冲击成本 {total_cost:.4f}")

    port_ret, neu_ret, return_dates, return_costs = trim_active_returns(
        port_ret, neu_ret, all_dates, daily_total_weights, daily_costs,
    )

    backtest_data = {
        "daily_returns": port_ret,
        "neutral_returns": neu_ret,
        "return_dates": return_dates,
        "return_costs": return_costs,
        "daily_weights": daily_total_weights,
        "daily_costs": daily_costs,
        "dates": all_dates,
        "codes": all_codes,
        "diagnostics": diag,
        "diagnostic_counts": dict(diag_counts),
        "beta_estimates": beta_estimates,
        "leverage_values": leverage_values,
    }
    return port_ret, neu_ret, backtest_data
