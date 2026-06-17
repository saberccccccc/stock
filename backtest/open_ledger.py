"""Reusable helpers for open-price share-ledger backtests."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest.reports import calc_extended_metrics, calc_metrics


def parse_float_list(raw):
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def load_alpha_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["date"] = pd.Timestamp(row["date"])
            rows.append(row)
    rows.sort(key=lambda row: row["date"])
    return rows


def load_ohlc_money(data_dir, codes, money_scale, progress_every):
    open_series = {}
    close_series = {}
    money_series = {}
    for i, code in enumerate(codes, start=1):
        path = Path(data_dir) / f"{code}.csv"
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, usecols=["trade_date", "open", "close", "money"])
            frame.columns = frame.columns.str.strip().str.lower()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            frame = frame.set_index("trade_date").sort_index()
            open_series[code] = frame["open"].astype(float).replace([np.inf, -np.inf], np.nan)
            close_series[code] = frame["close"].astype(float).replace([np.inf, -np.inf], np.nan)
            money_series[code] = (frame["money"].astype(float) * float(money_scale)).replace([np.inf, -np.inf], np.nan)
        except Exception:
            continue
        if progress_every > 0 and i % progress_every == 0:
            print(f"loaded OHLC data {i}/{len(codes)}", flush=True)
    all_dates = pd.DatetimeIndex(sorted(set().union(*[series.index for series in open_series.values()])))
    open_df = pd.DataFrame({code: series.reindex(all_dates) for code, series in open_series.items()}, index=all_dates)
    close_df = pd.DataFrame({code: series.reindex(all_dates) for code, series in close_series.items()}, index=all_dates)
    money_df = pd.DataFrame({code: series.reindex(all_dates) for code, series in money_series.items()}, index=all_dates)
    return open_df, close_df, money_df


def recompute_adv(money, adv_window):
    min_periods = max(3, int(adv_window) // 4)
    return money.rolling(int(adv_window), min_periods=min_periods).mean().shift(1)


def save_stage_breakdown(out_dir, returns_by_tag):
    yearly_rows = []
    monthly_rows = []
    for tag, returns_df in returns_by_tag.items():
        if returns_df.empty:
            continue
        frame = returns_df.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["year"] = frame["date"].dt.year
        frame["month"] = frame["date"].dt.to_period("M").astype(str)
        for year, group in frame.groupby("year"):
            ann, sharpe, mdd = calc_metrics(group["return"].to_numpy(float))
            yearly_rows.append({
                "tag": tag,
                "period": str(year),
                "days": len(group),
                "ann": ann,
                "sharpe": sharpe,
                "mdd": mdd,
                "sum_return": float(group["return"].sum()),
            })
        ann, sharpe, mdd = calc_metrics(frame["return"].to_numpy(float))
        yearly_rows.append({
            "tag": tag,
            "period": "all",
            "days": len(frame),
            "ann": ann,
            "sharpe": sharpe,
            "mdd": mdd,
            "sum_return": float(frame["return"].sum()),
        })
        for month, group in frame.groupby("month"):
            monthly_rows.append({
                "tag": tag,
                "month": month,
                "days": len(group),
                "sum_return": float(group["return"].sum()),
                "mean_return": float(group["return"].mean()),
            })
    if yearly_rows:
        pd.DataFrame(yearly_rows).to_csv(Path(out_dir) / "yearly_summary.csv", index=False)
    if monthly_rows:
        pd.DataFrame(monthly_rows).to_csv(Path(out_dir) / "monthly_summary.csv", index=False)


def build_desired_target(row, current_codes, target_frac, hold_frac):
    codes = list(row["codes"])
    n = len(codes)
    target_n = max(1, int(n * target_frac))
    hold_n = max(target_n, int(n * hold_frac))
    rank_map = {code: i for i, code in enumerate(codes)}

    kept = [code for code in current_codes if rank_map.get(code, n + 1) < hold_n]
    if len(kept) > target_n:
        kept = sorted(kept, key=lambda c: rank_map.get(c, n + 1))[:target_n]

    selected = list(kept)
    selected_set = set(selected)
    for code in codes:
        if len(selected) >= target_n:
            break
        if code not in selected_set:
            selected.append(code)
            selected_set.add(code)
    return selected, kept, target_n, hold_n, rank_map


def weights_from_selected(selected, code2idx, n_codes, gross_weight, max_weight):
    weights = np.zeros(n_codes, dtype=np.float64)
    if not selected:
        return weights
    equal_weight = min(max_weight, 1.0 / len(selected))
    for code in selected:
        idx = code2idx.get(code)
        if idx is not None:
            weights[idx] = equal_weight
    gross = np.sum(np.abs(weights))
    if gross > 1e-12:
        weights = weights / gross * float(gross_weight)
    return weights


def load_index_returns(data_dir, index_file, all_dates):
    path = Path(data_dir) / index_file
    if not path.exists():
        return pd.Series(np.nan, index=all_dates), pd.Series(0.0, index=all_dates)
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    date_col = "trade_date" if "trade_date" in frame.columns else frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.set_index(date_col).sort_index()
    close = frame["close"].astype(float).reindex(all_dates)
    daily = close.pct_change().fillna(0.0)
    return close, daily


def compute_market_multiplier(
    idx_close,
    idx_daily,
    ret_daily,
    col_cur,
    mode,
    min_mult,
    max_mult,
    legacy_bear_mult=0.7,
    legacy_crash_mult=0.3,
):
    if mode in (None, "", "none"):
        return 1.0
    if col_cur < 60 or not np.isfinite(idx_close.iloc[col_cur]):
        return float(max_mult)

    idx_cur = float(idx_close.iloc[col_cur])
    idx_ma60 = float(idx_close.iloc[col_cur - 60:col_cur].mean())
    if mode == "legacy":
        market_mult = float(legacy_bear_mult) if idx_cur < idx_ma60 else 1.0
        if col_cur >= 120 and np.isfinite(idx_close.iloc[col_cur - 120]) and idx_close.iloc[col_cur - 120] > 0:
            idx_ret_6m = idx_cur / float(idx_close.iloc[col_cur - 120]) - 1.0
            if idx_ret_6m < -0.10:
                market_mult = min(market_mult, float(legacy_crash_mult))
        return float(np.clip(market_mult, 0.0, max_mult))

    if mode != "dynamic":
        raise ValueError(f"Unknown market_timing_mode: {mode}")

    ma_score = 1.0 if idx_cur >= idx_ma60 else 0.0
    mom_score = 0.5
    if col_cur >= 20 and np.isfinite(idx_close.iloc[col_cur - 20]) and idx_close.iloc[col_cur - 20] > 0:
        mom20 = idx_cur / float(idx_close.iloc[col_cur - 20]) - 1.0
        mom_score = float(np.clip((mom20 + 0.08) / 0.16, 0.0, 1.0))

    breadth_score = 0.5
    if col_cur >= 20 and ret_daily.shape[1] >= col_cur:
        recent_rets = ret_daily[:, col_cur - 20:col_cur]
        finite = np.isfinite(recent_rets)
        if np.any(finite):
            breadth_score = float(np.nanmean(recent_rets[finite] > 0))

    vol_score = 0.5
    if col_cur >= 20:
        recent_idx_ret = np.asarray(idx_daily.iloc[col_cur - 20:col_cur], dtype=float)
        recent_idx_ret = recent_idx_ret[np.isfinite(recent_idx_ret)]
        if len(recent_idx_ret) > 5:
            ann_vol = float(np.std(recent_idx_ret) * np.sqrt(252))
            vol_score = float(1.0 - np.clip((ann_vol - 0.15) / 0.25, 0.0, 1.0))

    score = 0.4 * ma_score + 0.3 * mom_score + 0.2 * breadth_score + 0.1 * vol_score
    market_mult = min_mult + (max_mult - min_mult) * score
    if col_cur >= 120 and np.isfinite(idx_close.iloc[col_cur - 120]) and idx_close.iloc[col_cur - 120] > 0:
        idx_ret_6m = idx_cur / float(idx_close.iloc[col_cur - 120]) - 1.0
        if idx_ret_6m < -0.10:
            market_mult = min(market_mult, max(min_mult, 0.35))
    return float(np.clip(market_mult, min_mult, max_mult))


def limit_new_names(
    selected,
    kept,
    row,
    max_new_names,
    current_selected,
    target_n,
    exit_hold_frac=None,
    switch_gap_frac=0.0,
):
    if max_new_names <= 0 or not current_selected:
        return selected
    codes = list(row.get("codes", []))
    rank_map = {code: rank for rank, code in enumerate(codes)}
    if exit_hold_frac is not None and exit_hold_frac > 0:
        exit_n = max(int(len(codes) * float(exit_hold_frac)), target_n)
        eligible_current = [
            code for code in current_selected
            if rank_map.get(code, len(codes) + 1) < exit_n
        ]
    else:
        eligible_current = [code for code in current_selected if code in rank_map]
    current_ranked = sorted(
        eligible_current,
        key=lambda code: rank_map[code],
    )
    target_n = max(int(target_n), 1)
    max_new_names = max(int(max_new_names), 0)
    min_old = max(target_n - max_new_names, 0)
    switch_gap = max(int(len(codes) * float(switch_gap_frac)), 0)
    limited = current_ranked[:min(len(current_ranked), min_old)]
    selected_set = set(limited)
    added = 0
    current_set = set(current_selected)
    replacement_old = current_ranked[len(limited):]
    replacement_slot = 0
    for code in codes:
        if len(limited) >= target_n:
            break
        if code in selected_set:
            continue
        if code in current_set:
            continue
        if added >= max_new_names:
            break
        if switch_gap > 0 and replacement_slot < len(replacement_old):
            old_code = replacement_old[replacement_slot]
            if rank_map[code] + switch_gap >= rank_map[old_code]:
                continue
        limited.append(code)
        selected_set.add(code)
        added += 1
        replacement_slot += 1
    for code in current_ranked:
        if len(limited) >= target_n:
            break
        if code not in selected_set:
            limited.append(code)
            selected_set.add(code)
    for code in codes:
        if len(limited) >= target_n:
            break
        if code not in selected_set:
            limited.append(code)
            selected_set.add(code)
    return limited


def open_limit_trade_mask(open_df, close_df, day_pos, threshold):
    open_today = open_df.iloc[day_pos].to_numpy(dtype=np.float64)
    close_prev = close_df.iloc[day_pos - 1].to_numpy(dtype=np.float64) if day_pos > 0 else open_today
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = open_today / close_prev - 1.0
    gap[~np.isfinite(gap)] = np.nan
    buy_block = gap >= float(threshold)
    sell_block = gap <= -float(threshold)
    return buy_block, sell_block


def apply_open_ledger_constraints(
    desired_weights,
    current_shares,
    cash,
    equity,
    open_df,
    close_df,
    adv_df,
    day_pos,
    args,
):
    prices = open_df.iloc[day_pos].to_numpy(dtype=np.float64)
    safe_prices = np.where(np.isfinite(prices) & (prices > 0), prices, 0.0)
    adv_row = adv_df.iloc[day_pos].to_numpy(dtype=np.float64)
    buy_block, sell_block = open_limit_trade_mask(open_df, close_df, day_pos, args.limit_threshold)
    lot_size = max(int(args.lot_size), 1)

    desired_shares = np.zeros_like(current_shares, dtype=np.float64)
    valid_price = np.isfinite(prices) & (prices > 0)
    desired_shares[valid_price] = (
        np.floor(desired_weights[valid_price] * float(equity) / prices[valid_price] / lot_size)
        * lot_size
    )
    share_delta = desired_shares - current_shares
    retained = (current_shares > 0) & (desired_shares > 0)
    has_resize = np.abs(share_delta) >= 1
    within_band = retained & has_resize & (
        np.abs(share_delta) <= max(float(args.rebalance_band), 0.0) * np.maximum(desired_shares, lot_size)
    )
    share_delta[within_band] = 0.0

    executed_shares = np.zeros_like(share_delta)
    blocked_buy = blocked_sell = adv_blocked = capped = lot_blocked = missing_adv = no_open = 0
    order = np.concatenate((np.where(share_delta < 0)[0], np.where(share_delta > 0)[0]))

    for i in order:
        raw_shares = share_delta[i]
        if abs(raw_shares) < 1:
            continue
        price = prices[i]
        if not np.isfinite(price) or price <= 0:
            no_open += 1
            continue
        adv_cny = adv_row[i]
        if not np.isfinite(adv_cny) or adv_cny <= 0:
            missing_adv += 1
            continue
        if adv_cny < args.min_adv_cny:
            adv_blocked += 1
            continue
        if raw_shares > 0 and buy_block[i]:
            blocked_buy += 1
            continue
        if raw_shares < 0 and sell_block[i]:
            blocked_sell += 1
            continue

        max_trade_shares = (
            np.floor(float(args.adv_participation_cap) * adv_cny / price / lot_size)
            * lot_size
        )
        if max_trade_shares < lot_size:
            lot_blocked += 1
            continue
        candidate_shares = raw_shares
        if max_trade_shares < abs(raw_shares):
            capped += 1
            candidate_shares = np.sign(raw_shares) * max_trade_shares
        candidate_shares = np.sign(candidate_shares) * (
            np.floor(abs(candidate_shares) / lot_size) * lot_size
        )
        if abs(candidate_shares) < lot_size:
            lot_blocked += 1
            continue

        trade_value = abs(candidate_shares) * price
        commission = max(trade_value * float(args.commission_rate), float(args.min_commission_cny))
        stamp_tax = trade_value * float(args.stamp_tax_rate) if candidate_shares < 0 else 0.0
        slippage = trade_value * float(args.slippage_rate)
        fees = commission + stamp_tax + slippage

        if candidate_shares > 0:
            affordable = (
                np.floor(
                    max(cash - float(args.min_commission_cny), 0.0)
                    / (price * (1.0 + float(args.commission_rate) + float(args.slippage_rate)))
                    / lot_size
                )
                * lot_size
            )
            if affordable < candidate_shares:
                candidate_shares = affordable
                if candidate_shares < lot_size:
                    lot_blocked += 1
                    continue
                trade_value = candidate_shares * price
                commission = max(trade_value * float(args.commission_rate), float(args.min_commission_cny))
                stamp_tax = 0.0
                slippage = trade_value * float(args.slippage_rate)
                fees = commission + slippage
            cash -= trade_value + fees
        else:
            candidate_shares = -min(abs(candidate_shares), current_shares[i])
            if abs(candidate_shares) < lot_size:
                lot_blocked += 1
                continue
            trade_value = abs(candidate_shares) * price
            commission = max(trade_value * float(args.commission_rate), float(args.min_commission_cny))
            stamp_tax = trade_value * float(args.stamp_tax_rate)
            slippage = trade_value * float(args.slippage_rate)
            fees = commission + stamp_tax + slippage
            cash += trade_value - fees
        executed_shares[i] = candidate_shares

    new_shares = current_shares + executed_shares
    executed_values = executed_shares * safe_prices
    desired_values = share_delta * safe_prices
    total_commission = total_stamp_tax = total_slippage = 0.0
    for shares, price in zip(executed_shares, prices):
        if abs(shares) < 1 or not np.isfinite(price):
            continue
        value = abs(shares) * price
        total_commission += max(value * float(args.commission_rate), float(args.min_commission_cny))
        total_stamp_tax += value * float(args.stamp_tax_rate) if shares < 0 else 0.0
        total_slippage += value * float(args.slippage_rate)
    total_cost = total_commission + total_stamp_tax + total_slippage
    info = {
        "blocked_buy": blocked_buy,
        "blocked_sell": blocked_sell,
        "adv_blocked": adv_blocked,
        "missing_adv": missing_adv,
        "no_open": no_open,
        "capped": capped,
        "lot_blocked": lot_blocked,
        "band_skipped": int(np.count_nonzero(within_band)),
        "turnover": float(np.sum(np.abs(executed_values)) / max(equity, 1.0)),
        "desired_turnover": float(np.sum(np.abs(desired_values)) / max(equity, 1.0)),
        "executed_turnover": float(np.sum(np.abs(executed_values)) / max(equity, 1.0)),
        "unfilled_turnover": float(
            np.sum(np.abs((share_delta - executed_shares) * safe_prices)) / max(equity, 1.0)
        ),
        "cost": float(total_cost / max(equity, 1.0)),
        "commission": float(total_commission / max(equity, 1.0)),
        "stamp_tax": float(total_stamp_tax / max(equity, 1.0)),
        "slippage": float(total_slippage / max(equity, 1.0)),
    }
    return new_shares, cash, executed_shares, info


def summarize_open_ledger_result(
    returns_active,
    diag_df,
    closed_ages,
    target_frac,
    hold_frac,
    args,
):
    ann, sharpe, mdd = calc_metrics(returns_active)
    ext = calc_extended_metrics(returns_active)
    return {
        "target_frac": target_frac,
        "hold_frac": hold_frac,
        "n_return_days": int(len(returns_active)),
        "ann": float(ann),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "calmar": float(ext.get("calmar", 0.0)),
        "sortino": float(ext.get("sortino", 0.0)),
        "win_rate": float(ext.get("win_rate", 0.0)),
        "avg_daily_return": float(np.mean(returns_active)) if len(returns_active) else 0.0,
        "vol": float(np.std(returns_active) * np.sqrt(252)) if len(returns_active) else 0.0,
        "avg_turnover": float(diag_df["turnover"].mean()) if "turnover" in diag_df else 0.0,
        "avg_executed_turnover": float(diag_df["executed_turnover"].mean()) if "executed_turnover" in diag_df else 0.0,
        "avg_unfilled_turnover": float(diag_df["unfilled_turnover"].mean()) if "unfilled_turnover" in diag_df else 0.0,
        "avg_holding_days": float(np.mean(closed_ages)) if closed_ages else 0.0,
        "avg_names": float(diag_df["selected_n"].mean()) if "selected_n" in diag_df else 0.0,
        "avg_gross_weight": float(diag_df["gross_weight"].mean()) if "gross_weight" in diag_df else 0.0,
        "market_timing_mode": args.market_timing_mode,
        "avg_market_mult": float(diag_df["market_mult"].mean()) if "market_mult" in diag_df else 1.0,
        "total_cost": float(diag_df["cost"].sum()) if "cost" in diag_df else 0.0,
        "total_commission": float(diag_df["commission"].sum()) if "commission" in diag_df else 0.0,
        "total_stamp_tax": float(diag_df["stamp_tax"].sum()) if "stamp_tax" in diag_df else 0.0,
        "total_slippage": float(diag_df["slippage"].sum()) if "slippage" in diag_df else 0.0,
        "blocked_buy": int(diag_df["blocked_buy"].sum()) if "blocked_buy" in diag_df else 0,
        "blocked_sell": int(diag_df["blocked_sell"].sum()) if "blocked_sell" in diag_df else 0,
        "adv_blocked": int(diag_df["adv_blocked"].sum()) if "adv_blocked" in diag_df else 0,
        "missing_adv": int(diag_df["missing_adv"].sum()) if "missing_adv" in diag_df else 0,
        "no_open": int(diag_df["no_open"].sum()) if "no_open" in diag_df else 0,
        "capped": int(diag_df["capped"].sum()) if "capped" in diag_df else 0,
        "lot_blocked": int(diag_df["lot_blocked"].sum()) if "lot_blocked" in diag_df else 0,
        "band_skipped": int(diag_df["band_skipped"].sum()) if "band_skipped" in diag_df else 0,
        "execution_lag": int(args.execution_lag),
        "lot_size": int(args.lot_size),
        "min_commission_cny": float(args.min_commission_cny),
        "rebalance_band": float(args.rebalance_band),
        "max_new_names": int(getattr(args, "max_new_names", 0)),
        "risk_target_frac": (
            float(args.risk_target_frac)
            if getattr(args, "risk_target_frac", None) is not None
            else np.nan
        ),
        "risk_target_market_mult_below": float(
            getattr(args, "risk_target_market_mult_below", 1.0)
        ),
        "avg_effective_target_frac": (
            float(diag_df["effective_target_frac"].mean())
            if "effective_target_frac" in diag_df
            else float(target_frac)
        ),
        "exit_hold_frac": float(getattr(args, "exit_hold_frac", 0.0) or 0.0),
        "switch_gap_frac": float(getattr(args, "switch_gap_frac", 0.0) or 0.0),
    }


def run_open_ledger(alpha_rows, open_df, close_df, adv_df, target_frac, hold_frac, args, idx_close, idx_daily):
    codes = list(open_df.columns)
    code2idx = {code: i for i, code in enumerate(codes)}
    all_dates = open_df.index
    n_codes, t_total = len(codes), len(all_dates)
    open_mark = open_df.ffill().to_numpy(dtype=np.float64, copy=True)
    open_mark[~np.isfinite(open_mark)] = 0.0
    close_mat = close_df.to_numpy(dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        close_ret_daily = close_mat[:, 1:] / close_mat[:, :-1] - 1.0
    close_ret_daily[~np.isfinite(close_ret_daily)] = 0.0

    row_by_day = {}
    for row in alpha_rows:
        pos = all_dates.searchsorted(row["date"], side="right") + max(int(args.execution_lag), 0)
        if 0 < pos < t_total:
            row_by_day[int(pos)] = row
    entry_days = sorted(row_by_day)
    if not entry_days:
        return {}, pd.DataFrame(), pd.DataFrame()

    current_shares = np.zeros(n_codes, dtype=np.float64)
    cash = float(args.portfolio_value)
    current_selected = []
    equity_curve = np.full(t_total, np.nan, dtype=np.float64)
    diag_rows = []
    closed_ages = []
    holding_ages = {}
    market_args = SimpleNamespace(
        market_timing_mode=args.market_timing_mode,
        market_min_mult=args.market_min_mult,
        market_max_mult=args.market_max_mult,
        legacy_bear_mult=args.legacy_bear_mult,
        legacy_crash_mult=args.legacy_crash_mult,
    )

    for day in range(t_total):
        marked_prices = open_mark[day]
        equity_before_trade = float(cash + np.dot(current_shares, marked_prices))
        row = row_by_day.get(day)
        if row is not None:
            market_mult = 1.0
            if args.market_timing_mode != "none":
                market_mult = compute_market_multiplier(
                    idx_close,
                    idx_daily,
                    close_ret_daily,
                    max(day - 1, 0),
                    market_args.market_timing_mode,
                    market_args.market_min_mult,
                    market_args.market_max_mult,
                    market_args.legacy_bear_mult,
                    market_args.legacy_crash_mult,
                )
            if getattr(args, "use_row_market_mult", False):
                for transform_key in (
                    "breadth_market_transform",
                    "state_market_transform",
                ):
                    transform = row.get(transform_key)
                    if isinstance(transform, dict) and transform.get("triggered"):
                        row_mult = transform.get("effective_market_mult")
                        if row_mult is not None:
                            market_mult = min(float(market_mult), float(row_mult))
            effective_target_frac = float(target_frac)
            risk_target_frac = getattr(args, "risk_target_frac", None)
            if (
                risk_target_frac is not None
                and market_mult < float(getattr(args, "risk_target_market_mult_below", 1.0))
            ):
                effective_target_frac = min(float(target_frac), float(risk_target_frac))
            if getattr(args, "use_row_target_frac", False):
                for transform_key in (
                    "breadth_target_transform",
                    "state_target_transform",
                ):
                    transform = row.get(transform_key)
                    if isinstance(transform, dict) and transform.get("triggered"):
                        row_target = transform.get("effective_target_frac")
                        if row_target is not None:
                            effective_target_frac = min(
                                float(effective_target_frac),
                                float(row_target),
                            )
            selected, kept, target_n, hold_n, _ = build_desired_target(
                row,
                current_selected,
                effective_target_frac,
                hold_frac,
            )
            selected = limit_new_names(
                selected,
                kept,
                row,
                max(int(getattr(args, "max_new_names", 0)), 0),
                current_selected,
                target_n,
                getattr(args, "exit_hold_frac", None),
                getattr(args, "switch_gap_frac", 0.0),
            )
            desired = weights_from_selected(selected, code2idx, n_codes, market_mult, args.max_weight)
            new_shares, cash, executed_shares, exec_info = apply_open_ledger_constraints(
                desired, current_shares, cash, equity_before_trade, open_df, close_df, adv_df, day, args
            )
            live_idx = np.where(new_shares >= max(int(args.lot_size), 1))[0]
            live_codes = [codes[i] for i in live_idx]
            prev_set = set(current_selected)
            live_set = set(live_codes)
            for code in prev_set - live_set:
                closed_ages.append(holding_ages.get(code, 1))
                holding_ages.pop(code, None)
            for code in live_codes:
                holding_ages[code] = holding_ages.get(code, 0) + 1
            current_selected = live_codes
            current_shares = new_shares
            equity_after_trade = float(cash + np.dot(current_shares, marked_prices))
            invested_value = float(np.dot(current_shares, marked_prices))
            diag_rows.append({
                "day": int(day),
                "date": str(all_dates[day]),
                "effective_target_frac": float(effective_target_frac),
                "target_n": int(target_n),
                "hold_n": int(hold_n),
                "kept_n": int(len(kept)),
                "selected_n": int(len(live_codes)),
                "desired_selected_n": int(len(selected)),
                "max_new_names": int(getattr(args, "max_new_names", 0)),
                "exit_hold_frac": float(getattr(args, "exit_hold_frac", 0.0) or 0.0),
                "switch_gap_frac": float(getattr(args, "switch_gap_frac", 0.0) or 0.0),
                "desired_new_names": int(len(set(selected) - prev_set)),
                "gross_weight": invested_value / max(equity_after_trade, 1.0),
                "cash_cny": float(cash),
                "equity_cny": equity_after_trade,
                "market_mult": float(market_mult),
                "avg_live_age": float(np.mean(list(holding_ages.values()))) if holding_ages else 0.0,
                **exec_info,
            })
        equity_curve[day] = float(cash + np.dot(current_shares, marked_prices))

    with np.errstate(divide="ignore", invalid="ignore"):
        returns = equity_curve[1:] / equity_curve[:-1] - 1.0
    returns[~np.isfinite(returns)] = 0.0
    start_idx = max(entry_days[0] - 1, 0)
    last_return_day = min(entry_days[-1] + 1, t_total - 1)
    returns_active = returns[start_idx:last_return_day]
    active_dates = all_dates[1:][start_idx:start_idx + len(returns_active)]

    closed_ages.extend(holding_ages.values())
    diag_df = pd.DataFrame(diag_rows)
    row = summarize_open_ledger_result(
        returns_active,
        diag_df,
        closed_ages,
        target_frac,
        hold_frac,
        args,
    )
    returns_df = pd.DataFrame({"date": active_dates, "return": returns_active})
    return row, returns_df, diag_df
