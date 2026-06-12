"""Retention backtest from saved alpha ranks with execution constraints.

This script intentionally reads precomputed alpha JSONL files so stricter
execution assumptions can be tested without recomputing model scores.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backtest.reports import calc_extended_metrics, calc_metrics
from core.research_protocol import (
    assert_alpha_rows_within_forward,
    assert_alpha_rows_within_research,
)
from run.backtest_temporal_retention import compute_market_multiplier, load_index_returns


def parse_args():
    parser = argparse.ArgumentParser(description="Constrained retention backtest from alpha JSONL")
    parser.add_argument("--alpha-jsonl", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-fracs", default="0.03")
    parser.add_argument("--hold-fracs", default="0.40")
    parser.add_argument("--weight-mode", default="equal", choices=["equal"])
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--market-timing-mode", default="legacy", choices=["none", "legacy", "dynamic"])
    parser.add_argument("--market-min-mult", type=float, default=0.20)
    parser.add_argument("--market-max-mult", type=float, default=1.00)
    parser.add_argument("--legacy-bear-mult", type=float, default=0.70)
    parser.add_argument("--legacy-crash-mult", type=float, default=0.30)
    parser.add_argument("--commission-rate", type=float, default=0.0001)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--index-file", default="hs300_index.csv")
    parser.add_argument("--portfolio-value", type=float, default=1000000.0)
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--adv-participation-cap", type=float, default=0.05)
    parser.add_argument("--min-adv-cny", type=float, default=3000000.0)
    parser.add_argument("--money-scale", type=float, default=1000.0)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--min-commission-cny", type=float, default=5.0)
    parser.add_argument("--execution-lag", type=int, default=0, help="Extra trading-day delay after the next tradable day")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--allow-forward", action="store_true")
    return parser.parse_args()


def load_alpha_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["date"] = pd.Timestamp(row["date"])
            rows.append(row)
    rows.sort(key=lambda r: r["date"])
    return rows


def load_close_money(data_dir, codes, money_scale, progress_every):
    close_series = {}
    money_series = {}
    for i, code in enumerate(codes, start=1):
        path = Path(data_dir) / f"{code}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["trade_date", "close", "money"])
            df.columns = df.columns.str.strip().str.lower()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date").sort_index()
            close_series[code] = df["close"].astype(float).replace([np.inf, -np.inf], np.nan)
            money_series[code] = (df["money"].astype(float) * float(money_scale)).replace([np.inf, -np.inf], np.nan)
        except Exception:
            continue
        if progress_every > 0 and i % progress_every == 0:
            print(f"loaded market data {i}/{len(codes)}", flush=True)

    all_dates = pd.DatetimeIndex(sorted(set().union(*[s.index for s in close_series.values()])))
    close = pd.DataFrame({c: s.reindex(all_dates) for c, s in close_series.items()}, index=all_dates)
    money = pd.DataFrame({c: s.reindex(all_dates) for c, s in money_series.items()}, index=all_dates)
    return close, money


def recompute_adv(money, adv_window):
    min_periods = max(3, int(adv_window) // 4)
    return money.rolling(int(adv_window), min_periods=min_periods).mean().shift(1)


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
    w = np.zeros(n_codes, dtype=np.float64)
    if not selected:
        return w
    ew = min(max_weight, 1.0 / len(selected))
    for code in selected:
        idx = code2idx.get(code)
        if idx is not None:
            w[idx] = ew
    gross = np.sum(np.abs(w))
    if gross > 1e-12:
        w = w / gross * float(gross_weight)
    return w


def limit_trade_mask(close_df, day_pos, codes, code2idx, threshold):
    close_today = close_df.iloc[day_pos]
    close_prev = close_df.iloc[day_pos - 1] if day_pos > 0 else close_today
    prev = close_prev.to_numpy(dtype=np.float64)
    today = close_today.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = today / prev - 1.0
    ret[~np.isfinite(ret)] = np.nan
    buy_block = np.zeros(len(codes), dtype=bool)
    sell_block = np.zeros(len(codes), dtype=bool)
    if threshold > 0:
        buy_block = ret >= float(threshold)
        sell_block = ret <= -float(threshold)
    return buy_block, sell_block


def apply_execution_constraints(
    desired_weights,
    current_shares,
    cash,
    equity,
    close_df,
    adv_df,
    day_pos,
    args,
):
    codes = list(close_df.columns)
    prices = close_df.iloc[day_pos].to_numpy(dtype=np.float64)
    safe_prices = np.where(np.isfinite(prices) & (prices > 0), prices, 0.0)
    adv_row = adv_df.iloc[day_pos].to_numpy(dtype=np.float64)
    buy_block, sell_block = limit_trade_mask(close_df, day_pos, codes, None, args.limit_threshold)
    lot_size = max(int(getattr(args, "lot_size", 1)), 1)
    desired_shares = np.zeros_like(current_shares, dtype=np.float64)
    valid_price = np.isfinite(prices) & (prices > 0)
    desired_shares[valid_price] = (
        np.floor(
            desired_weights[valid_price] * float(equity)
            / prices[valid_price]
            / lot_size
        )
        * lot_size
    )
    share_delta = desired_shares - current_shares
    executed_shares = np.zeros_like(share_delta)
    blocked_buy = blocked_sell = adv_blocked = capped = lot_blocked = 0
    missing_adv = 0

    # Sells are processed first so their proceeds can fund buys on the same close.
    order = np.concatenate((np.where(share_delta < 0)[0], np.where(share_delta > 0)[0]))
    for i in order:
        raw_shares = share_delta[i]
        if abs(raw_shares) < 1:
            continue
        price = prices[i]
        if not np.isfinite(price) or price <= 0:
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
        if max_trade_shares < abs(raw_shares):
            capped += 1
            candidate_shares = np.sign(raw_shares) * max_trade_shares
        else:
            candidate_shares = raw_shares

        candidate_shares = np.sign(candidate_shares) * (
            np.floor(abs(candidate_shares) / lot_size) * lot_size
        )
        if abs(candidate_shares) < lot_size:
            lot_blocked += 1
            continue

        trade_value = abs(candidate_shares) * price
        commission = max(
            trade_value * float(args.commission_rate),
            float(getattr(args, "min_commission_cny", 0.0)),
        )
        stamp_tax = trade_value * float(args.stamp_tax_rate) if candidate_shares < 0 else 0.0
        slippage = trade_value * float(args.slippage_rate)
        fees = commission + stamp_tax + slippage

        if candidate_shares > 0:
            affordable = np.floor(
                max(cash - float(getattr(args, "min_commission_cny", 0.0)), 0.0)
                / (price * (1.0 + float(args.commission_rate) + float(args.slippage_rate)))
                / lot_size
            ) * lot_size
            if affordable < candidate_shares:
                candidate_shares = affordable
                if candidate_shares < lot_size:
                    lot_blocked += 1
                    continue
                trade_value = candidate_shares * price
                commission = max(
                    trade_value * float(args.commission_rate),
                    float(getattr(args, "min_commission_cny", 0.0)),
                )
                slippage = trade_value * float(args.slippage_rate)
                stamp_tax = 0.0
                fees = commission + slippage
            cash -= trade_value + fees
        else:
            candidate_shares = -min(abs(candidate_shares), current_shares[i])
            if abs(candidate_shares) < lot_size:
                lot_blocked += 1
                continue
            trade_value = abs(candidate_shares) * price
            commission = max(
                trade_value * float(args.commission_rate),
                float(getattr(args, "min_commission_cny", 0.0)),
            )
            stamp_tax = trade_value * float(args.stamp_tax_rate)
            slippage = trade_value * float(args.slippage_rate)
            fees = commission + stamp_tax + slippage
            cash += trade_value - fees
        executed_shares[i] = candidate_shares

    new_shares = current_shares + executed_shares
    executed_values = executed_shares * safe_prices
    desired_values = share_delta * safe_prices
    total_commission = 0.0
    total_stamp_tax = 0.0
    total_slippage = 0.0
    for shares, price in zip(executed_shares, prices):
        if abs(shares) < 1 or not np.isfinite(price):
            continue
        value = abs(shares) * price
        total_commission += max(
            value * float(args.commission_rate),
            float(getattr(args, "min_commission_cny", 0.0)),
        )
        total_stamp_tax += value * float(args.stamp_tax_rate) if shares < 0 else 0.0
        total_slippage += value * float(args.slippage_rate)
    total_cost = total_commission + total_stamp_tax + total_slippage
    info = {
        "blocked_buy": blocked_buy,
        "blocked_sell": blocked_sell,
        "adv_blocked": adv_blocked,
        "missing_adv": missing_adv,
        "capped": capped,
        "lot_blocked": lot_blocked,
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


def run_constrained(alpha_rows, close_df, adv_df, target_frac, hold_frac, args, idx_close, idx_daily):
    codes = list(close_df.columns)
    code2idx = {c: i for i, c in enumerate(codes)}
    all_dates = close_df.index
    n_codes, t_total = len(codes), len(all_dates)
    close_mat = close_df.to_numpy(dtype=np.float64).T
    valuation_mat = close_df.ffill().to_numpy(dtype=np.float64, copy=True)
    valuation_mat[~np.isfinite(valuation_mat)] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_daily = close_mat[:, 1:] / close_mat[:, :-1] - 1.0
    ret_daily[~np.isfinite(ret_daily)] = 0.0

    row_by_day = {}
    for row in alpha_rows:
        pos = all_dates.searchsorted(row["date"], side="right") + max(int(getattr(args, "execution_lag", 0)), 0)
        if 0 < pos < t_total:
            row_by_day[int(pos)] = row
    entry_days = sorted(row_by_day)
    if not entry_days:
        return {}, np.asarray([], dtype=np.float64), pd.DataFrame()

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
        legacy_bear_mult=getattr(args, "legacy_bear_mult", 0.7),
        legacy_crash_mult=getattr(args, "legacy_crash_mult", 0.3),
    )

    for day in range(t_total):
        marked_prices = valuation_mat[day]
        equity_before_trade = float(cash + np.dot(current_shares, marked_prices))
        row = row_by_day.get(day)
        if row is not None:
            selected, kept, target_n, hold_n, _ = build_desired_target(row, current_selected, target_frac, hold_frac)
            market_mult = 1.0
            if args.market_timing_mode != "none":
                market_mult = compute_market_multiplier(
                    idx_close,
                    idx_daily,
                    ret_daily,
                    max(day - 1, 0),
                    market_args.market_timing_mode,
                    market_args.market_min_mult,
                    market_args.market_max_mult,
                    market_args.legacy_bear_mult,
                    market_args.legacy_crash_mult,
                )
            desired = weights_from_selected(selected, code2idx, n_codes, market_mult, args.max_weight)
            new_shares, cash, executed_shares, exec_info = apply_execution_constraints(
                desired,
                current_shares,
                cash,
                equity_before_trade,
                close_df,
                adv_df,
                day,
                args,
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
                "target_n": int(target_n),
                "hold_n": int(hold_n),
                "kept_n": int(len(kept)),
                "selected_n": int(len(live_codes)),
                "desired_selected_n": int(len(selected)),
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
    first_entry = entry_days[0]
    last_return_day = min(entry_days[-1] + 1, t_total - 1)
    returns_active = returns[max(first_entry - 1, 0):last_return_day]
    active_dates = all_dates[1:][max(first_entry - 1, 0):max(first_entry - 1, 0) + len(returns_active)]

    closed_ages.extend(holding_ages.values())
    avg_holding_days = float(np.mean(closed_ages)) if closed_ages else 0.0
    diag_df = pd.DataFrame(diag_rows)
    ann, sharpe, mdd = calc_metrics(returns_active)
    ext = calc_extended_metrics(returns_active)
    row = {
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
        "avg_holding_days": avg_holding_days,
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
        "capped": int(diag_df["capped"].sum()) if "capped" in diag_df else 0,
        "lot_blocked": int(diag_df["lot_blocked"].sum()) if "lot_blocked" in diag_df else 0,
        "execution_lag": int(getattr(args, "execution_lag", 0)),
        "lot_size": int(getattr(args, "lot_size", 1)),
        "min_commission_cny": float(getattr(args, "min_commission_cny", 0.0)),
    }
    returns_df = pd.DataFrame({"date": active_dates, "return": returns_active})
    return row, returns_df, diag_df


def save_stage_breakdown(out_dir, returns_by_tag):
    yearly_rows = []
    monthly_rows = []
    for tag, returns_df in returns_by_tag.items():
        if returns_df.empty:
            continue
        df = returns_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.to_period("M").astype(str)
        for year, g in df.groupby("year"):
            ann, sharpe, mdd = calc_metrics(g["return"].to_numpy(float))
            yearly_rows.append({
                "tag": tag,
                "period": str(year),
                "days": len(g),
                "ann": ann,
                "sharpe": sharpe,
                "mdd": mdd,
                "sum_return": float(g["return"].sum()),
            })
        ann, sharpe, mdd = calc_metrics(df["return"].to_numpy(float))
        yearly_rows.append({
            "tag": tag,
            "period": "all",
            "days": len(df),
            "ann": ann,
            "sharpe": sharpe,
            "mdd": mdd,
            "sum_return": float(df["return"].sum()),
        })
        for month, g in df.groupby("month"):
            monthly_rows.append({
                "tag": tag,
                "month": month,
                "days": len(g),
                "sum_return": float(g["return"].sum()),
                "mean_return": float(g["return"].mean()),
            })
    if yearly_rows:
        pd.DataFrame(yearly_rows).to_csv(out_dir / "yearly_summary.csv", index=False)
    if monthly_rows:
        pd.DataFrame(monthly_rows).to_csv(out_dir / "monthly_summary.csv", index=False)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_rows = load_alpha_rows(args.alpha_jsonl)
    if args.allow_forward:
        assert_alpha_rows_within_forward(alpha_rows, context="execution-constrained forward test")
    else:
        assert_alpha_rows_within_research(alpha_rows, context="execution-constrained backtest")
    all_codes = sorted({code for row in alpha_rows for code in row["codes"]})
    print(f"alpha_days={len(alpha_rows)} codes={len(all_codes)}", flush=True)
    close, money = load_close_money(args.data_dir, all_codes, args.money_scale, args.progress_every)
    adv = recompute_adv(money, args.adv_window)
    idx_close, idx_daily = load_index_returns(args.data_dir, args.index_file, close.index)

    target_fracs = [float(x.strip()) for x in args.target_fracs.split(",") if x.strip()]
    hold_fracs = [float(x.strip()) for x in args.hold_fracs.split(",") if x.strip()]
    summary_rows = []
    returns_by_tag = {}
    for target_frac in target_fracs:
        for hold_frac in hold_fracs:
            if hold_frac < target_frac:
                continue
            row, returns_df, diag_df = run_constrained(
                alpha_rows, close, adv, target_frac, hold_frac, args, idx_close, idx_daily
            )
            row.update({
                "alpha_jsonl": args.alpha_jsonl,
                "portfolio_value": args.portfolio_value,
                "adv_participation_cap": args.adv_participation_cap,
                "min_adv_cny": args.min_adv_cny,
                "limit_threshold": args.limit_threshold,
                "execution_lag": args.execution_lag,
                "lot_size": args.lot_size,
                "min_commission_cny": args.min_commission_cny,
            })
            summary_rows.append(row)
            tag = f"target{int(round(target_frac * 1000)):03d}_hold{int(round(hold_frac * 1000)):03d}"
            returns_df.to_csv(out_dir / f"returns_{tag}.csv", index=False)
            diag_df.to_csv(out_dir / f"diagnostics_{tag}.csv", index=False)
            returns_by_tag[tag] = returns_df
            print(
                f"target={target_frac:.3f} hold={hold_frac:.3f} ann={row['ann']:.2f}% "
                f"sharpe={row['sharpe']:.3f} mdd={row['mdd'] * 100:.2f}% "
                f"exec_turnover={row['avg_executed_turnover']:.3f} "
                f"unfilled={row['avg_unfilled_turnover']:.3f} names={row['avg_names']:.1f}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "execution_constrained_summary.csv", index=False)
    save_stage_breakdown(out_dir, returns_by_tag)
    md = [
        "# Execution-constrained retention backtest",
        "",
        f"- alpha_jsonl: `{args.alpha_jsonl}`",
        f"- portfolio_value: `{args.portfolio_value:,.0f}`",
        f"- ADV cap: `{args.adv_participation_cap:.2%}`",
        f"- min ADV CNY: `{args.min_adv_cny:,.0f}`",
        f"- limit threshold: `{args.limit_threshold:.2%}`",
        f"- execution lag: `{args.execution_lag}` trading days",
        f"- costs: commission={args.commission_rate}, stamp_tax={args.stamp_tax_rate}, slippage={args.slippage_rate}",
        "",
        "| target | hold | ann | Sharpe | mdd | exec turnover | unfilled turnover | avg names | capped | adv blocked | buy block | sell block |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['target_frac']:.3f} | {row['hold_frac']:.3f} | {row['ann']:.2f}% | "
            f"{row['sharpe']:.3f} | {row['mdd'] * 100:.2f}% | {row['avg_executed_turnover']:.3f} | "
            f"{row['avg_unfilled_turnover']:.3f} | {row['avg_names']:.1f} | {row['capped']} | "
            f"{row['adv_blocked']} | {row['blocked_buy']} | {row['blocked_sell']} |"
        )
    (out_dir / "execution_constrained_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved constrained summary: {out_dir / 'execution_constrained_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
