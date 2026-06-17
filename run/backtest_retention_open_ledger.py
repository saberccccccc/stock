"""Open-price share-ledger backtest for saved retention alpha ranks.

This combines the next-open execution timing from open-to-open tests with the
cash/share ledger constraints from the small-account execution backtest.
"""

import argparse
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

from backtest.presets import PRESETS, apply_preset_to_namespace, explicit_cli_dests, get_preset
from backtest.open_ledger import limit_new_names, load_alpha_rows, parse_float_list
from backtest.reports import calc_extended_metrics, calc_metrics
from backtest.stress import STRESSES, get_stress
from core.research_protocol import (
    assert_alpha_rows_within_forward,
    assert_alpha_rows_within_research,
)
from run.backtest_retention_execution_constraints import (
    build_desired_target,
    recompute_adv,
    save_stage_breakdown,
    weights_from_selected,
)
from run.backtest_retention_open_execution import load_ohlc_money, open_limit_trade_mask
from run.backtest_temporal_retention import compute_market_multiplier, load_index_returns


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Open-price share ledger from alpha JSONL")
    parser.add_argument("--alpha-jsonl", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--stress", choices=sorted(STRESSES), default=None)
    parser.add_argument("--target-fracs", default="0.006")
    parser.add_argument("--hold-fracs", default="0.10")
    parser.add_argument("--portfolio-values", default="500000,1000000")
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
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--adv-participation-cap", type=float, default=0.05)
    parser.add_argument("--min-adv-cny", type=float, default=3000000.0)
    parser.add_argument("--money-scale", type=float, default=1000.0)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--min-commission-cny", type=float, default=5.0)
    parser.add_argument("--rebalance-band", type=float, default=0.20)
    parser.add_argument("--max-new-names", type=int, default=0)
    parser.add_argument("--risk-target-frac", type=float, default=None)
    parser.add_argument("--risk-target-market-mult-below", type=float, default=1.0)
    parser.add_argument("--use-row-target-frac", action="store_true")
    parser.add_argument("--use-row-market-mult", action="store_true")
    parser.add_argument("--exit-hold-frac", type=float, default=None)
    parser.add_argument("--switch-gap-frac", type=float, default=0.0)
    parser.add_argument("--execution-lag", type=int, default=0)
    parser.add_argument("--max-data-date", default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--allow-forward", action="store_true")
    args = parser.parse_args(argv)
    explicit = explicit_cli_dests(sys.argv[1:] if argv is None else argv)
    if args.preset or args.stress:
        preset = get_preset(args.preset or "official_open_price_share_ledger")
        if args.stress:
            preset = get_stress(args.stress).apply(preset)
        args = apply_preset_to_namespace(args, preset, explicit_dests=explicit)
    return args


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
    codes = list(open_df.columns)
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


def run_open_ledger(alpha_rows, open_df, close_df, adv_df, target_frac, hold_frac, args, idx_close, idx_daily):
    codes = list(open_df.columns)
    code2idx = {c: i for i, c in enumerate(codes)}
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
    returns_df = pd.DataFrame({"date": active_dates, "return": returns_active})
    return row, returns_df, diag_df


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_rows = load_alpha_rows(args.alpha_jsonl)
    if args.allow_forward:
        assert_alpha_rows_within_forward(alpha_rows, context="open ledger forward test")
    else:
        assert_alpha_rows_within_research(alpha_rows, context="open ledger research test")

    all_codes = sorted({code for row in alpha_rows for code in row["codes"]})
    print(f"alpha_days={len(alpha_rows)} codes={len(all_codes)}", flush=True)
    open_df, close_df, money_df = load_ohlc_money(args.data_dir, all_codes, args.money_scale, args.progress_every)
    if args.max_data_date:
        max_data_date = pd.Timestamp(args.max_data_date)
        open_df = open_df.loc[open_df.index <= max_data_date]
        close_df = close_df.loc[close_df.index <= max_data_date]
        money_df = money_df.loc[money_df.index <= max_data_date]
    adv_df = recompute_adv(money_df, args.adv_window)
    idx_close, idx_daily = load_index_returns(args.data_dir, args.index_file, close_df.index)

    summary_rows = []
    returns_by_tag = {}
    for portfolio_value in parse_float_list(args.portfolio_values):
        args.portfolio_value = portfolio_value
        for target_frac in parse_float_list(args.target_fracs):
            for hold_frac in parse_float_list(args.hold_fracs):
                if hold_frac < target_frac:
                    continue
                row, returns_df, diag_df = run_open_ledger(
                    alpha_rows, open_df, close_df, adv_df, target_frac, hold_frac, args, idx_close, idx_daily
                )
                if not row:
                    continue
                row.update({
                    "alpha_jsonl": args.alpha_jsonl,
                    "portfolio_value": portfolio_value,
                    "adv_participation_cap": args.adv_participation_cap,
                    "min_adv_cny": args.min_adv_cny,
                    "limit_threshold": args.limit_threshold,
                })
                summary_rows.append(row)
                tag = (
                    f"pv{int(portfolio_value / 1e4):04d}w_"
                    f"target{int(round(target_frac * 1000)):03d}_hold{int(round(hold_frac * 1000)):03d}"
                )
                returns_df.to_csv(out_dir / f"returns_{tag}.csv", index=False)
                diag_df.to_csv(out_dir / f"diagnostics_{tag}.csv", index=False)
                returns_by_tag[tag] = returns_df
                print(
                    f"pv={portfolio_value/1e4:.0f}w target={target_frac:.3f} hold={hold_frac:.3f} "
                    f"ann={row['ann']:.2f}% sharpe={row['sharpe']:.3f} "
                    f"mdd={row['mdd']*100:.2f}% exec_to={row['avg_executed_turnover']:.3f} "
                    f"unfilled={row['avg_unfilled_turnover']:.3f}",
                    flush=True,
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "open_ledger_summary.csv", index=False)
    save_stage_breakdown(out_dir, returns_by_tag)
    print(f"Saved open ledger summary: {out_dir / 'open_ledger_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
