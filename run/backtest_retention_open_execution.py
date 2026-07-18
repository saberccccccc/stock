"""Open-price execution stress test for saved retention alpha ranks."""

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

from backtest.reports import calc_extended_metrics, calc_metrics
from alpha.io import load_alpha_rows as load_shared_alpha_rows
from core.research_protocol import (
    assert_alpha_rows_within_forward,
    assert_alpha_rows_within_research,
    resolve_market_data_end_date,
)
from run.backtest_temporal_daily_top import explicit_cost
from run.backtest_temporal_retention import compute_market_multiplier, load_index_returns
from run.backtest_retention_execution_constraints import (
    build_desired_target,
    weights_from_selected,
    recompute_adv,
    save_stage_breakdown,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Retention backtest with next-open execution")
    parser.add_argument("--alpha-jsonl", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-fracs", default="0.03")
    parser.add_argument("--hold-fracs", default="0.30,0.40,0.50")
    parser.add_argument("--return-mode", default="open_to_open", choices=["open_to_open", "open_to_close"])
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--market-timing-mode", default="legacy", choices=["none", "legacy", "dynamic"])
    parser.add_argument("--market-min-mult", type=float, default=0.20)
    parser.add_argument("--market-max-mult", type=float, default=1.00)
    parser.add_argument("--commission-rate", type=float, default=0.0001)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--index-file", default="hs300_index.csv")
    parser.add_argument("--portfolio-value", type=float, default=100000000.0)
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--adv-participation-cap", type=float, default=0.05)
    parser.add_argument("--min-adv-cny", type=float, default=20000000.0)
    parser.add_argument("--money-scale", type=float, default=1000.0)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--execution-lag", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-data-date", default=None)
    parser.add_argument("--allow-forward", action="store_true")
    return parser.parse_args()


def load_alpha_rows(path):
    return load_shared_alpha_rows(path, timestamp_dates=True)


def load_ohlc_money(data_dir, codes, money_scale, progress_every):
    open_series = {}
    close_series = {}
    money_series = {}
    for i, code in enumerate(codes, start=1):
        path = Path(data_dir) / f"{code}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["trade_date", "open", "close", "money"])
            df.columns = df.columns.str.strip().str.lower()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date").sort_index()
            open_series[code] = df["open"].astype(float).replace([np.inf, -np.inf], np.nan)
            close_series[code] = df["close"].astype(float).replace([np.inf, -np.inf], np.nan)
            money_series[code] = (df["money"].astype(float) * float(money_scale)).replace([np.inf, -np.inf], np.nan)
        except Exception:
            continue
        if progress_every > 0 and i % progress_every == 0:
            print(f"loaded OHLC data {i}/{len(codes)}", flush=True)
    all_dates = pd.DatetimeIndex(sorted(set().union(*[s.index for s in open_series.values()])))
    open_df = pd.DataFrame({c: s.reindex(all_dates) for c, s in open_series.items()}, index=all_dates)
    close_df = pd.DataFrame({c: s.reindex(all_dates) for c, s in close_series.items()}, index=all_dates)
    money_df = pd.DataFrame({c: s.reindex(all_dates) for c, s in money_series.items()}, index=all_dates)
    return open_df, close_df, money_df


def open_limit_trade_mask(open_df, close_df, day_pos, threshold):
    open_today = open_df.iloc[day_pos].to_numpy(dtype=np.float64)
    close_prev = close_df.iloc[day_pos - 1].to_numpy(dtype=np.float64) if day_pos > 0 else open_today
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = open_today / close_prev - 1.0
    gap[~np.isfinite(gap)] = np.nan
    buy_block = gap >= float(threshold)
    sell_block = gap <= -float(threshold)
    return buy_block, sell_block


def apply_open_execution_constraints(desired, current, open_df, close_df, adv_df, day_pos, args):
    adv_row = adv_df.iloc[day_pos].to_numpy(dtype=np.float64)
    open_row = open_df.iloc[day_pos].to_numpy(dtype=np.float64)
    buy_block, sell_block = open_limit_trade_mask(open_df, close_df, day_pos, args.limit_threshold)
    delta = desired - current
    executed = np.zeros_like(delta)
    blocked_buy = blocked_sell = adv_blocked = capped = missing_adv = no_open = 0
    for i, raw_delta in enumerate(delta):
        if abs(raw_delta) <= 1e-12:
            continue
        if not np.isfinite(open_row[i]) or open_row[i] <= 0:
            no_open += 1
            continue
        adv_cny = adv_row[i]
        if not np.isfinite(adv_cny) or adv_cny <= 0:
            missing_adv += 1
            continue
        if adv_cny < args.min_adv_cny:
            adv_blocked += 1
            continue
        if raw_delta > 0 and buy_block[i]:
            blocked_buy += 1
            continue
        if raw_delta < 0 and sell_block[i]:
            blocked_sell += 1
            continue
        max_abs_delta = float(args.adv_participation_cap) * adv_cny / max(float(args.portfolio_value), 1.0)
        if max_abs_delta < abs(raw_delta):
            capped += 1
            executed[i] = np.sign(raw_delta) * max_abs_delta
        else:
            executed[i] = raw_delta
    new_current = current + executed
    gross = np.sum(np.abs(new_current))
    if gross > 1.0:
        new_current = new_current / gross
    info = {
        "blocked_buy": blocked_buy,
        "blocked_sell": blocked_sell,
        "adv_blocked": adv_blocked,
        "missing_adv": missing_adv,
        "no_open": no_open,
        "capped": capped,
        "desired_turnover": float(np.sum(np.abs(delta))),
        "executed_turnover": float(np.sum(np.abs(executed))),
        "unfilled_turnover": float(np.sum(np.abs(delta - executed))),
    }
    return new_current, executed, info


def run_open_execution(alpha_rows, open_df, close_df, adv_df, target_frac, hold_frac, args, idx_close, idx_daily):
    codes = list(open_df.columns)
    code2idx = {c: i for i, c in enumerate(codes)}
    all_dates = open_df.index
    n_codes, t_total = len(codes), len(all_dates)
    open_mat = open_df.to_numpy(dtype=np.float64).T
    close_mat = close_df.to_numpy(dtype=np.float64).T

    if args.return_mode == "open_to_open":
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_mat = open_mat[:, 1:] / open_mat[:, :-1] - 1.0
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_mat = close_mat / open_mat - 1.0
        ret_mat = ret_mat[:, 1:]
    ret_mat[~np.isfinite(ret_mat)] = 0.0
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

    current = np.zeros(n_codes, dtype=np.float64)
    current_selected = []
    weights = np.zeros((t_total, n_codes), dtype=np.float32)
    daily_costs = np.zeros(t_total, dtype=np.float64)
    diag_rows = []
    holding_ages = {}
    closed_ages = []
    market_args = SimpleNamespace(
        market_timing_mode=args.market_timing_mode,
        market_min_mult=args.market_min_mult,
        market_max_mult=args.market_max_mult,
    )

    for day in range(t_total):
        row = row_by_day.get(day)
        if row is not None:
            selected, kept, target_n, hold_n, _ = build_desired_target(row, current_selected, target_frac, hold_frac)
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
                )
            desired = weights_from_selected(selected, code2idx, n_codes, market_mult, args.max_weight)
            new_current, executed, exec_info = apply_open_execution_constraints(
                desired, current, open_df, close_df, adv_df, day, args
            )
            cost = explicit_cost(executed, args.commission_rate, args.stamp_tax_rate, args.slippage_rate)
            daily_costs[day] = cost["cost"]
            live_idx = np.where(np.abs(new_current) > 1e-12)[0]
            live_codes = [codes[i] for i in live_idx]
            prev_set = set(current_selected)
            live_set = set(live_codes)
            for code in prev_set - live_set:
                closed_ages.append(holding_ages.get(code, 1))
                holding_ages.pop(code, None)
            for code in live_codes:
                holding_ages[code] = holding_ages.get(code, 0) + 1
            current_selected = live_codes
            current = new_current
            diag_rows.append({
                "day": int(day),
                "date": str(all_dates[day]),
                "target_n": int(target_n),
                "hold_n": int(hold_n),
                "kept_n": int(len(kept)),
                "selected_n": int(len(live_codes)),
                "desired_selected_n": int(len(selected)),
                "gross_weight": float(np.sum(np.abs(current))),
                "market_mult": float(market_mult),
                "avg_live_age": float(np.mean(list(holding_ages.values()))) if holding_ages else 0.0,
                **cost,
                **exec_info,
            })
        weights[day] = current.astype(np.float32)

    returns = []
    for day in range(1, t_total):
        w = weights[day].astype(np.float64)
        returns.append(float(np.dot(w, ret_mat[:, day - 1]) - daily_costs[day]))
    returns = np.asarray(returns, dtype=np.float64)
    first_entry = entry_days[0]
    last_return_day = min(entry_days[-1] + 1, t_total - 1)
    start_idx = max(first_entry - 1, 0)
    returns_active = returns[start_idx:last_return_day]
    active_dates = all_dates[1:][start_idx:start_idx + len(returns_active)]
    closed_ages.extend(holding_ages.values())
    diag_df = pd.DataFrame(diag_rows)
    ann, sharpe, mdd = calc_metrics(returns_active)
    ext = calc_extended_metrics(returns_active)
    row = {
        "target_frac": target_frac,
        "hold_frac": hold_frac,
        "return_mode": args.return_mode,
        "execution_lag": int(args.execution_lag),
        "n_return_days": int(len(returns_active)),
        "ann": float(ann),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "calmar": float(ext.get("calmar", 0.0)),
        "sortino": float(ext.get("sortino", 0.0)),
        "win_rate": float(ext.get("win_rate", 0.0)),
        "avg_daily_return": float(np.mean(returns_active)) if len(returns_active) else 0.0,
        "vol": float(np.std(returns_active) * np.sqrt(252)) if len(returns_active) else 0.0,
        "avg_executed_turnover": float(diag_df["executed_turnover"].mean()) if "executed_turnover" in diag_df else 0.0,
        "avg_unfilled_turnover": float(diag_df["unfilled_turnover"].mean()) if "unfilled_turnover" in diag_df else 0.0,
        "avg_holding_days": float(np.mean(closed_ages)) if closed_ages else 0.0,
        "avg_names": float(diag_df["selected_n"].mean()) if "selected_n" in diag_df else 0.0,
        "avg_gross_weight": float(diag_df["gross_weight"].mean()) if "gross_weight" in diag_df else 0.0,
        "market_timing_mode": args.market_timing_mode,
        "avg_market_mult": float(diag_df["market_mult"].mean()) if "market_mult" in diag_df else 1.0,
        "total_cost": float(daily_costs.sum()),
        "blocked_buy": int(diag_df["blocked_buy"].sum()) if "blocked_buy" in diag_df else 0,
        "blocked_sell": int(diag_df["blocked_sell"].sum()) if "blocked_sell" in diag_df else 0,
        "adv_blocked": int(diag_df["adv_blocked"].sum()) if "adv_blocked" in diag_df else 0,
        "missing_adv": int(diag_df["missing_adv"].sum()) if "missing_adv" in diag_df else 0,
        "no_open": int(diag_df["no_open"].sum()) if "no_open" in diag_df else 0,
        "capped": int(diag_df["capped"].sum()) if "capped" in diag_df else 0,
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
            yearly_rows.append({"tag": tag, "period": str(year), "days": len(g), "ann": ann, "sharpe": sharpe, "mdd": mdd, "sum_return": float(g["return"].sum())})
        ann, sharpe, mdd = calc_metrics(df["return"].to_numpy(float))
        yearly_rows.append({"tag": tag, "period": "all", "days": len(df), "ann": ann, "sharpe": sharpe, "mdd": mdd, "sum_return": float(df["return"].sum())})
        for month, g in df.groupby("month"):
            monthly_rows.append({"tag": tag, "month": month, "days": len(g), "sum_return": float(g["return"].sum()), "mean_return": float(g["return"].mean())})
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
        assert_alpha_rows_within_forward(alpha_rows, context="open-execution forward test")
    else:
        assert_alpha_rows_within_research(alpha_rows, context="open-execution research test")
    all_codes = sorted({code for row in alpha_rows for code in row["codes"]})
    print(f"alpha_days={len(alpha_rows)} codes={len(all_codes)} return_mode={args.return_mode}", flush=True)
    open_df, close_df, money_df = load_ohlc_money(args.data_dir, all_codes, args.money_scale, args.progress_every)
    max_data_date = resolve_market_data_end_date(
        args.max_data_date,
        allow_forward=args.allow_forward,
    )
    if max_data_date is not None:
        open_df = open_df.loc[open_df.index <= max_data_date]
        close_df = close_df.loc[close_df.index <= max_data_date]
        money_df = money_df.loc[money_df.index <= max_data_date]
    adv_df = recompute_adv(money_df, args.adv_window)
    idx_close, idx_daily = load_index_returns(args.data_dir, args.index_file, close_df.index)
    target_fracs = [float(x.strip()) for x in args.target_fracs.split(",") if x.strip()]
    hold_fracs = [float(x.strip()) for x in args.hold_fracs.split(",") if x.strip()]
    summary_rows = []
    returns_by_tag = {}
    for target_frac in target_fracs:
        for hold_frac in hold_fracs:
            if hold_frac < target_frac:
                continue
            row, returns_df, diag_df = run_open_execution(alpha_rows, open_df, close_df, adv_df, target_frac, hold_frac, args, idx_close, idx_daily)
            if not row:
                print(
                    f"target={target_frac:.3f} hold={hold_frac:.3f} skipped: no executable overlap",
                    flush=True,
                )
                continue
            row.update({
                "alpha_jsonl": args.alpha_jsonl,
                "portfolio_value": args.portfolio_value,
                "adv_participation_cap": args.adv_participation_cap,
                "min_adv_cny": args.min_adv_cny,
                "limit_threshold": args.limit_threshold,
            })
            summary_rows.append(row)
            tag = f"target{int(round(target_frac * 1000)):03d}_hold{int(round(hold_frac * 1000)):03d}"
            returns_df.to_csv(out_dir / f"returns_{tag}.csv", index=False)
            diag_df.to_csv(out_dir / f"diagnostics_{tag}.csv", index=False)
            returns_by_tag[tag] = returns_df
            print(
                f"target={target_frac:.3f} hold={hold_frac:.3f} ann={row['ann']:.2f}% "
                f"sharpe={row['sharpe']:.3f} mdd={row['mdd'] * 100:.2f}% "
                f"exec_turnover={row['avg_executed_turnover']:.3f} unfilled={row['avg_unfilled_turnover']:.3f}",
                flush=True,
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "open_execution_summary.csv", index=False)
    save_stage_breakdown(out_dir, returns_by_tag)
    print(f"Saved open execution summary: {out_dir / 'open_execution_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
