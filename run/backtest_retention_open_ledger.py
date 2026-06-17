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
from backtest.open_ledger import (
    apply_open_ledger_constraints,
    build_desired_target,
    limit_new_names,
    load_alpha_rows,
    parse_float_list,
    summarize_open_ledger_result,
    weights_from_selected,
)
from backtest.stress import STRESSES, get_stress
from core.research_protocol import (
    assert_alpha_rows_within_forward,
    assert_alpha_rows_within_research,
)
from run.backtest_retention_execution_constraints import (
    recompute_adv,
    save_stage_breakdown,
)
from run.backtest_retention_open_execution import load_ohlc_money
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
