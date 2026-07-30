"""Open-price share-ledger backtest for saved retention alpha ranks.

This combines the next-open execution timing from open-to-open tests with the
cash/share ledger constraints from the small-account execution backtest.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)
os.chdir(ROOT)

from alpha.io import resolve_alpha_source
from backtest.presets import PRESETS, apply_preset_to_namespace, explicit_cli_dests, get_preset
from backtest.open_ledger import (
    infer_ohlc_load_window,
    load_alpha_rows,
    load_global_risk_features,
    load_index_returns,
    load_execution_market_frames,
    parse_float_list,
    prepare_execution_constraint_masks,
    recompute_adv,
    run_open_ledger,
    save_stage_breakdown,
)
from backtest.market_data_contract import add_execution_market_data_args
from backtest.stress import STRESSES, get_stress

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Open-price share ledger from alpha JSONL")
    alpha_group = parser.add_mutually_exclusive_group(required=True)
    alpha_group.add_argument("--alpha-jsonl")
    alpha_group.add_argument(
        "--alpha-manifest",
        default=None,
        help="Optional JSON manifest mapping portfolio_value to alpha JSONL sources.",
    )
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
    parser.add_argument("--industry-csv", default="data/stock_industry.csv")
    parser.add_argument(
        "--max-industry-weight",
        type=float,
        default=0.0,
        help="Optional selection-level industry budget; 0 disables it.",
    )
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--adv-participation-cap", type=float, default=0.05)
    parser.add_argument("--defensive-tilt", type=float, default=0.0, help="Blend weights toward defense (0=off, 0.3=moderate)")
    parser.add_argument(
        "--defensive-tilt-market-mult-below",
        type=float,
        default=1.0,
        help="Apply defensive tilt only when market_mult is below this threshold; 1.0 keeps legacy/global behavior.",
    )
    parser.add_argument("--min-adv-cny", type=float, default=3000000.0)
    parser.add_argument("--money-scale", type=float, default=1000.0)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--min-commission-cny", type=float, default=5.0)
    parser.add_argument("--rebalance-band", type=float, default=0.20)
    parser.add_argument("--max-new-names", type=int, default=0)
    parser.add_argument(
        "--max-new-names-mode",
        choices=("legacy", "at_most"),
        default="legacy",
    )
    parser.add_argument("--risk-target-frac", type=float, default=None)
    parser.add_argument("--risk-target-market-mult-below", type=float, default=1.0)
    parser.add_argument("--active-drawdown-throttle-lookback", type=int, default=0)
    parser.add_argument(
        "--active-drawdown-throttle-mode",
        default="fixed",
        choices=["fixed", "continuous"],
        help="fixed uses the legacy constant scale; continuous scales smoothly by active drawdown severity.",
    )
    parser.add_argument("--active-drawdown-throttle-trigger", type=float, default=0.0)
    parser.add_argument("--active-drawdown-throttle-scale", type=float, default=1.0)
    parser.add_argument(
        "--active-drawdown-throttle-continuous-width",
        type=float,
        default=0.0,
        help="Active-return distance from trigger to full min scale in continuous mode.",
    )
    parser.add_argument(
        "--active-drawdown-throttle-steps",
        default="",
        help="Optional stair-step trigger:scale pairs, e.g. '-0.03:0.85,-0.06:0.65'.",
    )
    parser.add_argument("--active-drawdown-throttle-cooldown", type=int, default=0)
    parser.add_argument(
        "--active-drawdown-throttle-condition",
        default="active_only",
        choices=[
            "active_only",
            "crowding",
            "crowding_momentum",
            "crowding_momentum_volatility",
        ],
    )
    parser.add_argument("--active-drawdown-throttle-min-top-industry-weight", type=float, default=0.0)
    parser.add_argument("--active-drawdown-throttle-min-industry-hhi", type=float, default=0.0)
    parser.add_argument("--active-drawdown-throttle-max-momentum20", type=float, default=0.0)
    parser.add_argument("--active-drawdown-throttle-min-volatility60", type=float, default=0.0)
    parser.add_argument(
        "--global-risk-features",
        default=None,
        help="Optional A-share-date-aligned global overnight feature file (.parquet/.csv).",
    )
    parser.add_argument(
        "--global-risk-overlay-mode",
        default="none",
        choices=["none", "defensive_pressure", "defensive_pressure_continuous"],
    )
    parser.add_argument("--global-risk-pressure-col", default="global_defensive_pressure")
    parser.add_argument("--global-risk-pressure-threshold", type=float, default=0.04)
    parser.add_argument("--global-risk-pressure-width", type=float, default=0.04)
    parser.add_argument("--global-risk-market-scale", type=float, default=0.8)
    parser.add_argument("--global-risk-target-frac", type=float, default=None)
    parser.add_argument(
        "--state-aware-selection-mode",
        default="none",
        choices=["none", "risk_rank"],
        help="Selection-layer rerank for new candidates under fragile states.",
    )
    parser.add_argument(
        "--state-aware-selection-pressure-col",
        default=None,
        help="Global feature column for selection-layer stress; defaults to --global-risk-pressure-col.",
    )
    parser.add_argument("--state-aware-selection-pressure-threshold", type=float, default=0.035)
    parser.add_argument("--state-aware-selection-pressure-width", type=float, default=0.055)
    parser.add_argument("--state-aware-selection-min-stress", type=float, default=0.0)
    parser.add_argument(
        "--state-aware-selection-rank-penalty",
        type=float,
        default=0.0,
        help="Rank penalty as a fraction of universe size at full stress.",
    )
    parser.add_argument("--state-aware-selection-top-frac", type=float, default=0.006)
    parser.add_argument("--state-aware-selection-crowd-scale", type=float, default=0.10)
    parser.add_argument("--state-aware-selection-momentum-weight", type=float, default=0.40)
    parser.add_argument("--state-aware-selection-beta-weight", type=float, default=0.20)
    parser.add_argument("--state-aware-selection-vol-weight", type=float, default=0.20)
    parser.add_argument("--state-aware-selection-industry-weight", type=float, default=0.20)
    parser.add_argument("--use-row-target-frac", action="store_true")
    parser.add_argument("--use-row-market-mult", action="store_true")
    parser.add_argument("--exit-hold-frac", type=float, default=None)
    parser.add_argument("--switch-gap-frac", type=float, default=0.0)
    parser.add_argument("--execution-lag", type=int, default=0)
    parser.add_argument("--weighting-mode", default="equal", choices=["equal", "score"],
                        help="Portfolio weighting: equal (default) or score (bps-proportional)")
    parser.add_argument("--start-date", default=None, help="Inclusive alpha signal start date")
    parser.add_argument("--end-date", default=None, help="Inclusive alpha signal end date")
    parser.add_argument("--max-data-date", default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--allow-forward", action="store_true", help="Accepted for old commands; open-ledger backtests now allow arbitrary date ranges.")
    parser.add_argument("--ohlc-cache-dir", default="cache/open_ledger_ohlc")
    parser.add_argument("--no-ohlc-cache", action="store_true")
    parser.add_argument("--ohlc-matrix-cache-dir", default="cache/open_ledger_ohlc_matrix")
    parser.add_argument("--no-ohlc-matrix-cache", action="store_true")
    add_execution_market_data_args(parser)
    parser.add_argument("--execution-mask-cache-dir", default="cache/open_ledger_execution_masks")
    parser.add_argument("--no-execution-mask-cache", action="store_true")
    parser.add_argument("--rebuild-ohlc-matrix-cache", action="store_true")
    parser.add_argument("--load-lookback-days", type=int, default=160)
    parser.add_argument(
        "--execution-constraint-mode",
        choices=("proxy", "realistic"),
        default="proxy",
        help="proxy keeps legacy 9.5%% open-gap blocking; realistic uses OHLC volume/money and limit masks.",
    )
    parser.add_argument(
        "--block-intraday-limit-touch",
        action="store_true",
        help="In realistic mode, also block orders when high/low touched the limit intraday.",
    )
    parser.add_argument("--limit-price-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--min-buy-listing-days",
        type=int,
        default=60,
        help="In realistic mode, block new buys until this many trading days after listing; 0 disables.",
    )
    parser.add_argument(
        "--no-limit-first-trading-days",
        type=int,
        default=5,
        help="In realistic mode, disable limit-up/down checks during the first N trading days after listing.",
    )
    args = parser.parse_args(argv)
    explicit = explicit_cli_dests(sys.argv[1:] if argv is None else argv)
    if args.preset or args.stress:
        preset = get_preset(args.preset or "official_open_price_share_ledger")
        if args.stress:
            preset = get_stress(args.stress).apply(preset)
        args = apply_preset_to_namespace(args, preset, explicit_dests=explicit)
    return args


def filter_alpha_rows_by_date(rows, start_date=None, end_date=None):
    if start_date is None and end_date is None:
        return rows
    start = pd.Timestamp(start_date) if start_date is not None else None
    end = pd.Timestamp(end_date) if end_date is not None else None
    if start is not None and end is not None and end < start:
        raise ValueError("--end-date must be >= --start-date")
    filtered = []
    for row in rows:
        dt = pd.Timestamp(row["date"])
        if start is not None and dt < start:
            continue
        if end is not None and dt > end:
            continue
        filtered.append(row)
    return filtered


def summarize_alpha_rows(rows):
    if not rows:
        raise ValueError("alpha rows are empty")
    dates = [pd.Timestamp(row["date"]) for row in rows]
    return {
        "alpha_days": len(rows),
        "code_count": len({code for row in rows for code in row.get("codes", [])}),
        "first_signal": str(min(dates).date()),
        "last_signal": str(max(dates).date()),
    }


def collect_alpha_inputs(args):
    portfolio_values = parse_float_list(args.portfolio_values)
    if not portfolio_values:
        raise ValueError("--portfolio-values must not be empty")

    resolved_sources = {}
    alpha_rows_by_path = {}
    alpha_stats_by_path = {}
    for portfolio_value in portfolio_values:
        resolved = resolve_alpha_source(
            alpha_jsonl=args.alpha_jsonl,
            alpha_manifest=args.alpha_manifest,
            portfolio_value=portfolio_value,
        )
        resolved_sources[float(portfolio_value)] = resolved
        if resolved.alpha_jsonl in alpha_rows_by_path:
            continue
        rows = filter_alpha_rows_by_date(
            load_alpha_rows(resolved.alpha_jsonl),
            args.start_date,
            args.end_date,
        )
        if not rows:
            raise ValueError(
                "no alpha rows remain after date filter "
                f"for source={resolved.alpha_jsonl} start={args.start_date} end={args.end_date}"
            )
        alpha_rows_by_path[resolved.alpha_jsonl] = rows
        alpha_stats_by_path[resolved.alpha_jsonl] = summarize_alpha_rows(rows)
    return portfolio_values, resolved_sources, alpha_rows_by_path, alpha_stats_by_path


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (
        portfolio_values,
        resolved_sources,
        alpha_rows_by_path,
        alpha_stats_by_path,
    ) = collect_alpha_inputs(args)

    combined_rows = [row for rows in alpha_rows_by_path.values() for row in rows]
    all_codes = sorted({
        code
        for rows in alpha_rows_by_path.values()
        for row in rows
        for code in row.get("codes", [])
    })
    first_signal = min(pd.Timestamp(row["date"]) for row in combined_rows).date()
    last_signal = max(pd.Timestamp(row["date"]) for row in combined_rows).date()
    print(
        f"alpha_source_count={len(alpha_rows_by_path)} codes={len(all_codes)} "
        f"signals={first_signal}..{last_signal}",
        flush=True,
    )
    for portfolio_value in portfolio_values:
        resolved_source = resolved_sources[float(portfolio_value)]
        alpha_stats = alpha_stats_by_path[resolved_source.alpha_jsonl]
        source_name = resolved_source.rule_name or ("direct" if resolved_source.mode == "direct" else "unnamed")
        print(
            f"alpha_source pv={portfolio_value/1e4:.0f}w mode={resolved_source.mode} "
            f"rule={source_name} path={resolved_source.alpha_jsonl} "
            f"signals={alpha_stats['first_signal']}..{alpha_stats['last_signal']} "
            f"days={alpha_stats['alpha_days']} codes={alpha_stats['code_count']}",
            flush=True,
        )

    max_data_date = pd.Timestamp(args.max_data_date).normalize() if args.max_data_date else None
    load_start, load_end = infer_ohlc_load_window(
        combined_rows,
        max_data_date=max_data_date,
        execution_lag=args.execution_lag,
        lookback_days=args.load_lookback_days,
    )
    print(
        f"loading OHLC window={load_start.date() if load_start is not None else 'all'}.."
        f"{load_end.date() if load_end is not None else 'all'} backend={args.ohlc_backend} "
        f"matrix_cache="
        f"{'off' if args.no_ohlc_matrix_cache else args.ohlc_matrix_cache_dir} "
        f"window_cache={'off' if args.no_ohlc_cache else args.ohlc_cache_dir}",
        flush=True,
    )
    market_frames = load_execution_market_frames(
        args,
        all_codes,
        start_date=load_start,
        end_date=load_end,
    )
    open_df = market_frames["open"]
    close_df = market_frames["close"]
    money_df = market_frames["money"]
    adv_df = recompute_adv(money_df, args.adv_window)
    execution_masks = prepare_execution_constraint_masks(
        args,
        all_codes,
        open_df,
        close_df,
        money_df,
        load_start=load_start,
        load_end=load_end,
        ohlcv_frames=market_frames if args.ohlc_backend != "legacy" else None,
    )
    if execution_masks is not None:
        print(
            "using realistic execution constraints "
            f"block_intraday_touch={args.block_intraday_limit_touch}",
            flush=True,
        )
    idx_close, idx_daily = load_index_returns(args.data_dir, args.index_file, close_df.index)
    if args.global_risk_overlay_mode != "none" and not args.global_risk_features:
        raise ValueError("--global-risk-features is required when global risk overlay is enabled")
    if args.global_risk_features:
        args._global_risk_frame = load_global_risk_features(
            args.global_risk_features,
            open_df.index,
        )

    summary_rows = []
    returns_by_tag = {}
    for portfolio_value in portfolio_values:
        args.portfolio_value = portfolio_value
        resolved_source = resolved_sources[float(portfolio_value)]
        alpha_rows = alpha_rows_by_path[resolved_source.alpha_jsonl]
        alpha_stats = alpha_stats_by_path[resolved_source.alpha_jsonl]
        for target_frac in parse_float_list(args.target_fracs):
            for hold_frac in parse_float_list(args.hold_fracs):
                if hold_frac < target_frac:
                    continue
                row, returns_df, diag_df = run_open_ledger(
                    alpha_rows,
                    open_df,
                    close_df,
                    adv_df,
                    target_frac,
                    hold_frac,
                    args,
                    idx_close,
                    idx_daily,
                    execution_masks=execution_masks,
                )
                if not row:
                    continue
                backtest_start = (
                    str(pd.Timestamp(returns_df["date"].min()).date())
                    if not returns_df.empty
                    else ""
                )
                backtest_end = (
                    str(pd.Timestamp(returns_df["date"].max()).date())
                    if not returns_df.empty
                    else ""
                )
                row.update({
                    "alpha_jsonl": resolved_source.alpha_jsonl,
                    "alpha_input_path": resolved_source.request_path,
                    "alpha_input_kind": resolved_source.mode,
                    "alpha_manifest": resolved_source.manifest_path,
                    "alpha_source_name": resolved_source.rule_name,
                    "alpha_source_rule_index": (
                        resolved_source.rule_index
                        if resolved_source.rule_index >= 0
                        else float("nan")
                    ),
                    "alpha_source_min_portfolio_value": (
                        resolved_source.min_portfolio_value
                        if resolved_source.min_portfolio_value is not None
                        else float("nan")
                    ),
                    "alpha_source_max_portfolio_value": (
                        resolved_source.max_portfolio_value
                        if resolved_source.max_portfolio_value is not None
                        else float("nan")
                    ),
                    "alpha_start_date": alpha_stats["first_signal"],
                    "alpha_end_date": alpha_stats["last_signal"],
                    "signal_start": alpha_stats["first_signal"],
                    "signal_end": alpha_stats["last_signal"],
                    "backtest_start": backtest_start,
                    "backtest_end": backtest_end,
                    "portfolio_value": portfolio_value,
                    "adv_participation_cap": args.adv_participation_cap,
                    "defensive_tilt": args.defensive_tilt,
                    "defensive_tilt_market_mult_below": args.defensive_tilt_market_mult_below,
                    "active_drawdown_throttle_lookback": args.active_drawdown_throttle_lookback,
                    "active_drawdown_throttle_trigger": args.active_drawdown_throttle_trigger,
                    "active_drawdown_throttle_scale": args.active_drawdown_throttle_scale,
                    "active_drawdown_throttle_steps": args.active_drawdown_throttle_steps,
                    "active_drawdown_throttle_cooldown": args.active_drawdown_throttle_cooldown,
                    "active_drawdown_throttle_condition": args.active_drawdown_throttle_condition,
                    "active_drawdown_throttle_min_top_industry_weight": args.active_drawdown_throttle_min_top_industry_weight,
                    "active_drawdown_throttle_min_industry_hhi": args.active_drawdown_throttle_min_industry_hhi,
                    "active_drawdown_throttle_max_momentum20": args.active_drawdown_throttle_max_momentum20,
                    "active_drawdown_throttle_min_volatility60": args.active_drawdown_throttle_min_volatility60,
                    "min_adv_cny": args.min_adv_cny,
                    "limit_threshold": args.limit_threshold,
                    "execution_constraint_mode": args.execution_constraint_mode,
                    "ohlc_backend": args.ohlc_backend,
                    "market_daily_store_root": (
                        str(Path(args.market_daily_store_root).resolve())
                        if args.ohlc_backend == "monthly"
                        else ""
                    ),
                    "ohlc_monthly_cache_dir": (
                        str(Path(args.ohlc_monthly_cache_dir).resolve())
                        if args.ohlc_backend == "monthly"
                        else ""
                    ),
                    "block_intraday_limit_touch": args.block_intraday_limit_touch,
                    "min_buy_listing_days": args.min_buy_listing_days,
                    "no_limit_first_trading_days": args.no_limit_first_trading_days,
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
