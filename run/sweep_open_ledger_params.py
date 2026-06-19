"""Sweep open-price share-ledger execution parameters on val/test Alpha files."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.open_ledger import (
    load_alpha_rows,
    load_index_returns,
    load_ohlc_money,
    parse_float_list,
    recompute_adv,
    run_open_ledger,
)
from core.research_protocol import assert_alpha_rows_within_research, assert_research_end_date


def parse_int_list(raw):
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep open-ledger parameters")
    parser.add_argument("--val-alpha", required=True)
    parser.add_argument("--test-alpha", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--target-fracs", default="0.005,0.006,0.007")
    parser.add_argument("--hold-fracs", default="0.10,0.12,0.15")
    parser.add_argument("--rebalance-bands", default="0.20,0.25")
    parser.add_argument("--max-replace-names", default="4,5,6")
    parser.add_argument("--exit-hold-fracs", default="")
    parser.add_argument("--switch-gap-fracs", default="0.0")
    parser.add_argument("--portfolio-values", default="500000,1000000")
    parser.add_argument("--market-modes", default="legacy")
    parser.add_argument("--market-min-mults", default="0.20")
    parser.add_argument("--max-weight", type=float, default=0.05)
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
    parser.add_argument("--execution-lag", type=int, default=0)
    parser.add_argument("--max-data-date", default="2026-05-18")
    parser.add_argument("--progress-every", type=int, default=2500)
    return parser.parse_args()


def load_market_bundle(alpha_path, args):
    rows = load_alpha_rows(alpha_path)
    assert_alpha_rows_within_research(rows, context=f"open-ledger sweep alpha {alpha_path}")
    all_codes = sorted({code for row in rows for code in row["codes"]})
    print(f"Loading market data for {alpha_path}: days={len(rows)} codes={len(all_codes)}", flush=True)
    open_df, close_df, money_df = load_ohlc_money(args.data_dir, all_codes, args.money_scale, args.progress_every)
    if args.max_data_date:
        max_data_date = assert_research_end_date(args.max_data_date, context="open-ledger sweep")
        open_df = open_df.loc[open_df.index <= max_data_date]
        close_df = close_df.loc[close_df.index <= max_data_date]
        money_df = money_df.loc[money_df.index <= max_data_date]
    adv_df = recompute_adv(money_df, args.adv_window)
    idx_close, idx_daily = load_index_returns(args.data_dir, args.index_file, close_df.index)
    return rows, open_df, close_df, adv_df, idx_close, idx_daily


def make_run_args(
    args,
    portfolio_value,
    hold_frac,
    rebalance_band,
    max_replace,
    exit_hold_frac,
    switch_gap_frac,
    market_mode,
    market_min_mult,
):
    return SimpleNamespace(
        portfolio_value=float(portfolio_value),
        max_weight=float(args.max_weight),
        market_timing_mode=market_mode,
        market_min_mult=float(market_min_mult),
        market_max_mult=1.0,
        legacy_bear_mult=0.70,
        legacy_crash_mult=0.30,
        commission_rate=float(args.commission_rate),
        stamp_tax_rate=float(args.stamp_tax_rate),
        slippage_rate=float(args.slippage_rate),
        adv_participation_cap=float(args.adv_participation_cap),
        min_adv_cny=float(args.min_adv_cny),
        limit_threshold=float(args.limit_threshold),
        lot_size=int(args.lot_size),
        min_commission_cny=float(args.min_commission_cny),
        rebalance_band=float(rebalance_band),
        max_new_names=int(max_replace),
        exit_hold_frac=None if exit_hold_frac is None else float(exit_hold_frac),
        switch_gap_frac=float(switch_gap_frac),
        execution_lag=int(args.execution_lag),
    )


def summarize_scores(frame):
    rows = []
    keys = [
        "target_frac",
        "hold_frac",
        "rebalance_band",
        "max_replace_names",
        "exit_hold_frac",
        "switch_gap_frac",
        "market_timing_mode",
        "market_min_mult",
        "portfolio_value",
    ]
    for key, group in frame.groupby(keys, dropna=False):
        item = dict(zip(keys, key))
        val = group[group["split"] == "val"]
        test = group[group["split"] == "test"]
        if val.empty or test.empty:
            continue
        item.update({
            "val_ann": float(val["ann"].mean()),
            "val_sharpe": float(val["sharpe"].mean()),
            "val_mdd": float(val["mdd"].mean()),
            "val_turn": float(val["avg_executed_turnover"].mean()),
            "test_ann": float(test["ann"].mean()),
            "test_sharpe": float(test["sharpe"].mean()),
            "test_mdd": float(test["mdd"].mean()),
            "test_turn": float(test["avg_executed_turnover"].mean()),
            "ann_min": float(min(val["ann"].mean(), test["ann"].mean())),
            "sharpe_min": float(min(val["sharpe"].mean(), test["sharpe"].mean())),
            "mdd_max": float(max(val["mdd"].mean(), test["mdd"].mean())),
            "turn_mean": float(group["avg_executed_turnover"].mean()),
            "cost_mean": float(group["total_cost"].mean()),
        })
        item["score"] = (
            item["sharpe_min"]
            + 0.002 * item["ann_min"]
            - 0.8 * item["mdd_max"]
            - 0.15 * item["turn_mean"]
        )
        rows.append(item)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["score", "sharpe_min", "ann_min"], ascending=False)
    return out


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    (out_dir / "sweep_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bundles = {
        "val": load_market_bundle(args.val_alpha, args),
        "test": load_market_bundle(args.test_alpha, args),
    }
    target_fracs = parse_float_list(args.target_fracs)
    hold_fracs = parse_float_list(args.hold_fracs)
    rebalance_bands = parse_float_list(args.rebalance_bands)
    max_replaces = parse_int_list(args.max_replace_names)
    exit_hold_fracs = [None]
    if str(args.exit_hold_fracs).strip():
        exit_hold_fracs = parse_float_list(args.exit_hold_fracs)
    switch_gap_fracs = parse_float_list(args.switch_gap_fracs)
    portfolio_values = parse_float_list(args.portfolio_values)
    market_modes = [x.strip() for x in args.market_modes.split(",") if x.strip()]
    market_min_mults = parse_float_list(args.market_min_mults)

    summary_rows = []
    total = (
        len(bundles)
        * len(target_fracs)
        * len(hold_fracs)
        * len(rebalance_bands)
        * len(max_replaces)
        * len(exit_hold_fracs)
        * len(switch_gap_fracs)
        * len(portfolio_values)
        * len(market_modes)
        * len(market_min_mults)
    )
    done = 0
    for split, bundle in bundles.items():
        rows, open_df, close_df, adv_df, idx_close, idx_daily = bundle
        for target_frac in target_fracs:
            for hold_frac in hold_fracs:
                if hold_frac < target_frac:
                    continue
                for rebalance_band in rebalance_bands:
                    for max_replace in max_replaces:
                        for exit_hold_frac in exit_hold_fracs:
                            for switch_gap_frac in switch_gap_fracs:
                                for market_mode in market_modes:
                                    min_mult_values = market_min_mults if market_mode == "dynamic" else [0.20]
                                    for market_min_mult in min_mult_values:
                                        for portfolio_value in portfolio_values:
                                            run_args = make_run_args(
                                                args,
                                                portfolio_value,
                                                hold_frac,
                                                rebalance_band,
                                                max_replace,
                                                exit_hold_frac,
                                                switch_gap_frac,
                                                market_mode,
                                                market_min_mult,
                                            )
                                            result, _, _ = run_open_ledger(
                                                rows,
                                                open_df,
                                                close_df,
                                                adv_df,
                                                target_frac,
                                                hold_frac,
                                                run_args,
                                                idx_close,
                                                idx_daily,
                                            )
                                            if result:
                                                result.update({
                                                    "split": split,
                                                    "portfolio_value": float(portfolio_value),
                                                    "rebalance_band": float(rebalance_band),
                                                    "max_replace_names": int(max_replace),
                                                    "exit_hold_frac": 0.0 if exit_hold_frac is None else float(exit_hold_frac),
                                                    "switch_gap_frac": float(switch_gap_frac),
                                                    "market_min_mult": float(market_min_mult),
                                                })
                                                summary_rows.append(result)
                                            done += 1
                                            if done % 25 == 0:
                                                print(f"sweep progress {done}/{total}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "sweep_summary.csv", index=False)
    ranked = summarize_scores(summary)
    ranked.to_csv(out_dir / "sweep_ranked.csv", index=False)
    print(f"Saved summary: {out_dir / 'sweep_summary.csv'}", flush=True)
    print(f"Saved ranked: {out_dir / 'sweep_ranked.csv'}", flush=True)
    if not ranked.empty:
        cols = [
            "target_frac",
            "hold_frac",
            "rebalance_band",
            "max_replace_names",
            "portfolio_value",
            "val_ann",
            "val_sharpe",
            "test_ann",
            "test_sharpe",
            "turn_mean",
            "score",
        ]
        print(ranked[cols].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
