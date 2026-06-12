"""Sweep execution constraints for saved retention alpha ranks."""

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

from run.backtest_retention_execution_constraints import (
    load_alpha_rows,
    load_close_money,
    recompute_adv,
    run_constrained,
    save_stage_breakdown,
)
from core.research_protocol import (
    assert_alpha_rows_within_forward,
    assert_alpha_rows_within_research,
)
from run.backtest_temporal_retention import load_index_returns


def parse_float_list(raw):
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep strict execution assumptions")
    parser.add_argument("--alpha-jsonl", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-fracs", default="0.03")
    parser.add_argument("--hold-fracs", default="0.30,0.40")
    parser.add_argument("--portfolio-values", default="500000,1000000")
    parser.add_argument("--adv-participation-caps", default="0.03,0.05,0.10")
    parser.add_argument("--min-adv-cnys", default="1000000,3000000,5000000")
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--money-scale", type=float, default=1000.0)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--min-commission-cny", type=float, default=5.0)
    parser.add_argument("--rebalance-bands", default="0.0")
    parser.add_argument("--execution-lag", type=int, default=0)
    parser.add_argument("--market-timing-mode", default="legacy", choices=["none", "legacy", "dynamic"])
    parser.add_argument("--market-min-mult", type=float, default=0.20)
    parser.add_argument("--market-max-mult", type=float, default=1.00)
    parser.add_argument("--legacy-bear-mult", type=float, default=0.70)
    parser.add_argument("--legacy-crash-mult", type=float, default=0.30)
    parser.add_argument("--commission-rate", type=float, default=0.0001)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--index-file", default="hs300_index.csv")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--allow-forward", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_rows = load_alpha_rows(args.alpha_jsonl)
    if args.allow_forward:
        assert_alpha_rows_within_forward(alpha_rows, context="execution constraint forward sweep")
    else:
        assert_alpha_rows_within_research(alpha_rows, context="execution constraint research sweep")
    all_codes = sorted({code for row in alpha_rows for code in row["codes"]})
    print(f"alpha_days={len(alpha_rows)} codes={len(all_codes)}", flush=True)
    close, money = load_close_money(args.data_dir, all_codes, args.money_scale, args.progress_every)
    adv = recompute_adv(money, args.adv_window)
    idx_close, idx_daily = load_index_returns(args.data_dir, args.index_file, close.index)

    target_fracs = parse_float_list(args.target_fracs)
    hold_fracs = parse_float_list(args.hold_fracs)
    portfolio_values = parse_float_list(args.portfolio_values)
    caps = parse_float_list(args.adv_participation_caps)
    min_advs = parse_float_list(args.min_adv_cnys)
    rebalance_bands = parse_float_list(args.rebalance_bands)

    summary_rows = []
    returns_by_tag = {}
    total = (
        len(target_fracs)
        * len(hold_fracs)
        * len(portfolio_values)
        * len(caps)
        * len(min_advs)
        * len(rebalance_bands)
    )
    done = 0
    for portfolio_value in portfolio_values:
        for cap in caps:
            for min_adv in min_advs:
                for rebalance_band in rebalance_bands:
                    run_args = SimpleNamespace(
                        max_weight=args.max_weight,
                        market_timing_mode=args.market_timing_mode,
                        market_min_mult=args.market_min_mult,
                        market_max_mult=args.market_max_mult,
                        legacy_bear_mult=args.legacy_bear_mult,
                        legacy_crash_mult=args.legacy_crash_mult,
                        commission_rate=args.commission_rate,
                        stamp_tax_rate=args.stamp_tax_rate,
                        slippage_rate=args.slippage_rate,
                        portfolio_value=portfolio_value,
                        adv_participation_cap=cap,
                        min_adv_cny=min_adv,
                        limit_threshold=args.limit_threshold,
                        execution_lag=args.execution_lag,
                        lot_size=args.lot_size,
                        min_commission_cny=args.min_commission_cny,
                        rebalance_band=rebalance_band,
                        allow_forward=args.allow_forward,
                    )
                    for target_frac in target_fracs:
                        for hold_frac in hold_fracs:
                            if hold_frac < target_frac:
                                continue
                            row, returns_df, diag_df = run_constrained(
                                alpha_rows, close, adv, target_frac, hold_frac, run_args, idx_close, idx_daily
                            )
                            row.update({
                                "alpha_jsonl": args.alpha_jsonl,
                                "portfolio_value": portfolio_value,
                                "adv_participation_cap": cap,
                                "min_adv_cny": min_adv,
                                "limit_threshold": args.limit_threshold,
                                "execution_lag": args.execution_lag,
                                "lot_size": args.lot_size,
                                "min_commission_cny": args.min_commission_cny,
                                "rebalance_band": rebalance_band,
                            })
                            summary_rows.append(row)
                            tag = (
                                f"pv{int(portfolio_value / 1e4):04d}w_cap{int(cap * 1000):03d}_"
                                f"minadv{int(min_adv / 1e6):03d}m_"
                                f"band{int(round(rebalance_band * 100)):03d}_"
                                f"target{int(round(target_frac * 1000)):03d}_hold{int(round(hold_frac * 1000)):03d}"
                            )
                            returns_by_tag[tag] = returns_df
                            diag_df.to_csv(out_dir / f"diagnostics_{tag}.csv", index=False)
                            returns_df.to_csv(out_dir / f"returns_{tag}.csv", index=False)
                            done += 1
                            print(
                                f"{done}/{total} pv={portfolio_value/1e4:.0f}w CNY cap={cap:.2%} "
                                f"minadv={min_adv/1e6:.0f}m band={rebalance_band:.0%} "
                                f"target={target_frac:.3f} hold={hold_frac:.3f} ann={row['ann']:.2f}% "
                                f"sharpe={row['sharpe']:.3f} exec_to={row['avg_executed_turnover']:.3f} "
                                f"unfilled={row['avg_unfilled_turnover']:.3f}",
                                flush=True,
                            )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "execution_constraint_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    save_stage_breakdown(out_dir, returns_by_tag)

    top = summary.sort_values(["sharpe", "ann"], ascending=False).head(20)
    top.to_csv(out_dir / "top_by_sharpe.csv", index=False)
    lines = [
        "# Execution Constraint Sweep",
        "",
        f"- alpha_jsonl: `{args.alpha_jsonl}`",
        f"- target_fracs: `{args.target_fracs}`",
        f"- hold_fracs: `{args.hold_fracs}`",
        f"- portfolio_values: `{args.portfolio_values}`",
        f"- adv_participation_caps: `{args.adv_participation_caps}`",
        f"- min_adv_cnys: `{args.min_adv_cnys}`",
        f"- execution_lag: `{args.execution_lag}`",
        f"- rebalance_bands: `{args.rebalance_bands}`",
        "",
        "## Top By Sharpe",
        "",
        "| portfolio | cap | min ADV | target | hold | ann | Sharpe | mdd | exec turnover | unfilled turnover |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top.to_dict("records"):
        lines.append(
            f"| {row['portfolio_value']/1e4:.0f}万 | {row['adv_participation_cap']:.2%} | "
            f"{row['min_adv_cny']/1e6:.0f}M | {row['target_frac']:.3f} | {row['hold_frac']:.3f} | "
            f"{row['ann']:.2f}% | {row['sharpe']:.3f} | {row['mdd']*100:.2f}% | "
            f"{row['avg_executed_turnover']:.3f} | {row['avg_unfilled_turnover']:.3f} |"
        )
    (out_dir / "execution_constraint_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved sweep summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
