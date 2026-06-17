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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backtest.presets import PRESETS, apply_preset_to_namespace, explicit_cli_dests, get_preset
from backtest.open_ledger import (
    load_alpha_rows,
    load_index_returns,
    load_ohlc_money,
    parse_float_list,
    recompute_adv,
    run_open_ledger,
    save_stage_breakdown,
)
from backtest.stress import STRESSES, get_stress
from core.research_protocol import (
    assert_alpha_rows_within_forward,
    assert_alpha_rows_within_research,
)

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
