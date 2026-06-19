"""Parameter sweep for open-price share-ledger Alpha execution."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

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


def parse_alpha_specs(raw):
    specs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, path = item.split("=", 1)
        specs.append((name.strip(), Path(path.strip())))
    if not specs:
        raise ValueError("No alpha specs provided")
    return specs


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep open-price share-ledger params")
    parser.add_argument("--alpha-specs", required=True, help="Comma list like max090=path,max095=path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--target-fracs", default="0.004,0.005,0.006,0.008")
    parser.add_argument("--hold-fracs", default="0.08,0.10,0.12,0.15")
    parser.add_argument("--rebalance-bands", default="0.15,0.20,0.25,0.30")
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
    parser.add_argument("--execution-lag", type=int, default=0)
    parser.add_argument("--max-data-date", default="2026-05-18")
    parser.add_argument("--progress-every", type=int, default=2500)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "open_price_ledger_param_sweep_summary.csv"

    alpha_specs = parse_alpha_specs(args.alpha_specs)
    rows_by_name = {}
    all_codes = set()
    for name, path in alpha_specs:
        rows = load_alpha_rows(path)
        assert_alpha_rows_within_research(rows, context=f"open-ledger sweep alpha {name}")
        rows_by_name[name] = rows
        all_codes.update(code for row in rows for code in row["codes"])
        print(f"loaded alpha {name}: days={len(rows)} path={path}", flush=True)

    all_codes = sorted(all_codes)
    print(f"loading shared OHLC: codes={len(all_codes)}", flush=True)
    open_df, close_df, money_df = load_ohlc_money(args.data_dir, all_codes, args.money_scale, args.progress_every)
    if args.max_data_date:
        max_data_date = assert_research_end_date(args.max_data_date, context="open-ledger sweep")
        open_df = open_df.loc[open_df.index <= max_data_date]
        close_df = close_df.loc[close_df.index <= max_data_date]
        money_df = money_df.loc[money_df.index <= max_data_date]
    adv_df = recompute_adv(money_df, args.adv_window)
    idx_close, idx_daily = load_index_returns(args.data_dir, args.index_file, close_df.index)

    targets = parse_float_list(args.target_fracs)
    holds = parse_float_list(args.hold_fracs)
    bands = parse_float_list(args.rebalance_bands)
    capitals = parse_float_list(args.portfolio_values)

    summary_rows = []
    total = len(alpha_specs) * len(bands) * len(capitals) * sum(1 for t in targets for h in holds if h >= t)
    done = 0
    for alpha_name, alpha_rows in rows_by_name.items():
        for band in bands:
            for capital in capitals:
                run_args = SimpleNamespace(**vars(args))
                run_args.rebalance_band = float(band)
                run_args.portfolio_value = float(capital)
                for target in targets:
                    for hold in holds:
                        if hold < target:
                            continue
                        row, _, _ = run_open_ledger(
                            alpha_rows,
                            open_df,
                            close_df,
                            adv_df,
                            target,
                            hold,
                            run_args,
                            idx_close,
                            idx_daily,
                        )
                        if not row:
                            continue
                        row.update({
                            "alpha_name": alpha_name,
                            "portfolio_value": float(capital),
                            "adv_participation_cap": float(args.adv_participation_cap),
                            "min_adv_cny": float(args.min_adv_cny),
                            "limit_threshold": float(args.limit_threshold),
                        })
                        summary_rows.append(row)
                        done += 1
                        if done % 10 == 0 or done == total:
                            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
                            print(
                                f"{done}/{total} {alpha_name} band={band:.2f} "
                                f"pv={capital/1e4:.0f}w target={target:.3f} hold={hold:.2f} "
                                f"ann={row['ann']:.2f}% sharpe={row['sharpe']:.3f}",
                                flush=True,
                            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    if not summary.empty:
        sort_cols = ["sharpe", "ann"]
        top = summary.sort_values(sort_cols, ascending=[False, False]).head(50)
        top.to_csv(output_dir / "top50_by_sharpe.csv", index=False)
        balanced = summary.assign(
            score=summary["sharpe"] * 10.0 + summary["ann"] / 10.0 - summary["mdd"] * 10.0 - summary["total_cost"]
        ).sort_values("score", ascending=False).head(50)
        balanced.to_csv(output_dir / "top50_balanced_score.csv", index=False)
    print(f"Saved sweep summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
