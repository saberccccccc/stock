"""Capacity and liquidity diagnostics for retention-first alpha portfolios."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ADV participation for retention portfolios")
    parser.add_argument("--alpha-jsonl", required=True)
    parser.add_argument("--diagnostics-csv", default=None, help="Optional retention diagnostics with market_mult by date")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-frac", type=float, required=True)
    parser.add_argument("--hold-frac", type=float, required=True)
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--portfolio-values", default="500000,1000000")
    parser.add_argument("--money-scale", type=float, default=1000.0, help="raw money column scale to CNY")
    parser.add_argument("--progress-every", type=int, default=500)
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


def read_amount_series(data_dir, code, money_scale):
    path = Path(data_dir) / f"{code}.csv"
    if not path.exists():
        return None
    try:
        cols = pd.read_csv(path, nrows=0).columns.str.strip().str.lower().tolist()
        usecols = ["trade_date"]
        if "money" in cols:
            usecols.append("money")
        else:
            usecols.extend(["close", "volume"])
        df = pd.read_csv(path, usecols=usecols)
        df.columns = df.columns.str.strip().str.lower()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
        if "money" in df:
            amount = df["money"].astype(float) * float(money_scale)
        else:
            amount = df["close"].astype(float) * df["volume"].astype(float)
        amount = amount.replace([np.inf, -np.inf], np.nan)
        return amount
    except Exception:
        return None


def build_amount_and_adv(data_dir, codes, adv_window, money_scale, progress_every):
    series = {}
    for i, code in enumerate(codes, start=1):
        s = read_amount_series(data_dir, code, money_scale)
        if s is not None and not s.empty:
            series[code] = s
        if progress_every > 0 and i % progress_every == 0:
            print(f"loaded liquidity {i}/{len(codes)}", flush=True)
    all_dates = pd.DatetimeIndex(sorted(set().union(*[s.index for s in series.values()])))
    amount_cols = {}
    adv_cols = {}
    min_periods = max(3, int(adv_window) // 4)
    for code, s in series.items():
        re = s.reindex(all_dates)
        amount_cols[code] = re
        adv_cols[code] = re.rolling(int(adv_window), min_periods=min_periods).mean().shift(1)
    amount = pd.DataFrame(amount_cols, index=all_dates)
    adv = pd.DataFrame(adv_cols, index=all_dates)
    return amount, adv


def load_market_mult(path):
    if not path:
        return {}
    diag_path = Path(path)
    if not diag_path.exists():
        return {}
    df = pd.read_csv(diag_path)
    if "date" not in df.columns or "market_mult" not in df.columns:
        return {}
    df["date"] = pd.to_datetime(df["date"])
    return dict(zip(df["date"], df["market_mult"].astype(float)))


def reconstruct_daily_weights(alpha_rows, trading_dates, target_frac, hold_frac, market_mult_by_date):
    current_selected = []
    weights_by_date = {}
    selected_rows = []

    for row in alpha_rows:
        pos = trading_dates.searchsorted(row["date"], side="right")
        if pos <= 0 or pos >= len(trading_dates):
            continue
        entry_date = trading_dates[pos]
        codes = list(row["codes"])
        n = len(codes)
        if n == 0:
            continue
        target_n = max(1, int(n * target_frac))
        hold_n = max(target_n, int(n * hold_frac))
        rank_map = {code: i for i, code in enumerate(codes)}

        kept = [code for code in current_selected if rank_map.get(code, n + 1) < hold_n]
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

        mult = float(market_mult_by_date.get(entry_date, 1.0))
        weight = mult / max(len(selected), 1)
        weights_by_date[entry_date] = {code: weight for code in selected}
        selected_rows.append({
            "date": entry_date,
            "target_n": target_n,
            "hold_n": hold_n,
            "selected_n": len(selected),
            "kept_n": len(kept),
            "market_mult": mult,
        })
        current_selected = selected

    return weights_by_date, pd.DataFrame(selected_rows)


def summarize_participation(trade_df, portfolio_value):
    d = trade_df.copy()
    d["trade_value"] = d["abs_delta_weight"] * float(portfolio_value)
    valid = d["adv_cny"].replace([np.inf, -np.inf], np.nan) > 0
    d["participation"] = np.nan
    d.loc[valid, "participation"] = d.loc[valid, "trade_value"] / d.loc[valid, "adv_cny"]
    p = d["participation"].dropna()
    if p.empty:
        return {
            "portfolio_value": portfolio_value,
            "trades": len(d),
            "coverage": 0.0,
            "mean_participation": np.nan,
            "p50_participation": np.nan,
            "p90_participation": np.nan,
            "p95_participation": np.nan,
            "p99_participation": np.nan,
            "max_participation": np.nan,
            "share_gt_5pct_adv": np.nan,
            "share_gt_10pct_adv": np.nan,
            "share_gt_20pct_adv": np.nan,
        }
    return {
        "portfolio_value": portfolio_value,
        "trades": len(d),
        "coverage": float(len(p) / max(len(d), 1)),
        "mean_participation": float(p.mean()),
        "p50_participation": float(p.quantile(0.50)),
        "p90_participation": float(p.quantile(0.90)),
        "p95_participation": float(p.quantile(0.95)),
        "p99_participation": float(p.quantile(0.99)),
        "max_participation": float(p.max()),
        "share_gt_5pct_adv": float((p > 0.05).mean()),
        "share_gt_10pct_adv": float((p > 0.10).mean()),
        "share_gt_20pct_adv": float((p > 0.20).mean()),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_rows = load_alpha_rows(args.alpha_jsonl)
    needed_codes = sorted({code for row in alpha_rows for code in row["codes"]})
    print(f"alpha_days={len(alpha_rows)} needed_codes={len(needed_codes)}", flush=True)
    amount, adv = build_amount_and_adv(
        args.data_dir,
        needed_codes,
        args.adv_window,
        args.money_scale,
        args.progress_every,
    )
    market_mult_by_date = load_market_mult(args.diagnostics_csv)
    weights_by_date, state_df = reconstruct_daily_weights(
        alpha_rows,
        amount.index,
        args.target_frac,
        args.hold_frac,
        market_mult_by_date,
    )

    prev = {}
    trade_rows = []
    daily_rows = []
    for date in sorted(weights_by_date):
        cur = weights_by_date[date]
        codes = sorted(set(prev) | set(cur))
        deltas = {code: cur.get(code, 0.0) - prev.get(code, 0.0) for code in codes}
        nonzero = {code: delta for code, delta in deltas.items() if abs(delta) > 1e-12}
        day_abs = float(sum(abs(delta) for delta in nonzero.values()))
        day_adv = adv.loc[date] if date in adv.index else pd.Series(dtype=float)
        for code, delta in nonzero.items():
            trade_rows.append({
                "date": date,
                "code": code,
                "delta_weight": float(delta),
                "abs_delta_weight": float(abs(delta)),
                "side": "buy" if delta > 0 else "sell",
                "adv_cny": float(day_adv.get(code, np.nan)),
                "amount_cny": float(amount.at[date, code]) if code in amount.columns and date in amount.index else np.nan,
            })
        daily_rows.append({
            "date": date,
            "trade_count": len(nonzero),
            "turnover": day_abs,
            "selected_n": len(cur),
            "gross_weight": float(sum(abs(w) for w in cur.values())),
        })
        prev = cur

    trade_df = pd.DataFrame(trade_rows)
    daily_df = pd.DataFrame(daily_rows)
    if not trade_df.empty:
        trade_df["date"] = pd.to_datetime(trade_df["date"])
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"])

    portfolio_values = [float(x.strip()) for x in args.portfolio_values.split(",") if x.strip()]
    summary = pd.DataFrame([summarize_participation(trade_df, v) for v in portfolio_values])
    state_df.to_csv(out_dir / "capacity_state.csv", index=False)
    daily_df.to_csv(out_dir / "capacity_daily_trades.csv", index=False)
    trade_df.to_csv(out_dir / "capacity_trade_lots.csv", index=False)
    summary.to_csv(out_dir / "capacity_summary.csv", index=False)

    lines = [
        "# Retention Capacity Diagnostics",
        "",
        f"- alpha_jsonl: `{args.alpha_jsonl}`",
        f"- diagnostics_csv: `{args.diagnostics_csv}`",
        f"- target_frac: `{args.target_frac:.3f}`",
        f"- hold_frac: `{args.hold_frac:.3f}`",
        f"- ADV window: `{args.adv_window}` trading days, shifted by 1 day",
        f"- money_scale: `{args.money_scale:g}`",
        "",
        "## Daily Trading",
        "",
        f"- active days: `{len(daily_df)}`",
        f"- avg selected names: `{daily_df['selected_n'].mean():.1f}`" if not daily_df.empty else "- avg selected names: `nan`",
        f"- avg turnover: `{daily_df['turnover'].mean():.3f}`" if not daily_df.empty else "- avg turnover: `nan`",
        f"- p95 turnover: `{daily_df['turnover'].quantile(0.95):.3f}`" if not daily_df.empty else "- p95 turnover: `nan`",
        "",
        "## Participation By Portfolio Value",
        "",
        "| portfolio | coverage | p50 ADV% | p90 ADV% | p95 ADV% | p99 ADV% | max ADV% | >5% ADV | >10% ADV | >20% ADV |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['portfolio_value']:,.0f} | {row['coverage']:.2%} | "
            f"{row['p50_participation'] * 100:.2f}% | {row['p90_participation'] * 100:.2f}% | "
            f"{row['p95_participation'] * 100:.2f}% | {row['p99_participation'] * 100:.2f}% | "
            f"{row['max_participation'] * 100:.2f}% | {row['share_gt_5pct_adv']:.2%} | "
            f"{row['share_gt_10pct_adv']:.2%} | {row['share_gt_20pct_adv']:.2%} |"
        )
    (out_dir / "capacity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved capacity diagnostics: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
