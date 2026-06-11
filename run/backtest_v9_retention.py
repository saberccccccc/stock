"""Run V9 alpha through the newer retention-first full-cost engine."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backtest.engine import detect_regime
from core.research_protocol import RESEARCH_END_DATE, assert_research_end_date
from backtest.predictors import PersistentPredictor
from backtest.reports import calc_extended_metrics, calc_metrics
from backtest.runtime import build_v9_backtest_config, load_dl_predictor
from data.pipeline import build_cross_section_dataset, samples_from_precomputed_metadata
from run.backtest_temporal_daily_top import build_universe_matrix_light, load_price_volume_light
from run.backtest_temporal_retention import load_index_returns, run_one_retention
from run.v9_long_only_optimization import V9RankPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="V9 alpha under retention-first daily backtest")
    parser.add_argument("--checkpoint", default="checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt")
    parser.add_argument("--split", default="val", choices=["val", "test", "all"])
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", default="backtest_results_v9_retention_20260531")
    parser.add_argument("--predictor-mode", default="average", choices=["none", "average", "momentum", "composite"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--target-fracs", default="0.20,0.30")
    parser.add_argument("--hold-fracs", default="0.80,0.90")
    parser.add_argument("--market-timing-mode", default="legacy", choices=["none", "legacy", "dynamic"])
    parser.add_argument("--market-min-mult", type=float, default=0.20)
    parser.add_argument("--market-max-mult", type=float, default=1.00)
    parser.add_argument("--weight-mode", default="equal", choices=["equal", "rank_linear", "alpha_positive"])
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--commission-rate", type=float, default=0.0001)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--index-file", default="hs300_index.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=80)
    parser.add_argument("--limit-dates", type=int, default=None)
    return parser.parse_args()


def split_bounds(args):
    if args.start_date:
        start = pd.Timestamp(args.start_date)
        end = assert_research_end_date(args.end_date, context="V9 retention backtest")
        return start, end
    if args.split == "val":
        return pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
    if args.split == "test":
        return pd.Timestamp("2025-01-01"), RESEARCH_END_DATE
    return None, RESEARCH_END_DATE


def filter_samples_by_date(samples, args):
    start, end = split_bounds(args)
    out = []
    for sample in samples:
        date = pd.Timestamp(sample["date"])
        if start is not None and date < start:
            continue
        if end is not None and date >= end:
            continue
        out.append(sample)
    if args.limit_dates is not None:
        out = out[: int(args.limit_dates)]
    return out


def load_v9_samples_and_predictor(args):
    cfg = build_v9_backtest_config()
    result = build_cross_section_dataset(cfg, use_cache=True)
    if not isinstance(result, dict):
        train_samples, val_samples = result
        all_samples = train_samples + val_samples
    else:
        cfg.low_feat_dim = result.get("low_agg_dim", getattr(cfg, "low_feat_dim", 14))
        train_samples = samples_from_precomputed_metadata(result, "train")
        val_samples = samples_from_precomputed_metadata(result, "val")
        all_samples = train_samples + val_samples

    base = load_dl_predictor(args.checkpoint, train_samples, cfg, args.device)
    raw = V9RankPredictor(base, "v9_raw", cache={})
    if args.predictor_mode == "none":
        predictor = raw
    else:
        predictor = PersistentPredictor(raw, window=args.window, mode=args.predictor_mode)
    samples = filter_samples_by_date(all_samples, args)
    samples.sort(key=lambda s: pd.Timestamp(s["date"]))
    return cfg, samples, predictor


def compute_v9_alpha_rows(samples, predictor, progress_every=80):
    rows = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        n = int(sample["X"].shape[0])
        if n < 10:
            continue
        valid = np.ones(n, dtype=bool)
        regime = detect_regime(sample)
        alpha = predictor.predict_alpha(sample, valid, regime)
        if alpha.shape[0] != n:
            raise RuntimeError(f"alpha length mismatch: alpha={alpha.shape[0]} n={n}")
        order = np.argsort(alpha)[::-1]
        codes = np.asarray(sample["codes"], dtype=object)
        rows.append({
            "date": pd.Timestamp(sample["date"]),
            "codes": codes[order].tolist(),
            "alpha": alpha[order].astype(float).tolist(),
            "n_stocks": n,
        })
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"alpha {i + 1}/{len(samples)} | n={n} | time={(time.time() - t0) / 60:.1f}m", flush=True)
    return rows


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
            ret = g["return"].to_numpy(float)
            ann, sharpe, mdd = calc_metrics(ret)
            yearly_rows.append({
                "tag": tag,
                "period": str(year),
                "days": len(g),
                "ann": ann,
                "sharpe": sharpe,
                "mdd": mdd,
                "sum_return": float(np.sum(ret)),
            })
        ret = df["return"].to_numpy(float)
        ann, sharpe, mdd = calc_metrics(ret)
        yearly_rows.append({
            "tag": tag,
            "period": "all",
            "days": len(df),
            "ann": ann,
            "sharpe": sharpe,
            "mdd": mdd,
            "sum_return": float(np.sum(ret)),
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

    cfg, samples, predictor = load_v9_samples_and_predictor(args)
    print(
        f"Computing V9 alphas: samples={len(samples)}, split={args.split}, "
        f"predictor={predictor.name}",
        flush=True,
    )
    alpha_rows = compute_v9_alpha_rows(samples, predictor, args.progress_every)
    with (out_dir / "v9_daily_alpha_top_order.jsonl").open("w", encoding="utf-8") as f:
        for row in alpha_rows:
            f.write(json.dumps({**row, "date": str(row["date"])}, ensure_ascii=False) + "\n")

    all_codes = sorted(set(code for row in alpha_rows for code in row["codes"]))
    print(f"Loading prices for {len(all_codes)} V9 codes...", flush=True)
    price_dict, vol_dict = load_price_volume_light(cfg.data_dir, all_codes)
    price_mat, _, all_dates, code2idx = build_universe_matrix_light(price_dict, vol_dict, all_codes)
    idx_close, idx_daily = load_index_returns(cfg.data_dir, args.index_file, all_dates)

    retention_args = SimpleNamespace(
        weight_mode=args.weight_mode,
        max_weight=args.max_weight,
        market_timing_mode=args.market_timing_mode,
        market_min_mult=args.market_min_mult,
        market_max_mult=args.market_max_mult,
        commission_rate=args.commission_rate,
        stamp_tax_rate=args.stamp_tax_rate,
        slippage_rate=args.slippage_rate,
    )
    target_fracs = [float(x.strip()) for x in args.target_fracs.split(",") if x.strip()]
    hold_fracs = [float(x.strip()) for x in args.hold_fracs.split(",") if x.strip()]
    summary_rows = []
    returns_by_tag = {}
    for target_frac in target_fracs:
        for hold_frac in hold_fracs:
            if hold_frac < target_frac:
                continue
            row, returns, diag_df = run_one_retention(
                alpha_rows,
                price_mat,
                all_dates,
                code2idx,
                target_frac,
                hold_frac,
                retention_args,
                idx_close,
                idx_daily,
            )
            row.update({
                "checkpoint": args.checkpoint,
                "split": args.split,
                "predictor": predictor.name,
                "window": args.window if args.predictor_mode != "none" else 1,
            })
            summary_rows.append(row)
            tag = f"target{int(round(target_frac * 1000)):03d}_hold{int(round(hold_frac * 1000)):03d}"
            signal_dates = [pd.Timestamp(row["date"]) for row in alpha_rows]
            if signal_dates:
                active_dates = pd.date_range(signal_dates[0], periods=len(returns), freq="D")
            else:
                active_dates = pd.DatetimeIndex([])
            # Prefer actual trading dates from the price matrix around active returns.
            if len(returns):
                first_entry = all_dates.searchsorted(signal_dates[0], side="right") if signal_dates else 1
                start_idx = max(first_entry - 1, 0)
                active_dates = all_dates[1:][start_idx:start_idx + len(returns)]
            returns_df = pd.DataFrame({"date": active_dates, "return": returns})
            returns_df.to_csv(out_dir / f"returns_{tag}.csv", index=False)
            diag_df.to_csv(out_dir / f"diagnostics_{tag}.csv", index=False)
            returns_by_tag[tag] = returns_df
            print(
                f"target={target_frac:.3f} hold={hold_frac:.3f} ann={row['ann']:.2f}% "
                f"sharpe={row['sharpe']:.3f} mdd={row['mdd'] * 100:.2f}% "
                f"turnover={row['avg_turnover']:.3f} hold_days={row['avg_holding_days']:.2f} "
                f"market={row['market_timing_mode']} mult={row['avg_market_mult']:.2f}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "v9_retention_summary.csv", index=False)
    save_stage_breakdown(out_dir, returns_by_tag)
    print(f"Saved summary: {out_dir / 'v9_retention_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
