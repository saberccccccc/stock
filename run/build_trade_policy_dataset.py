#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a hold/sell policy dataset from V9 alpha states.

The first policy version learns whether an existing holding should be kept.
Buying remains alpha-driven in v1; this dataset focuses on replacing hand-written
sell thresholds such as "exit if rank drops below top15%".
"""
import argparse
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from backtest.engine import detect_regime
from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_dl_predictor
from run.v9_long_only_optimization import V9RankPredictor


def rank_pct_desc(alpha):
    """Return 0 for best rank and 1 for worst rank."""
    order = np.argsort(alpha)[::-1]
    out = np.empty(len(alpha), dtype=np.float64)
    out[order] = np.arange(len(alpha), dtype=np.float64)
    return out / max(len(alpha) - 1, 1)


def safe_get_price(price_dict, code, dt):
    df = price_dict.get(code)
    if df is None:
        return np.nan
    try:
        return float(df.loc[dt, "close"])
    except Exception:
        try:
            pos = df.index.searchsorted(dt, side="right") - 1
            if pos < 0:
                return np.nan
            return float(df.iloc[pos]["close"])
        except Exception:
            return np.nan


def compute_ret_features(price_dict, code, dt):
    df = price_dict.get(code)
    out = {
        "ret_1d": np.nan,
        "ret_3d": np.nan,
        "ret_5d": np.nan,
        "vol_10d": np.nan,
        "vol_20d": np.nan,
        "drawdown_20d": np.nan,
    }
    if df is None:
        return out
    try:
        pos = df.index.searchsorted(dt, side="right") - 1
        if pos <= 0:
            return out
        close = df["close"].astype(float).values
        cur = close[pos]
        if not np.isfinite(cur) or cur <= 0:
            return out
        for n in (1, 3, 5):
            if pos - n >= 0 and close[pos - n] > 0:
                out[f"ret_{n}d"] = float(cur / close[pos - n] - 1.0)
        for n in (10, 20):
            if pos - n >= 0:
                hist = close[pos - n:pos + 1]
                rr = hist[1:] / hist[:-1] - 1.0
                rr = rr[np.isfinite(rr)]
                if len(rr) > 2:
                    out[f"vol_{n}d"] = float(np.std(rr))
        if pos - 20 >= 0:
            high = np.nanmax(close[pos - 20:pos + 1])
            if np.isfinite(high) and high > 0:
                out["drawdown_20d"] = float(cur / high - 1.0)
    except Exception:
        pass
    return out


def market_features(price_dict, dt):
    idx = price_dict.get("__INDEX__")
    if idx is None:
        return {}
    return {}


def build_split_rows(
    split_name,
    samples,
    predictor,
    price_dict,
    horizon,
    top_frac,
    replace_frac,
    cost_buffer,
    alpha_window,
    limit_samples=None,
):
    rows = []
    holdings = {}
    alpha_hist = {}
    rank_hist = {}
    alpha_ma = defaultdict(lambda: deque(maxlen=alpha_window))

    if limit_samples:
        samples = samples[:limit_samples]

    for sample in samples:
        dt = pd.Timestamp(sample["date"])
        y_seq = sample["y_seq"]
        valid = np.isfinite(y_seq).all(axis=1)
        if valid.sum() < 50:
            continue

        regime = detect_regime(sample)
        alpha = predictor.predict_alpha(sample, valid, regime)
        alpha = np.asarray(alpha, dtype=np.float64)
        codes = [sample["codes"][i] for i in range(len(sample["codes"])) if valid[i]]
        y_valid = y_seq[valid]
        if len(codes) != len(alpha) or horizon - 1 >= y_valid.shape[1]:
            continue

        rank = rank_pct_desc(alpha)
        code_to_i = {c: i for i, c in enumerate(codes)}
        future_ret = y_valid[:, horizon - 1].astype(np.float64)

        k_buy = max(1, int(len(codes) * top_frac))
        k_replace = max(1, int(len(codes) * replace_frac))
        order = np.argsort(alpha)[::-1]
        top_buy_codes = [codes[i] for i in order[:k_buy]]
        replace_candidates = [i for i in order[:k_replace] if codes[i] not in holdings]
        replace_vals = future_ret[replace_candidates]
        replace_vals = replace_vals[np.isfinite(replace_vals)]
        replace_median = float(np.median(replace_vals)) if len(replace_vals) else np.nan
        replace_mean = float(np.mean(replace_vals)) if len(replace_vals) else np.nan

        for code, state in list(holdings.items()):
            i = code_to_i.get(code)
            if i is None or not np.isfinite(future_ret[i]) or not np.isfinite(replace_median):
                continue

            prev_a = alpha_hist.get(code, {}).get("last", np.nan)
            prev3_a = alpha_hist.get(code, {}).get("lag3", np.nan)
            prev_r = rank_hist.get(code, {}).get("last", np.nan)
            prev3_r = rank_hist.get(code, {}).get("lag3", np.nan)
            px = safe_get_price(price_dict, code, dt)
            entry_px = state.get("entry_price", np.nan)
            unrealized = px / entry_px - 1.0 if np.isfinite(px) and np.isfinite(entry_px) and entry_px > 0 else np.nan
            ret_feats = compute_ret_features(price_dict, code, dt)

            hold_ret = float(future_ret[i])
            label_hold = int(hold_ret >= replace_median - cost_buffer)
            rows.append({
                "split": split_name,
                "date": dt.strftime("%Y-%m-%d"),
                "code": code,
                "label_hold": label_hold,
                "hold_ret_fwd": hold_ret,
                "replace_ret_median": replace_median,
                "replace_ret_mean": replace_mean,
                "edge_vs_replace": hold_ret - replace_median,
                "alpha": float(alpha[i]),
                "alpha_rank_pct": float(rank[i]),
                "alpha_change_1d": float(alpha[i] - prev_a) if np.isfinite(prev_a) else np.nan,
                "alpha_change_3d": float(alpha[i] - prev3_a) if np.isfinite(prev3_a) else np.nan,
                "rank_change_1d": float(rank[i] - prev_r) if np.isfinite(prev_r) else np.nan,
                "rank_change_3d": float(rank[i] - prev3_r) if np.isfinite(prev3_r) else np.nan,
                "alpha_ma3": float(np.mean(alpha_ma[code])) if alpha_ma[code] else np.nan,
                "alpha_vs_ma3": float(alpha[i] - np.mean(alpha_ma[code])) if alpha_ma[code] else np.nan,
                "holding_days": int(state.get("holding_days", 0)),
                "entry_rank_pct": float(state.get("entry_rank_pct", np.nan)),
                "entry_alpha": float(state.get("entry_alpha", np.nan)),
                "rank_since_entry_change": float(rank[i] - state.get("entry_rank_pct", np.nan)),
                "alpha_since_entry_change": float(alpha[i] - state.get("entry_alpha", np.nan)),
                "unrealized_pnl": float(unrealized) if np.isfinite(unrealized) else np.nan,
                **ret_feats,
            })

        # Update histories after features are created, so changes are lagged.
        for i, code in enumerate(codes):
            hist_a = alpha_hist.setdefault(code, {})
            hist_r = rank_hist.setdefault(code, {})
            hist_a["lag3"] = hist_a.get("lag2", np.nan)
            hist_a["lag2"] = hist_a.get("last", np.nan)
            hist_a["last"] = float(alpha[i])
            hist_r["lag3"] = hist_r.get("lag2", np.nan)
            hist_r["lag2"] = hist_r.get("last", np.nan)
            hist_r["last"] = float(rank[i])
            alpha_ma[code].append(float(alpha[i]))

        # Baseline path for creating future holding states: each day holds top alpha names.
        next_holdings = {}
        for code in top_buy_codes:
            i = code_to_i[code]
            old = holdings.get(code)
            if old is not None:
                next_holdings[code] = {
                    **old,
                    "holding_days": int(old.get("holding_days", 0)) + 1,
                }
            else:
                next_holdings[code] = {
                    "entry_date": dt,
                    "entry_alpha": float(alpha[i]),
                    "entry_rank_pct": float(rank[i]),
                    "entry_price": safe_get_price(price_dict, code, dt),
                    "holding_days": 1,
                }
        holdings = next_holdings

    return rows


def main():
    parser = argparse.ArgumentParser(description="Build hold/sell trade policy dataset.")
    parser.add_argument("--checkpoint", default="checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt")
    parser.add_argument("--output-dir", default="trade_policy_data")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--top-frac", type=float, default=0.05)
    parser.add_argument("--replace-frac", type=float, default=0.05)
    parser.add_argument("--cost-buffer", type=float, default=0.005)
    parser.add_argument("--alpha-window", type=int, default=3)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_v9_backtest_config()
    runtime = load_backtest_runtime(cfg, use_cache=True)
    base = load_dl_predictor(args.checkpoint, runtime.train, runtime.cfg)

    rows = []
    for split_name, samples, limit in [
        ("train", runtime.train, args.limit_train),
        ("val", runtime.val, args.limit_val),
    ]:
        predictor = V9RankPredictor(base, "v9_raw", cache={})
        split_rows = build_split_rows(
            split_name,
            samples,
            predictor,
            runtime.price_dict,
            horizon=args.horizon,
            top_frac=args.top_frac,
            replace_frac=args.replace_frac,
            cost_buffer=args.cost_buffer,
            alpha_window=args.alpha_window,
            limit_samples=limit,
        )
        print(f"{split_name}: {len(split_rows)} policy rows")
        rows.extend(split_rows)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "hold_sell_policy_dataset.csv"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(out_dir / "hold_sell_policy_dataset.parquet", index=False)
    except Exception as exc:
        print(f"parquet skipped: {exc}")

    summary = (
        df.groupby("split")
        .agg(rows=("label_hold", "size"), hold_rate=("label_hold", "mean"), edge_mean=("edge_vs_replace", "mean"))
        .reset_index()
    )
    summary.to_csv(out_dir / "hold_sell_policy_dataset_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved dataset to: {csv_path}")


if __name__ == "__main__":
    main()
