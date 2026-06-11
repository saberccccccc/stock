#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build A->B switch value samples for a no-hand-threshold trade model."""
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
from backtest.predictors import PersistentPredictor
from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_dl_predictor
from run.build_trade_policy_dataset import compute_ret_features, rank_pct_desc, safe_get_price
from run.v9_long_only_optimization import V9RankPredictor


HORIZONS = (1, 3, 5)


def safe_get_volume(vol_dict, code, dt):
    ser = vol_dict.get(code)
    if ser is None:
        return np.nan
    try:
        return float(ser.loc[dt])
    except Exception:
        try:
            pos = ser.index.searchsorted(dt, side="right") - 1
            if pos < 0:
                return np.nan
            return float(ser.iloc[pos])
        except Exception:
            return np.nan


def prev_close_return(price_dict, code, dt):
    df = price_dict.get(code)
    if df is None:
        return np.nan
    try:
        pos = df.index.searchsorted(dt, side="right") - 1
        if pos <= 0:
            return np.nan
        close = df["close"].astype(float).values
        if close[pos - 1] <= 0:
            return np.nan
        return float(close[pos] / close[pos - 1] - 1.0)
    except Exception:
        return np.nan


def alpha_history_features(code, i, alpha, rank, alpha_hist, rank_hist, alpha_ma, prefix):
    prev_a = alpha_hist.get(code, {}).get("last", np.nan)
    prev3_a = alpha_hist.get(code, {}).get("lag3", np.nan)
    prev_r = rank_hist.get(code, {}).get("last", np.nan)
    prev3_r = rank_hist.get(code, {}).get("lag3", np.nan)
    ma = float(np.mean(alpha_ma[code])) if alpha_ma[code] else np.nan
    return {
        f"{prefix}_alpha": float(alpha[i]),
        f"{prefix}_rank_pct": float(rank[i]),
        f"{prefix}_alpha_change_1d": float(alpha[i] - prev_a) if np.isfinite(prev_a) else np.nan,
        f"{prefix}_alpha_change_3d": float(alpha[i] - prev3_a) if np.isfinite(prev3_a) else np.nan,
        f"{prefix}_rank_change_1d": float(rank[i] - prev_r) if np.isfinite(prev_r) else np.nan,
        f"{prefix}_rank_change_3d": float(rank[i] - prev3_r) if np.isfinite(prev3_r) else np.nan,
        f"{prefix}_alpha_ma3": ma,
        f"{prefix}_alpha_vs_ma3": float(alpha[i] - ma) if np.isfinite(ma) else np.nan,
    }


def prefixed_ret_features(price_dict, code, dt, prefix):
    feats = compute_ret_features(price_dict, code, dt)
    return {f"{prefix}_{k}": v for k, v in feats.items()}


def execution_features(price_dict, vol_dict, code, dt, position_weight, portfolio_value, prefix):
    px = safe_get_price(price_dict, code, dt)
    vol = safe_get_volume(vol_dict, code, dt)
    dollar_vol = px * vol * 100.0 if np.isfinite(px) and px > 0 and np.isfinite(vol) and vol > 0 else 0.0
    trade_notional = float(position_weight) * float(portfolio_value)
    adv_trade_ratio = trade_notional / dollar_vol if dollar_vol > 0 else np.nan
    close_ret = prev_close_return(price_dict, code, dt)
    return {
        f"{prefix}_dollar_vol": float(dollar_vol) if dollar_vol > 0 else np.nan,
        f"{prefix}_adv_trade_ratio": float(adv_trade_ratio) if np.isfinite(adv_trade_ratio) else np.nan,
        f"{prefix}_close_ret_1d": float(close_ret) if np.isfinite(close_ret) else np.nan,
        f"{prefix}_near_limit_up": float(close_ret >= 0.095) if np.isfinite(close_ret) else np.nan,
        f"{prefix}_near_limit_down": float(close_ret <= -0.095) if np.isfinite(close_ret) else np.nan,
    }


def estimate_switch_cost(row, args):
    explicit = 2.0 * float(args.commission_rate) + 2.0 * float(args.slippage_rate) + float(args.stamp_tax_rate)
    sell_ratio = row.get("A_adv_trade_ratio", np.nan)
    buy_ratio = row.get("B_adv_trade_ratio", np.nan)
    sell_impact = float(args.impact_coeff) * min(float(sell_ratio), 1.0) ** 2 if np.isfinite(sell_ratio) else 0.0
    buy_impact = float(args.impact_coeff) * min(float(buy_ratio), 1.0) ** 2 if np.isfinite(buy_ratio) else 0.0
    # Execution risk proxy is recorded as a cost-like field for analysis. It is
    # based only on current limit/liquidity information and is not optimized here.
    limit_risk = 0.0
    if row.get("B_near_limit_up", 0.0) == 1.0:
        limit_risk += explicit
    if row.get("A_near_limit_down", 0.0) == 1.0:
        limit_risk += explicit
    return {
        "switch_explicit_cost": explicit,
        "switch_impact_cost": sell_impact + buy_impact,
        "switch_execution_risk_cost": limit_risk,
        "switch_full_cost": explicit + sell_impact + buy_impact + limit_risk,
    }


def choose_candidate_indices(order_desc, holdings_set, a_i, alpha, industry_ids, max_pairs, rng):
    candidates = [int(i) for i in order_desc if int(i) != int(a_i) and int(i) not in holdings_set]
    if not candidates:
        return []

    selected = []
    selected.append(candidates[0])

    a_ind = int(industry_ids[a_i]) if len(industry_ids) > a_i else -1
    same = [i for i in candidates if int(industry_ids[i]) == a_ind]
    if same:
        selected.append(same[0])

    better_than_a = [i for i in candidates if alpha[i] > alpha[a_i]]
    if better_than_a:
        selected.append(min(better_than_a, key=lambda i: alpha[i] - alpha[a_i]))

    if len(candidates) > 1:
        selected.append(candidates[int(rng.integers(0, len(candidates)))])

    out = []
    seen = set()
    for i in selected:
        if i not in seen:
            out.append(i)
            seen.add(i)
        if len(out) >= max_pairs:
            break
    return out


def build_split_rows(split, samples, predictor, price_dict, vol_dict, args, limit_samples=None):
    rows = []
    holdings = {}
    alpha_hist = {}
    rank_hist = {}
    alpha_ma = defaultdict(lambda: deque(maxlen=args.alpha_window))
    rng = np.random.default_rng(args.seed + (0 if split == "train" else 100000))

    if limit_samples:
        samples = samples[:limit_samples]

    total_samples = len(samples)
    for sample_idx, sample in enumerate(samples, start=1):
        if args.progress_every and sample_idx % args.progress_every == 0:
            print(
                f"[{split}] {sample_idx}/{total_samples} samples, "
                f"rows={len(rows)}, holdings={len(holdings)}",
                flush=True,
            )
        dt = pd.Timestamp(sample["date"])
        y_seq = sample["y_seq"]
        if y_seq.shape[1] < max(HORIZONS):
            continue
        # Do not use future-label availability to define today's tradable/ranked
        # universe. Missing future labels are filtered per A->B sample below.
        valid = np.ones(len(sample["codes"]), dtype=bool)

        regime = detect_regime(sample)
        raw_alpha = np.asarray(predictor.predict_alpha(sample, valid, regime), dtype=np.float64)
        if len(raw_alpha) != len(sample["codes"]):
            continue
        alpha_valid = np.isfinite(raw_alpha)
        if np.count_nonzero(alpha_valid) < 2:
            continue
        alpha = raw_alpha[alpha_valid]
        codes = [sample["codes"][i] for i in np.flatnonzero(alpha_valid)]
        y_valid = y_seq[alpha_valid]
        industry_ids = sample["industry_ids"][alpha_valid]

        rank = rank_pct_desc(alpha)
        code_to_i = {c: i for i, c in enumerate(codes)}
        order = np.argsort(alpha)[::-1]
        k_buy = max(1, int(len(codes) * args.holding_frac))
        k_candidate = max(k_buy, int(len(codes) * args.candidate_frac))
        position_weight = min(args.max_weight, 1.0 / max(k_buy, 1))
        current_top_codes = [codes[int(i)] for i in order[:k_buy]]
        holdings_idx = {code_to_i[c] for c in holdings if c in code_to_i}
        candidate_order = order[:k_candidate]
        needed_idx = set(candidate_order.tolist())
        needed_idx.update(code_to_i[c] for c in holdings if c in code_to_i)
        needed_codes = [codes[int(i)] for i in needed_idx]
        alpha_feat_cache = {
            code: alpha_history_features(code, code_to_i[code], alpha, rank, alpha_hist, rank_hist, alpha_ma, "")
            for code in needed_codes
        }
        # Strip the leading "_" introduced by empty prefix.
        alpha_feat_cache = {
            code: {k[1:] if k.startswith("_") else k: v for k, v in feats.items()}
            for code, feats in alpha_feat_cache.items()
        }
        ret_feat_cache = {code: compute_ret_features(price_dict, code, dt) for code in needed_codes}
        exec_feat_cache = {
            code: execution_features(price_dict, vol_dict, code, dt, position_weight, args.portfolio_value, "")
            for code in needed_codes
        }
        exec_feat_cache = {
            code: {k[1:] if k.startswith("_") else k: v for k, v in feats.items()}
            for code, feats in exec_feat_cache.items()
        }
        price_cache = {code: safe_get_price(price_dict, code, dt) for code in needed_codes}

        for a_code, state in list(holdings.items()):
            a_i = code_to_i.get(a_code)
            if a_i is None:
                continue
            candidate_indices = choose_candidate_indices(
                candidate_order,
                holdings_idx,
                a_i,
                alpha,
                industry_ids,
                args.max_pairs_per_holding,
                rng,
            )
            for b_i in candidate_indices:
                b_code = codes[b_i]
                if not np.isfinite(y_valid[a_i, [h - 1 for h in HORIZONS]]).all():
                    continue
                if not np.isfinite(y_valid[b_i, [h - 1 for h in HORIZONS]]).all():
                    continue
                row = {
                    "split": split,
                    "date": dt.strftime("%Y-%m-%d"),
                    "A_code": a_code,
                    "B_code": b_code,
                    "market_regime": regime,
                    "same_industry": int(industry_ids[a_i] == industry_ids[b_i]),
                    "A_holding_days": int(state.get("holding_days", 0)),
                    "A_entry_rank_pct": float(state.get("entry_rank_pct", np.nan)),
                    "A_entry_alpha": float(state.get("entry_alpha", np.nan)),
                }
                row.update({f"A_{k}": v for k, v in alpha_feat_cache[a_code].items()})
                row.update({f"B_{k}": v for k, v in alpha_feat_cache[b_code].items()})
                row.update({f"A_{k}": v for k, v in ret_feat_cache[a_code].items()})
                row.update({f"B_{k}": v for k, v in ret_feat_cache[b_code].items()})
                row.update({f"A_{k}": v for k, v in exec_feat_cache[a_code].items()})
                row.update({f"B_{k}": v for k, v in exec_feat_cache[b_code].items()})

                a_px = price_cache.get(a_code, np.nan)
                entry_px = state.get("entry_price", np.nan)
                row["A_unrealized_pnl"] = (
                    float(a_px / entry_px - 1.0)
                    if np.isfinite(a_px) and np.isfinite(entry_px) and entry_px > 0
                    else np.nan
                )
                row["alpha_diff"] = row["B_alpha"] - row["A_alpha"]
                row["rank_advantage"] = row["A_rank_pct"] - row["B_rank_pct"]
                row["ret_1d_diff"] = row["B_ret_1d"] - row["A_ret_1d"]
                row["ret_3d_diff"] = row["B_ret_3d"] - row["A_ret_3d"]
                row["ret_5d_diff"] = row["B_ret_5d"] - row["A_ret_5d"]
                row["vol_20d_diff"] = row["B_vol_20d"] - row["A_vol_20d"]
                row["drawdown_20d_diff"] = row["B_drawdown_20d"] - row["A_drawdown_20d"]

                row.update(estimate_switch_cost(row, args))
                for h in HORIZONS:
                    a_ret = float(y_valid[a_i, h - 1])
                    b_ret = float(y_valid[b_i, h - 1])
                    row[f"A_ret_fwd_h{h}"] = a_ret
                    row[f"B_ret_fwd_h{h}"] = b_ret
                    row[f"switch_edge_raw_h{h}"] = b_ret - a_ret
                    row[f"switch_edge_net_h{h}"] = b_ret - a_ret - row["switch_full_cost"]
                    row[f"switch_success_h{h}"] = int(row[f"switch_edge_net_h{h}"] > 0.0)
                rows.append(row)

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

        next_holdings = {}
        for code in current_top_codes:
            i = code_to_i[code]
            old = holdings.get(code)
            if old is not None:
                next_holdings[code] = {**old, "holding_days": int(old.get("holding_days", 0)) + 1}
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
    parser = argparse.ArgumentParser(description="Build switch value A->B dataset.")
    parser.add_argument("--checkpoint", default="checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt")
    parser.add_argument("--output-dir", default="switch_value_data_20260530")
    parser.add_argument("--holding-frac", type=float, default=0.05)
    parser.add_argument("--candidate-frac", type=float, default=0.10)
    parser.add_argument("--predictor-mode", default="none", choices=["none", "average", "momentum", "composite"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--max-pairs-per-holding", type=int, default=4)
    parser.add_argument("--alpha-window", type=int, default=3)
    parser.add_argument("--portfolio-value", type=float, default=1e8)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--commission-rate", type=float, default=0.0001)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--impact-coeff", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--skip-csv", action="store_true")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_v9_backtest_config()
    runtime = load_backtest_runtime(cfg, use_cache=True)
    base = load_dl_predictor(args.checkpoint, runtime.train, runtime.cfg)

    rows = []
    for split, samples, limit in [
        ("train", runtime.train, args.limit_train),
        ("val", runtime.val, args.limit_val),
    ]:
        raw_predictor = V9RankPredictor(base, "v9_raw", cache={})
        if args.predictor_mode == "none":
            predictor = raw_predictor
        else:
            predictor = PersistentPredictor(raw_predictor, window=args.window, mode=args.predictor_mode)
        split_rows = build_split_rows(
            split,
            samples,
            predictor,
            runtime.price_dict,
            runtime.vol_dict,
            args,
            limit_samples=limit,
        )
        print(f"{split}: {len(split_rows)} switch rows")
        split_df = pd.DataFrame(split_rows)
        if not split_df.empty:
            if not args.skip_csv:
                split_df.to_csv(out_dir / f"switch_value_dataset_{split}.csv", index=False)
            try:
                split_df.to_parquet(out_dir / f"switch_value_dataset_{split}.parquet", index=False)
            except Exception as exc:
                print(f"{split} parquet skipped: {exc}")
        rows.extend(split_rows)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "switch_value_dataset.csv"
    if not args.skip_csv:
        df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(out_dir / "switch_value_dataset.parquet", index=False)
    except Exception as exc:
        print(f"parquet skipped: {exc}")

    summary = (
        df.groupby("split")
        .agg(
            rows=("switch_edge_net_h5", "size"),
            edge_h1_mean=("switch_edge_net_h1", "mean"),
            edge_h3_mean=("switch_edge_net_h3", "mean"),
            edge_h5_mean=("switch_edge_net_h5", "mean"),
            success_h5=("switch_success_h5", "mean"),
            cost_mean=("switch_full_cost", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "switch_value_dataset_summary.csv", index=False)
    (out_dir / "switch_value_dataset_config.json").write_text(
        pd.Series(vars(args)).to_json(indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    saved_path = out_dir / "switch_value_dataset.parquet" if args.skip_csv else csv_path
    print(f"Saved dataset to: {saved_path}")


if __name__ == "__main__":
    main()
