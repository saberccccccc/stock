#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Daily backtest for the v1 hold/sell trade policy.

This backtest does not use fixed 5-day holding semantics. It re-scores the
cross-section on each validation sample date, lets the policy adjust existing
holdings, and fills the remaining book with alpha-ranked names.
"""
import argparse
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from backtest.engine import (
    _compute_daily_portfolio_returns,
    _rolling_beta_neutralize,
    build_universe_matrix,
    calc_metrics,
    detect_regime,
    execute_order_with_impact,
)
from backtest.reports import save_summary_csv
from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_dl_predictor
from run.build_trade_policy_dataset import compute_ret_features, rank_pct_desc, safe_get_price
from run.v9_long_only_optimization import V9RankPredictor


def legacy_market_mult(idx_close, col_cur):
    if col_cur < 60 or not np.isfinite(idx_close.iloc[col_cur]):
        return 1.0
    idx_ma60 = idx_close.iloc[col_cur - 60:col_cur].mean()
    idx_cur = idx_close.iloc[col_cur]
    mult = 1.0
    if idx_cur < idx_ma60:
        mult = 0.7
    if col_cur >= 120 and np.isfinite(idx_close.iloc[col_cur - 120]) and idx_close.iloc[col_cur - 120] > 0:
        idx_ret_6m = idx_close.iloc[col_cur] / idx_close.iloc[col_cur - 120] - 1.0
        if idx_ret_6m < -0.10:
            mult = min(mult, 0.3)
    return float(mult)


def make_feature_row(code, i, alpha, rank, alpha_hist, rank_hist, alpha_ma, holdings, price_dict, dt):
    state = holdings.get(code, {})
    prev_a = alpha_hist.get(code, {}).get("last", np.nan)
    prev3_a = alpha_hist.get(code, {}).get("lag3", np.nan)
    prev_r = rank_hist.get(code, {}).get("last", np.nan)
    prev3_r = rank_hist.get(code, {}).get("lag3", np.nan)
    px = safe_get_price(price_dict, code, dt)
    entry_px = state.get("entry_price", np.nan)
    unrealized = px / entry_px - 1.0 if np.isfinite(px) and np.isfinite(entry_px) and entry_px > 0 else np.nan
    ret_feats = compute_ret_features(price_dict, code, dt)
    return {
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
    }


def predict_hold_prob(bundle, feature_row):
    df = pd.DataFrame([feature_row])
    X = df.reindex(columns=bundle["feature_cols"])
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:, 1][0])
    return float(model.decision_function(X)[0])


def load_index_series(cfg, all_dates):
    idx_path = Path(cfg.data_dir) / "hs300_index.csv"
    if not idx_path.exists():
        return pd.Series(np.nan, index=all_dates), pd.Series(0.0, index=all_dates)
    idx_df = pd.read_csv(idx_path)
    date_col = "trade_date" if "trade_date" in idx_df.columns else "date"
    idx_df[date_col] = pd.to_datetime(idx_df[date_col])
    idx_df.set_index(date_col, inplace=True)
    idx_close = idx_df["close"].reindex(all_dates)
    return idx_close, idx_close.pct_change().fillna(0)


def run_policy_backtest(args):
    cfg = build_v9_backtest_config()
    runtime = load_backtest_runtime(cfg, use_cache=True)
    base = load_dl_predictor(args.checkpoint, runtime.train, runtime.cfg)
    predictor = V9RankPredictor(base, "v9_raw", cache={})
    bundle = joblib.load(args.policy_model)

    all_codes = sorted(set(c for s in runtime.val for c in s["codes"]))
    price_mat, vol_mat, all_dates, code2idx = build_universe_matrix(runtime.price_dict, runtime.vol_dict, all_codes)
    idx_close, idx_daily = load_index_series(runtime.cfg, all_dates)
    T_total = len(all_dates)
    ret_daily = np.full((len(all_codes), T_total - 1), np.nan)
    for i in range(len(all_codes)):
        p = price_mat[i]
        ret_daily[i] = p[1:] / p[:-1] - 1.0
    ret_daily[~np.isfinite(ret_daily)] = 0.0

    date2idx = {}
    for s in runtime.val:
        dt = pd.Timestamp(s["date"])
        pos = all_dates.searchsorted(dt, side="right") - 1
        date2idx[dt] = max(pos, 0)

    daily_weights = [np.zeros(len(all_codes), dtype=np.float64) for _ in range(T_total)]
    daily_costs = np.zeros(T_total, dtype=np.float64)
    current_w = np.zeros(len(all_codes), dtype=np.float64)
    holdings = {}
    alpha_hist = {}
    rank_hist = {}
    alpha_ma = defaultdict(lambda: deque(maxlen=args.alpha_window))
    last_fill_start = None
    diag = defaultdict(list)

    samples = runtime.val[:args.limit_val] if args.limit_val else runtime.val
    for sample in samples:
        dt = pd.Timestamp(sample["date"])
        col_cur = date2idx.get(dt)
        if col_cur is None or col_cur < args.hist_window:
            continue
        entry_day = col_cur + 1
        if entry_day >= T_total:
            continue

        if last_fill_start is not None:
            for d in range(last_fill_start, min(entry_day, T_total)):
                daily_weights[d] = current_w.copy()

        codes_all = sample["codes"]
        idx_all = [code2idx[c] for c in codes_all]
        price_hist = price_mat[idx_all, col_cur - args.hist_window:col_cur]
        valid = np.sum(~np.isnan(price_hist), axis=1) >= 0.7 * args.hist_window
        if not np.any(valid):
            continue

        codes = [codes_all[i] for i in range(len(codes_all)) if valid[i]]
        idx_full = np.array([idx_all[i] for i in range(len(idx_all)) if valid[i]], dtype=int)
        regime = detect_regime(sample)
        alpha = np.asarray(predictor.predict_alpha(sample, valid, regime), dtype=np.float64)
        if len(alpha) != len(codes):
            continue

        rank = rank_pct_desc(alpha)
        alpha_score = 1.0 - rank
        scores = alpha_score.copy()
        code_to_i = {c: i for i, c in enumerate(codes)}

        policy_evaluated = 0
        for code in list(holdings.keys()):
            i = code_to_i.get(code)
            if i is None:
                continue
            feat = make_feature_row(code, i, alpha, rank, alpha_hist, rank_hist, alpha_ma, holdings, runtime.price_dict, dt)
            hold_prob = predict_hold_prob(bundle, feat)
            scores[i] = hold_prob
            policy_evaluated += 1

        k = max(1, int(len(codes) * args.top_frac))
        selected_local = np.argsort(scores)[::-1][:k]
        selected_codes = [codes[i] for i in selected_local]

        market_mult = legacy_market_mult(idx_close, col_cur) if args.market_timing == "legacy" else 1.0
        target_w = np.zeros(len(all_codes), dtype=np.float64)
        gross = min(1.0, market_mult)
        weight = min(args.max_weight, gross / max(len(selected_local), 1))
        for i in selected_local:
            target_w[idx_full[i]] = weight

        price_next = price_mat[:, entry_day]
        vol_next = vol_mat[:, entry_day] if vol_mat is not None else np.ones(len(all_codes)) * 1e9
        tradable = np.isfinite(price_next) & (vol_next > 0)
        target_tradable = current_w + np.where(tradable, target_w - current_w, 0.0)
        current_w, impact_cost, fill_ratio, trade_exec = execute_order_with_impact(
            target_tradable,
            current_w,
            price_next,
            vol_next,
            adv_ratio=args.adv_limit_ratio,
            impact_coeff=args.impact_coeff,
            portfolio_value=args.portfolio_value,
            regime=regime,
        )
        daily_costs[entry_day] += impact_cost
        last_fill_start = entry_day

        next_holdings = {}
        for code in selected_codes:
            i = code_to_i[code]
            old = holdings.get(code)
            if old is not None:
                next_holdings[code] = {**old, "holding_days": int(old.get("holding_days", 0)) + 1}
            else:
                next_holdings[code] = {
                    "entry_date": dt,
                    "entry_alpha": float(alpha[i]),
                    "entry_rank_pct": float(rank[i]),
                    "entry_price": safe_get_price(runtime.price_dict, code, dt),
                    "holding_days": 1,
                }
        holdings = next_holdings

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

        diag["policy_evaluated"].append(policy_evaluated)
        diag["turnover"].append(float(np.sum(np.abs(trade_exec))))
        diag["gross"].append(float(np.sum(np.abs(current_w))))
        diag["impact_cost"].append(float(impact_cost))
        diag["market_mult"].append(float(market_mult))

    if last_fill_start is not None:
        for d in range(last_fill_start, T_total):
            daily_weights[d] = current_w.copy()

    raw_ret = _compute_daily_portfolio_returns(daily_weights, ret_daily, daily_costs, T_total)
    idx_ret_arr = np.asarray(idx_daily.iloc[1:len(raw_ret) + 1].values, dtype=np.float64)
    neu_ret, _ = _rolling_beta_neutralize(raw_ret, idx_ret_arr)

    active = np.array([np.sum(np.abs(w)) > 0 for w in daily_weights[1:]], dtype=bool)
    if np.any(active):
        first = int(np.argmax(active))
        last = len(active) - int(np.argmax(active[::-1]))
        raw_ret = raw_ret[first:last]
        neu_ret = neu_ret[first:last]

    ann_raw, sharpe_raw, mdd_raw = calc_metrics(raw_ret)
    ann_neu, sharpe_neu, mdd_neu = calc_metrics(neu_ret)
    row = {
        "mode": "trade_policy_v1",
        "top_frac": args.top_frac,
        "ann_raw": ann_raw,
        "sharpe_raw": sharpe_raw,
        "mdd_raw": mdd_raw,
        "ann_neu": ann_neu,
        "sharpe_neu": sharpe_neu,
        "mdd_neu": mdd_neu,
        "avg_turnover": float(np.mean(diag["turnover"])) if diag["turnover"] else 0.0,
        "avg_gross": float(np.mean(diag["gross"])) if diag["gross"] else 0.0,
        "avg_policy_evaluated": float(np.mean(diag["policy_evaluated"])) if diag["policy_evaluated"] else 0.0,
        "total_impact_cost": float(np.sum(diag["impact_cost"])) if diag["impact_cost"] else 0.0,
    }
    return row, raw_ret, neu_ret


def main():
    parser = argparse.ArgumentParser(description="Backtest daily trade policy v1.")
    parser.add_argument("--checkpoint", default="checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt")
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--output-dir", default="backtest_results_exp_trade_policy_v1")
    parser.add_argument("--top-frac", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--hist-window", type=int, default=60)
    parser.add_argument("--alpha-window", type=int, default=3)
    parser.add_argument("--market-timing", choices=["legacy", "none"], default="legacy")
    parser.add_argument("--adv-limit-ratio", type=float, default=0.02)
    parser.add_argument("--impact-coeff", type=float, default=0.1)
    parser.add_argument("--portfolio-value", type=float, default=1e8)
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    row, raw_ret, neu_ret = run_policy_backtest(args)
    save_summary_csv([row], out_dir / "trade_policy_v1_summary.csv", display_columns=list(row.keys()), title="Trade policy v1")
    pd.DataFrame({"daily_return": raw_ret, "neutral_return": neu_ret}).to_csv(out_dir / "trade_policy_v1_returns.csv", index=False)
    (out_dir / "trade_policy_v1_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
