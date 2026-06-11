#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retention-first daily backtest for the hold/sell trade policy.

This v2 script avoids the v1 mixed-score top-K issue. Existing holdings are
evaluated only by the learned hold/sell policy. Names below the learned
threshold are sold, surviving holdings are kept, and vacancies are filled from
the alpha-ranked buy pool.
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
from run.backtest_trade_policy import legacy_market_mult, load_index_series, make_feature_row, predict_hold_prob
from run.build_trade_policy_dataset import rank_pct_desc, safe_get_price
from run.v9_long_only_optimization import V9RankPredictor


def trim_returns_with_costs(raw_ret, neu_ret, daily_weights, daily_costs):
    active = np.array(
        [
            (np.sum(np.abs(w)) > 1e-12) or (abs(float(cost)) > 1e-12)
            for w, cost in zip(daily_weights[1:], daily_costs[1:])
        ],
        dtype=bool,
    )
    if not np.any(active):
        return raw_ret, np.asarray(neu_ret)
    first = int(np.argmax(active))
    last = len(active) - int(np.argmax(active[::-1]))
    return raw_ret[first:last], np.asarray(neu_ret)[first:last]


def build_equal_weight_target(all_codes, code2idx, selected_codes, gross, max_weight):
    target_w = np.zeros(len(all_codes), dtype=np.float64)
    if not selected_codes:
        return target_w
    weight = min(float(max_weight), float(gross) / max(len(selected_codes), 1))
    for code in selected_codes:
        idx = code2idx.get(code)
        if idx is not None:
            target_w[idx] = weight
    return target_w


def calc_explicit_trade_cost(trade_exec, commission_rate, stamp_tax_rate, slippage_rate):
    trade_exec = np.asarray(trade_exec, dtype=np.float64)
    buy_turnover = float(np.sum(np.maximum(trade_exec, 0.0)))
    sell_turnover = float(np.sum(np.maximum(-trade_exec, 0.0)))
    gross_turnover = buy_turnover + sell_turnover
    commission = float(commission_rate) * gross_turnover
    stamp_tax = float(stamp_tax_rate) * sell_turnover
    slippage = float(slippage_rate) * gross_turnover
    return commission + stamp_tax + slippage, {
        "buy_turnover": buy_turnover,
        "sell_turnover": sell_turnover,
        "explicit_commission_cost": commission,
        "explicit_stamp_tax_cost": stamp_tax,
        "explicit_slippage_cost": slippage,
    }


def update_holding_states(all_codes, current_w, old_holdings, selected_codes, alpha_by_code, rank_by_code, price_dict, dt):
    selected_set = set(selected_codes)
    next_holdings = {}
    active_idx = np.flatnonzero(np.abs(current_w) > 1e-10)
    for idx in active_idx:
        code = all_codes[int(idx)]
        old = old_holdings.get(code)
        if old is not None:
            next_holdings[code] = {**old, "holding_days": int(old.get("holding_days", 0)) + 1}
        elif code in selected_set:
            next_holdings[code] = {
                "entry_date": dt,
                "entry_alpha": float(alpha_by_code.get(code, np.nan)),
                "entry_rank_pct": float(rank_by_code.get(code, np.nan)),
                "entry_price": safe_get_price(price_dict, code, dt),
                "holding_days": 1,
            }
    return next_holdings


def run_policy_backtest_v2(args):
    cfg = build_v9_backtest_config()
    runtime = load_backtest_runtime(cfg, use_cache=True)
    base = load_dl_predictor(args.checkpoint, runtime.train, runtime.cfg)
    predictor = V9RankPredictor(base, "v9_raw", cache={})
    bundle = joblib.load(args.policy_model)
    sell_threshold = float(args.sell_threshold) if args.sell_threshold is not None else float(
        bundle.get("sell_threshold", bundle.get("metrics", {}).get("sell_threshold", 0.5))
    )

    all_codes = sorted(set(c for s in runtime.val for c in s["codes"]))
    price_mat, vol_mat, all_dates, code2idx = build_universe_matrix(runtime.price_dict, runtime.vol_dict, all_codes)
    idx_close, idx_daily = load_index_series(runtime.cfg, all_dates)
    t_total = len(all_dates)
    ret_daily = np.full((len(all_codes), t_total - 1), np.nan)
    for i in range(len(all_codes)):
        p = price_mat[i]
        ret_daily[i] = p[1:] / p[:-1] - 1.0
    ret_daily[~np.isfinite(ret_daily)] = 0.0

    date2idx = {}
    for sample in runtime.val:
        dt = pd.Timestamp(sample["date"])
        pos = all_dates.searchsorted(dt, side="right") - 1
        date2idx[dt] = max(pos, 0)

    daily_weights = [np.zeros(len(all_codes), dtype=np.float64) for _ in range(t_total)]
    daily_costs = np.zeros(t_total, dtype=np.float64)
    current_w = np.zeros(len(all_codes), dtype=np.float64)
    holdings = {}
    alpha_hist = {}
    rank_hist = {}
    alpha_ma = defaultdict(lambda: deque(maxlen=args.alpha_window))
    active_from_day = None
    diag = defaultdict(list)

    samples = runtime.val[:args.limit_val] if args.limit_val else runtime.val
    for sample in samples:
        dt = pd.Timestamp(sample["date"])
        col_cur = date2idx.get(dt)
        if col_cur is None or col_cur < args.hist_window:
            continue
        entry_day = col_cur + 1
        active_day = entry_day + 1
        if active_day >= t_total:
            continue

        if active_from_day is not None:
            for day in range(active_from_day, min(active_day, t_total)):
                daily_weights[day] = current_w.copy()

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
        code_to_i = {code: i for i, code in enumerate(codes)}
        alpha_by_code = {code: float(alpha[i]) for code, i in code_to_i.items()}
        rank_by_code = {code: float(rank[i]) for code, i in code_to_i.items()}

        k = max(1, int(len(codes) * args.top_frac))
        keep_rows = []
        policy_evaluated = 0
        missing_holdings = 0
        if args.mode == "policy":
            for code in list(holdings.keys()):
                i = code_to_i.get(code)
                if i is None:
                    missing_holdings += 1
                    continue
                feat = make_feature_row(code, i, alpha, rank, alpha_hist, rank_hist, alpha_ma, holdings, runtime.price_dict, dt)
                hold_prob = predict_hold_prob(bundle, feat)
                keep_rows.append((code, hold_prob, i))
                policy_evaluated += 1

        keep_rows.sort(key=lambda x: x[1], reverse=True)
        kept_codes = [code for code, hold_prob, _ in keep_rows if hold_prob >= sell_threshold]
        if len(kept_codes) > k:
            kept_codes = kept_codes[:k]
        kept_set = set(kept_codes)

        selected_codes = list(kept_codes)
        order = np.argsort(alpha)[::-1]
        for i in order:
            code = codes[int(i)]
            if code in kept_set:
                continue
            selected_codes.append(code)
            if len(selected_codes) >= k:
                break

        market_mult = legacy_market_mult(idx_close, col_cur) if args.market_timing == "legacy" else 1.0
        gross = min(1.0, market_mult)
        target_w = build_equal_weight_target(all_codes, code2idx, selected_codes, gross, args.max_weight)

        price_next = price_mat[:, entry_day]
        vol_next = vol_mat[:, entry_day] if vol_mat is not None else np.ones(len(all_codes)) * 1e9
        tradable = np.isfinite(price_next) & (vol_next > 0)
        target_tradable = current_w + np.where(tradable, target_w - current_w, 0.0)
        active_trade = np.abs(target_tradable - current_w) > 1e-8
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
        explicit_cost, explicit_diag = calc_explicit_trade_cost(
            trade_exec,
            commission_rate=args.commission_rate,
            stamp_tax_rate=args.stamp_tax_rate,
            slippage_rate=args.slippage_rate,
        )
        total_trade_cost = impact_cost + explicit_cost
        daily_costs[entry_day] += total_trade_cost
        active_from_day = active_day

        holdings = update_holding_states(
            all_codes,
            current_w,
            holdings,
            selected_codes,
            alpha_by_code,
            rank_by_code,
            runtime.price_dict,
            dt,
        )

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

        policy_sells = max(policy_evaluated - len(kept_codes), 0)
        diag["policy_evaluated"].append(float(policy_evaluated))
        diag["policy_keep_count"].append(float(len(kept_codes)))
        diag["policy_sell_count"].append(float(policy_sells))
        diag["missing_holding_count"].append(float(missing_holdings))
        diag["fill_count"].append(float(max(len(selected_codes) - len(kept_codes), 0)))
        diag["selected_count"].append(float(len(selected_codes)))
        diag["turnover"].append(float(np.sum(np.abs(trade_exec))))
        diag["buy_turnover"].append(explicit_diag["buy_turnover"])
        diag["sell_turnover"].append(explicit_diag["sell_turnover"])
        diag["gross"].append(float(np.sum(np.abs(current_w))))
        diag["impact_cost"].append(float(impact_cost))
        diag["explicit_cost"].append(float(explicit_cost))
        diag["explicit_commission_cost"].append(explicit_diag["explicit_commission_cost"])
        diag["explicit_stamp_tax_cost"].append(explicit_diag["explicit_stamp_tax_cost"])
        diag["explicit_slippage_cost"].append(explicit_diag["explicit_slippage_cost"])
        diag["total_trade_cost"].append(float(total_trade_cost))
        diag["market_mult"].append(float(market_mult))
        diag["avg_hold_prob"].append(float(np.mean([row[1] for row in keep_rows])) if keep_rows else np.nan)
        diag["fill_ratio"].append(float(np.mean(fill_ratio[active_trade])) if np.any(active_trade) else 0.0)

    if active_from_day is not None:
        for day in range(active_from_day, t_total):
            daily_weights[day] = current_w.copy()

    raw_ret = _compute_daily_portfolio_returns(daily_weights, ret_daily, daily_costs, t_total)
    idx_ret_arr = np.asarray(idx_daily.iloc[1:len(raw_ret) + 1].values, dtype=np.float64)
    neu_ret, _ = _rolling_beta_neutralize(raw_ret, idx_ret_arr)
    raw_ret, neu_ret = trim_returns_with_costs(raw_ret, neu_ret, daily_weights, daily_costs)

    ann_raw, sharpe_raw, mdd_raw = calc_metrics(raw_ret)
    ann_neu, sharpe_neu, mdd_neu = calc_metrics(neu_ret)
    row = {
        "mode": "trade_policy_v2_retention_first" if args.mode == "policy" else "daily_alpha_topk_baseline",
        "top_frac": args.top_frac,
        "sell_threshold": sell_threshold,
        "ann_raw": ann_raw,
        "sharpe_raw": sharpe_raw,
        "mdd_raw": mdd_raw,
        "ann_neu": ann_neu,
        "sharpe_neu": sharpe_neu,
        "mdd_neu": mdd_neu,
        "avg_turnover": float(np.nanmean(diag["turnover"])) if diag["turnover"] else 0.0,
        "avg_gross": float(np.nanmean(diag["gross"])) if diag["gross"] else 0.0,
        "avg_policy_evaluated": float(np.nanmean(diag["policy_evaluated"])) if diag["policy_evaluated"] else 0.0,
        "avg_policy_keep": float(np.nanmean(diag["policy_keep_count"])) if diag["policy_keep_count"] else 0.0,
        "avg_policy_sell": float(np.nanmean(diag["policy_sell_count"])) if diag["policy_sell_count"] else 0.0,
        "avg_fill_count": float(np.nanmean(diag["fill_count"])) if diag["fill_count"] else 0.0,
        "avg_selected_count": float(np.nanmean(diag["selected_count"])) if diag["selected_count"] else 0.0,
        "avg_fill_ratio": float(np.nanmean(diag["fill_ratio"])) if diag["fill_ratio"] else 0.0,
        "total_impact_cost": float(np.nansum(diag["impact_cost"])) if diag["impact_cost"] else 0.0,
        "total_explicit_cost": float(np.nansum(diag["explicit_cost"])) if diag["explicit_cost"] else 0.0,
        "total_commission_cost": float(np.nansum(diag["explicit_commission_cost"])) if diag["explicit_commission_cost"] else 0.0,
        "total_stamp_tax_cost": float(np.nansum(diag["explicit_stamp_tax_cost"])) if diag["explicit_stamp_tax_cost"] else 0.0,
        "total_slippage_cost": float(np.nansum(diag["explicit_slippage_cost"])) if diag["explicit_slippage_cost"] else 0.0,
        "total_trade_cost": float(np.nansum(diag["total_trade_cost"])) if diag["total_trade_cost"] else 0.0,
    }
    diag_df = pd.DataFrame({k: pd.Series(v) for k, v in diag.items()})
    return row, raw_ret, neu_ret, diag_df


def main():
    parser = argparse.ArgumentParser(description="Backtest daily trade policy v2 retention-first.")
    parser.add_argument("--checkpoint", default="checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt")
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--output-dir", default="backtest_results_trade_policy_v2_20260530")
    parser.add_argument("--top-frac", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--hist-window", type=int, default=60)
    parser.add_argument("--alpha-window", type=int, default=3)
    parser.add_argument("--market-timing", choices=["legacy", "none"], default="legacy")
    parser.add_argument("--adv-limit-ratio", type=float, default=0.02)
    parser.add_argument("--impact-coeff", type=float, default=0.1)
    parser.add_argument("--commission-rate", type=float, default=0.0001, help="Two-sided commission rate per traded notional.")
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005, help="Sell-side stamp tax rate.")
    parser.add_argument("--slippage-rate", type=float, default=0.0005, help="Two-sided slippage/spread rate per traded notional.")
    parser.add_argument("--portfolio-value", type=float, default=1e8)
    parser.add_argument("--sell-threshold", type=float, default=None)
    parser.add_argument("--mode", choices=["policy", "alpha_baseline"], default="policy")
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    row, raw_ret, neu_ret, diag_df = run_policy_backtest_v2(args)
    save_summary_csv([row], out_dir / "trade_policy_v2_summary.csv", display_columns=list(row.keys()), title="Trade policy v2")
    pd.DataFrame({"daily_return": raw_ret, "neutral_return": neu_ret}).to_csv(out_dir / "trade_policy_v2_returns.csv", index=False)
    diag_df.to_csv(out_dir / "trade_policy_v2_diagnostics.csv", index=False)
    (out_dir / "trade_policy_v2_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
