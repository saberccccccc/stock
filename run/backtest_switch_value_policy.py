#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backtest a retention-first switch-value policy.

Existing holdings are kept by default. A holding A is replaced by candidate B
only when the learned value model predicts positive net switch edge. Candidate
pairs are selected greedily by predicted value with one-to-one A/B matching.
"""
import argparse
import json
import os
import pickle
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
from backtest.predictors import PersistentPredictor
from backtest.reports import save_summary_csv
from backtest.runtime import build_v9_backtest_config, load_backtest_runtime, load_dl_predictor
from data.pipeline import build_cross_section_dataset, samples_from_precomputed_metadata
from run.backtest_trade_policy import legacy_market_mult, load_index_series
from run.backtest_trade_policy_v2 import (
    build_equal_weight_target,
    calc_explicit_trade_cost,
    trim_returns_with_costs,
)
from run.build_switch_value_dataset import (
    alpha_history_features,
    estimate_switch_cost,
    rank_pct_desc,
)
from run.v9_cache_utils import (
    load_alpha_rows_jsonl,
    load_universe_matrix_cache,
    save_alpha_rows_jsonl,
    save_universe_matrix_cache,
)
from run.v9_long_only_optimization import V9RankPredictor


def load_sample_splits(cfg, use_cache=True):
    result = build_cross_section_dataset(cfg, use_cache=use_cache)
    if isinstance(result, dict):
        cfg.low_feat_dim = result.get("low_agg_dim", getattr(cfg, "low_feat_dim", 14))
        return samples_from_precomputed_metadata(result, "train"), samples_from_precomputed_metadata(result, "val")
    return result


def resolve_metadata_cache(path=None):
    if path:
        return Path(path)
    cache_dir = PROJECT_ROOT / "cache"
    patterns = [
        "cross_section_v13_config_key_tech_market_funda_shareh_restr_macro_all_s40_t5_h10_min30*_meta.pkl",
        "cross_section*_s40_t5_h10_min30*_meta.pkl",
    ]
    for pattern in patterns:
        matches = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No cross-section metadata cache found under {cache_dir}")


def load_lightweight_samples_from_alpha_cache(alpha_rows, split, limit=None, metadata_cache=None):
    meta_path = resolve_metadata_cache(metadata_cache)
    with meta_path.open("rb") as f:
        meta = pickle.load(f)

    all_codes = np.asarray(meta["all_codes"]).astype(str)
    all_dates = pd.DatetimeIndex(pd.to_datetime(meta["all_dates"]))
    code_to_idx = {code: i for i, code in enumerate(all_codes)}
    date_to_idx = {pd.Timestamp(dt): i for i, dt in enumerate(all_dates)}
    industry_array = meta["industry_array"]
    n_stocks = len(all_codes)
    n_dates = len(all_dates)
    risk_mm = np.memmap(
        meta["risk_full_path"],
        dtype=np.int16,
        mode="r",
        shape=(n_stocks, n_dates, meta["risk_full_dim"]),
    )

    start, end = split_bounds(split)
    samples = []
    for row in alpha_rows:
        dt = pd.Timestamp(row["date"])
        if start is not None and dt < start:
            continue
        if end is not None and dt >= end:
            continue
        t = date_to_idx.get(dt)
        if t is None:
            continue
        codes = [str(c) for c in row["codes"] if str(c) in code_to_idx]
        if not codes:
            continue
        idx = np.asarray([code_to_idx[c] for c in codes], dtype=np.int64)
        samples.append({
            "date": dt,
            "codes": codes,
            "risk": risk_mm[idx, t, :].astype(np.float32) / 1000.0,
            "industry_ids": industry_array[idx, t].astype(np.int64),
        })
        if limit is not None and len(samples) >= int(limit):
            break
    print(f"Loaded lightweight alpha-cache samples: {len(samples)} | meta={meta_path}", flush=True)
    return samples


def split_bounds(split):
    if split == "val":
        return pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
    if split == "test":
        return pd.Timestamp("2025-01-01"), None
    return None, None


def filter_samples_by_split(samples, split, limit=None):
    start, end = split_bounds(split)
    out = []
    for sample in samples:
        dt = pd.Timestamp(sample["date"])
        if start is not None and dt < start:
            continue
        if end is not None and dt >= end:
            continue
        out.append(sample)
    out.sort(key=lambda s: pd.Timestamp(s["date"]))
    if limit is not None:
        out = out[: int(limit)]
    return out


def strip_empty_prefix(feats):
    return {k[1:] if k.startswith("_") else k: v for k, v in feats.items()}


def align_model_features(rows, feature_cols):
    if not rows:
        return pd.DataFrame(columns=feature_cols)
    x = pd.DataFrame(rows)
    x = pd.get_dummies(x, columns=[c for c in x.columns if x[c].dtype == object], dummy_na=True)
    return x.reindex(columns=feature_cols, fill_value=0.0)


class MatrixFeatureProvider:
    def __init__(self, price_mat, vol_mat, all_dates, code2idx):
        self.price_mat = price_mat
        self.vol_mat = vol_mat
        self.all_dates = all_dates
        self.code2idx = code2idx

    def date_col(self, dt):
        pos = self.all_dates.searchsorted(pd.Timestamp(dt), side="right") - 1
        return int(pos) if pos >= 0 else -1

    def price(self, code, dt):
        i = self.code2idx.get(code)
        pos = self.date_col(dt)
        if i is None or pos < 0:
            return np.nan
        return float(self.price_mat[i, pos])

    def volume(self, code, dt):
        if self.vol_mat is None:
            return np.nan
        i = self.code2idx.get(code)
        pos = self.date_col(dt)
        if i is None or pos < 0:
            return np.nan
        return float(self.vol_mat[i, pos])

    def ret_features(self, code, dt):
        out = {
            "ret_1d": np.nan,
            "ret_3d": np.nan,
            "ret_5d": np.nan,
            "vol_10d": np.nan,
            "vol_20d": np.nan,
            "drawdown_20d": np.nan,
        }
        i = self.code2idx.get(code)
        pos = self.date_col(dt)
        if i is None or pos <= 0:
            return out
        close = self.price_mat[i]
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
        return out

    def execution_features(self, code, dt, position_weight, portfolio_value):
        px = self.price(code, dt)
        vol = self.volume(code, dt)
        dollar_vol = px * vol * 100.0 if np.isfinite(px) and px > 0 and np.isfinite(vol) and vol > 0 else 0.0
        trade_notional = float(position_weight) * float(portfolio_value)
        adv_trade_ratio = trade_notional / dollar_vol if dollar_vol > 0 else np.nan
        close_ret = np.nan
        i = self.code2idx.get(code)
        pos = self.date_col(dt)
        if i is not None and pos > 0:
            prev = self.price_mat[i, pos - 1]
            cur = self.price_mat[i, pos]
            if np.isfinite(prev) and prev > 0 and np.isfinite(cur):
                close_ret = float(cur / prev - 1.0)
        return {
            "dollar_vol": float(dollar_vol) if dollar_vol > 0 else np.nan,
            "adv_trade_ratio": float(adv_trade_ratio) if np.isfinite(adv_trade_ratio) else np.nan,
            "close_ret_1d": float(close_ret) if np.isfinite(close_ret) else np.nan,
            "near_limit_up": float(close_ret >= 0.095) if np.isfinite(close_ret) else np.nan,
            "near_limit_down": float(close_ret <= -0.095) if np.isfinite(close_ret) else np.nan,
        }


def update_holding_states_from_provider(
    all_codes,
    current_w,
    old_holdings,
    selected_codes,
    alpha_by_code,
    rank_by_code,
    feature_provider,
    dt,
):
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
                "entry_price": feature_provider.price(code, dt),
                "holding_days": 1,
            }
    return next_holdings


def make_switch_feature_row(
    a_code,
    b_code,
    codes,
    code_to_i,
    alpha,
    rank,
    industry_ids,
    alpha_hist,
    rank_hist,
    alpha_ma,
    holdings,
    feature_provider,
    dt,
    regime,
    position_weight,
    portfolio_value,
    cost_args,
):
    a_i = code_to_i[a_code]
    b_i = code_to_i[b_code]
    state = holdings.get(a_code, {})

    row = {
        "market_regime": regime,
        "same_industry": int(industry_ids[a_i] == industry_ids[b_i]),
        "A_holding_days": int(state.get("holding_days", 0)),
        "A_entry_rank_pct": float(state.get("entry_rank_pct", np.nan)),
        "A_entry_alpha": float(state.get("entry_alpha", np.nan)),
    }
    a_alpha = strip_empty_prefix(
        alpha_history_features(a_code, a_i, alpha, rank, alpha_hist, rank_hist, alpha_ma, "")
    )
    b_alpha = strip_empty_prefix(
        alpha_history_features(b_code, b_i, alpha, rank, alpha_hist, rank_hist, alpha_ma, "")
    )
    a_ret = feature_provider.ret_features(a_code, dt)
    b_ret = feature_provider.ret_features(b_code, dt)
    a_exec = feature_provider.execution_features(a_code, dt, position_weight, portfolio_value)
    b_exec = feature_provider.execution_features(b_code, dt, position_weight, portfolio_value)

    row.update({f"A_{k}": v for k, v in a_alpha.items()})
    row.update({f"B_{k}": v for k, v in b_alpha.items()})
    row.update({f"A_{k}": v for k, v in a_ret.items()})
    row.update({f"B_{k}": v for k, v in b_ret.items()})
    row.update({f"A_{k}": v for k, v in a_exec.items()})
    row.update({f"B_{k}": v for k, v in b_exec.items()})

    a_px = feature_provider.price(a_code, dt)
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
    row.update(estimate_switch_cost(row, cost_args))
    return row


def select_switches(
    holdings,
    codes,
    order,
    code_to_i,
    alpha,
    rank,
    industry_ids,
    alpha_hist,
    rank_hist,
    alpha_ma,
    feature_provider,
    dt,
    regime,
    position_weight,
    portfolio_value,
    model_bundle,
    candidate_frac,
    max_pairs_per_holding,
    cost_args,
):
    feature_cols = model_bundle["feature_cols"]
    model = model_bundle["model"]
    holding_set = {code for code in holdings if code in code_to_i}
    k_candidate = max(1, int(len(codes) * candidate_frac))
    candidate_codes = [codes[int(i)] for i in order[:k_candidate] if codes[int(i)] not in holding_set]
    if not candidate_codes or not holding_set:
        return [], {"candidate_pairs": 0, "positive_pairs": 0, "avg_pred_switch": np.nan}

    pair_meta = []
    rows = []
    for a_code in list(holding_set):
        a_i = code_to_i[a_code]
        selected = []
        if candidate_codes:
            selected.append(candidate_codes[0])
        better = [c for c in candidate_codes if alpha[code_to_i[c]] > alpha[a_i]]
        if better:
            selected.append(min(better, key=lambda c: alpha[code_to_i[c]] - alpha[a_i]))
        same = [c for c in candidate_codes if int(industry_ids[code_to_i[c]]) == int(industry_ids[a_i])]
        if same:
            selected.append(same[0])
        for c in candidate_codes:
            if len(selected) >= max_pairs_per_holding:
                break
            selected.append(c)

        seen = set()
        for b_code in selected:
            if b_code in seen:
                continue
            seen.add(b_code)
            pair_meta.append((a_code, b_code))
            rows.append(
                make_switch_feature_row(
                    a_code,
                    b_code,
                    codes,
                    code_to_i,
                    alpha,
                    rank,
                    industry_ids,
                    alpha_hist,
                    rank_hist,
                    alpha_ma,
                    holdings,
                    feature_provider,
                    dt,
                    regime,
                    position_weight,
                    portfolio_value,
                    cost_args,
                )
            )

    x = align_model_features(rows, feature_cols)
    pred = np.asarray(model.predict(x), dtype=np.float64) if len(x) else np.array([], dtype=np.float64)
    scored = [
        (float(p), a_code, b_code)
        for (a_code, b_code), p in zip(pair_meta, pred)
        if np.isfinite(p) and p > 0.0
    ]
    scored.sort(reverse=True, key=lambda x: x[0])

    used_a = set()
    used_b = set()
    switches = []
    for p, a_code, b_code in scored:
        if a_code in used_a or b_code in used_b:
            continue
        switches.append((a_code, b_code, p))
        used_a.add(a_code)
        used_b.add(b_code)
    return switches, {
        "candidate_pairs": len(pair_meta),
        "positive_pairs": len(scored),
        "avg_pred_switch": float(np.mean(pred)) if len(pred) else np.nan,
    }


def run_switch_value_backtest(args):
    cfg = build_v9_backtest_config()
    can_skip_runtime_prices = bool(args.alpha_cache and args.matrix_cache and Path(args.matrix_cache).exists())
    if can_skip_runtime_prices:
        train_samples, val_samples = load_sample_splits(cfg, use_cache=True)
        runtime = None
    else:
        runtime = load_backtest_runtime(cfg, use_cache=True)
        train_samples, val_samples = runtime.train, runtime.val
    predictor_name = "cache"
    predictor = None
    if not args.alpha_cache:
        base = load_dl_predictor(args.checkpoint, train_samples, cfg)
        raw_predictor = V9RankPredictor(base, "v9_raw", cache={})
        if args.predictor_mode == "none":
            predictor = raw_predictor
        else:
            predictor = PersistentPredictor(raw_predictor, window=args.window, mode=args.predictor_mode)
        predictor_name = predictor.name
    model_bundle = joblib.load(args.switch_model)

    all_samples_for_eval = list(train_samples) + list(val_samples)
    split_samples = filter_samples_by_split(all_samples_for_eval, args.split, args.limit_val)
    if not split_samples:
        raise ValueError(f"No samples selected for split={args.split}")

    if args.alpha_cache:
        alpha_rows_cache = load_alpha_rows_jsonl(args.alpha_cache)
        alpha_by_date = {pd.Timestamp(row["date"]): row for row in alpha_rows_cache}
    else:
        alpha_by_date = {}

    if args.matrix_cache and Path(args.matrix_cache).exists():
        price_mat, vol_mat, all_dates, code2idx, all_codes = load_universe_matrix_cache(args.matrix_cache)
        print(f"Loaded matrix cache: {args.matrix_cache} | codes={len(all_codes)} dates={len(all_dates)}", flush=True)
    else:
        if runtime is None:
            raise ValueError("--matrix-cache must exist when --alpha-cache is used without full runtime prices")
        if args.alpha_cache:
            all_codes = sorted(set(c for row in alpha_rows_cache for c in row["codes"]))
        else:
            all_codes = sorted(set(c for s in split_samples for c in s["codes"]))
        price_mat, vol_mat, all_dates, code2idx = build_universe_matrix(runtime.price_dict, runtime.vol_dict, all_codes)
        if args.matrix_cache:
            save_universe_matrix_cache(args.matrix_cache, price_mat, vol_mat, all_dates, all_codes)
            print(f"Saved matrix cache: {args.matrix_cache}", flush=True)
    feature_provider = MatrixFeatureProvider(price_mat, vol_mat, all_dates, code2idx)
    idx_close, idx_daily = load_index_series(cfg, all_dates)
    t_total = len(all_dates)
    ret_daily = np.full((len(all_codes), t_total - 1), np.nan)
    for i in range(len(all_codes)):
        p = price_mat[i]
        ret_daily[i] = p[1:] / p[:-1] - 1.0
    ret_daily[~np.isfinite(ret_daily)] = 0.0

    date2idx = {}
    for sample in split_samples:
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
    generated_alpha_rows = []

    samples = split_samples
    for sample_no, sample in enumerate(samples, start=1):
        if args.progress_every and sample_no % args.progress_every == 0:
            print(f"[switch] {sample_no}/{len(samples)} days", flush=True)
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
        sample_matrix_pairs = [(i, code2idx[c]) for i, c in enumerate(codes_all) if c in code2idx]
        valid = np.zeros(len(codes_all), dtype=bool)
        if sample_matrix_pairs:
            sample_i = np.asarray([p[0] for p in sample_matrix_pairs], dtype=np.int64)
            matrix_i = np.asarray([p[1] for p in sample_matrix_pairs], dtype=np.int64)
            price_hist = price_mat[matrix_i, col_cur - args.hist_window:col_cur]
            valid[sample_i] = np.sum(~np.isnan(price_hist), axis=1) >= 0.7 * args.hist_window
        if not np.any(valid):
            continue

        regime = detect_regime(sample)
        if args.alpha_cache:
            alpha_row = alpha_by_date.get(dt)
            if alpha_row is None:
                continue
            sample_code_to_i = {code: i for i, code in enumerate(codes_all)}
            cached_codes = []
            cached_alpha = []
            cached_industry = []
            for code, value in zip(alpha_row["codes"], alpha_row["alpha"]):
                i = sample_code_to_i.get(code)
                if i is None or not valid[i]:
                    continue
                cached_codes.append(code)
                cached_alpha.append(float(value))
                cached_industry.append(int(sample["industry_ids"][i]))
            if len(cached_codes) < 2:
                continue
            codes = cached_codes
            alpha = np.asarray(cached_alpha, dtype=np.float64)
            industry_ids = np.asarray(cached_industry)
        else:
            raw_alpha = np.asarray(predictor.predict_alpha(sample, valid, regime), dtype=np.float64)
            codes_valid = [codes_all[i] for i in range(len(codes_all)) if valid[i]]
            if len(raw_alpha) != len(codes_valid):
                continue
            finite = np.isfinite(raw_alpha)
            if np.count_nonzero(finite) < 2:
                continue
            codes = [codes_valid[i] for i in np.flatnonzero(finite)]
            alpha = raw_alpha[finite]
            industry_ids = sample["industry_ids"][valid][finite]

        rank = rank_pct_desc(alpha)
        code_to_i = {code: i for i, code in enumerate(codes)}
        alpha_by_code = {code: float(alpha[i]) for code, i in code_to_i.items()}
        rank_by_code = {code: float(rank[i]) for code, i in code_to_i.items()}
        order = np.arange(len(alpha)) if args.alpha_cache else np.argsort(alpha)[::-1]
        if args.write_alpha_cache and not args.alpha_cache:
            generated_alpha_rows.append({
                "date": dt,
                "codes": [codes[int(i)] for i in order],
                "alpha": [float(alpha[int(i)]) for i in order],
                "n_stocks": int(len(codes)),
            })
        k = max(1, int(len(codes) * args.top_frac))
        position_weight = min(args.max_weight, 1.0 / max(k, 1))

        selected_set = {code for code in holdings if code in code_to_i}
        switches = []
        switch_stats = {"candidate_pairs": 0, "positive_pairs": 0, "avg_pred_switch": np.nan}
        if args.mode == "switch_value":
            switches, switch_stats = select_switches(
                holdings,
                codes,
                order,
                code_to_i,
                alpha,
                rank,
                industry_ids,
                alpha_hist,
                rank_hist,
                alpha_ma,
                feature_provider,
                dt,
                regime,
                position_weight,
                args.portfolio_value,
                model_bundle,
                args.candidate_frac,
                args.max_pairs_per_holding,
                args,
            )
            for a_code, b_code, _ in switches:
                selected_set.discard(a_code)
                selected_set.add(b_code)
        else:
            selected_set = set()

        for i in order:
            code = codes[int(i)]
            if code in selected_set:
                continue
            selected_set.add(code)
            if len(selected_set) >= k:
                break
        selected_codes = sorted(selected_set, key=lambda c: rank_by_code.get(c, 1.0))[:k]

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

        holdings = update_holding_states_from_provider(
            all_codes,
            current_w,
            holdings,
            selected_codes,
            alpha_by_code,
            rank_by_code,
            feature_provider,
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

        active_holding_days = [
            float(v.get("holding_days", np.nan))
            for v in holdings.values()
            if np.isfinite(v.get("holding_days", np.nan))
        ]
        diag["candidate_pairs"].append(float(switch_stats["candidate_pairs"]))
        diag["positive_pairs"].append(float(switch_stats["positive_pairs"]))
        diag["switch_count"].append(float(len(switches)))
        diag["avg_pred_switch"].append(float(switch_stats["avg_pred_switch"]))
        diag["selected_count"].append(float(len(selected_codes)))
        diag["avg_holding_days"].append(float(np.mean(active_holding_days)) if active_holding_days else np.nan)
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
        diag["fill_ratio"].append(float(np.mean(fill_ratio[active_trade])) if np.any(active_trade) else 0.0)

    if active_from_day is not None:
        for day in range(active_from_day, t_total):
            daily_weights[day] = current_w.copy()

    if args.write_alpha_cache and generated_alpha_rows:
        save_alpha_rows_jsonl(generated_alpha_rows, args.write_alpha_cache)
        print(f"Saved alpha cache: {args.write_alpha_cache}", flush=True)

    raw_ret = _compute_daily_portfolio_returns(daily_weights, ret_daily, daily_costs, t_total)
    idx_ret_arr = np.asarray(idx_daily.iloc[1:len(raw_ret) + 1].values, dtype=np.float64)
    neu_ret, _ = _rolling_beta_neutralize(raw_ret, idx_ret_arr)
    raw_ret, neu_ret = trim_returns_with_costs(raw_ret, neu_ret, daily_weights, daily_costs)

    ann_raw, sharpe_raw, mdd_raw = calc_metrics(raw_ret)
    ann_neu, sharpe_neu, mdd_neu = calc_metrics(neu_ret)
    row = {
        "split": args.split,
        "mode": "switch_value_retention_first" if args.mode == "switch_value" else "daily_alpha_topk_baseline",
        "predictor": predictor_name,
        "window": args.window if args.predictor_mode != "none" and not args.alpha_cache else 1,
        "top_frac": args.top_frac,
        "ann_raw": ann_raw,
        "sharpe_raw": sharpe_raw,
        "mdd_raw": mdd_raw,
        "ann_neu": ann_neu,
        "sharpe_neu": sharpe_neu,
        "mdd_neu": mdd_neu,
        "avg_turnover": float(np.nanmean(diag["turnover"])) if diag["turnover"] else 0.0,
        "avg_gross": float(np.nanmean(diag["gross"])) if diag["gross"] else 0.0,
        "avg_switch_count": float(np.nanmean(diag["switch_count"])) if diag["switch_count"] else 0.0,
        "avg_candidate_pairs": float(np.nanmean(diag["candidate_pairs"])) if diag["candidate_pairs"] else 0.0,
        "avg_positive_pairs": float(np.nanmean(diag["positive_pairs"])) if diag["positive_pairs"] else 0.0,
        "avg_holding_days": float(np.nanmean(diag["avg_holding_days"])) if diag["avg_holding_days"] else 0.0,
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
    parser = argparse.ArgumentParser(description="Backtest switch value retention-first policy.")
    parser.add_argument("--checkpoint", default="checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt")
    parser.add_argument("--switch-model", default="switch_value_models_20260530_fixed/switch_edge_lgb_h5/switch_value_model.pkl")
    parser.add_argument("--output-dir", default="backtest_results_switch_value_20260530")
    parser.add_argument("--split", choices=["val", "test", "valtest"], default="val")
    parser.add_argument("--top-frac", type=float, default=0.05)
    parser.add_argument("--candidate-frac", type=float, default=0.10)
    parser.add_argument("--predictor-mode", default="none", choices=["none", "average", "momentum", "composite"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--alpha-cache", default=None, help="Optional cached V9 alpha jsonl.")
    parser.add_argument("--write-alpha-cache", default=None, help="Write computed V9 alpha rows to this jsonl.")
    parser.add_argument("--matrix-cache", default=None, help="Optional price/volume matrix npz cache.")
    parser.add_argument("--max-pairs-per-holding", type=int, default=4)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--hist-window", type=int, default=60)
    parser.add_argument("--alpha-window", type=int, default=3)
    parser.add_argument("--market-timing", choices=["legacy", "none"], default="legacy")
    parser.add_argument("--adv-limit-ratio", type=float, default=0.02)
    parser.add_argument("--impact-coeff", type=float, default=0.1)
    parser.add_argument("--commission-rate", type=float, default=0.0001)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--portfolio-value", type=float, default=1e8)
    parser.add_argument("--mode", choices=["switch_value", "alpha_baseline"], default="switch_value")
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    row, raw_ret, neu_ret, diag_df = run_switch_value_backtest(args)
    save_summary_csv([row], out_dir / "switch_value_summary.csv", display_columns=list(row.keys()), title="Switch value policy")
    pd.DataFrame({"daily_return": raw_ret, "neutral_return": neu_ret}).to_csv(
        out_dir / "switch_value_returns.csv",
        index=False,
    )
    diag_df.to_csv(out_dir / "switch_value_diagnostics.csv", index=False)
    (out_dir / "switch_value_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
