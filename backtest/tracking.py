# -*- coding: utf-8 -*-
"""Reusable tracking helpers for saved backtest holdings."""

import os
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import calc_extended_metrics, load_price_volume
from core.config import DataConfig
from data.api_utils import SafeAPICaller, resolve_tushare_token

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CANDIDATE_PATTERNS = [
    "backtest_results_concentrated/v9_gat_avg_score_simple_ls_run_top_3pct_*_full_data.pkl",
    "backtest_results_concentrated/v9_gat_top_union_bottom_intersection_simple_ls_run_top_3pct_*_full_data.pkl",
    "backtest_results_concentrated/v9_gat_avg_score_simple_ls_top2pct_*_full_data.pkl",
    "backtest_results_concentrated/v9_gat_top_union_bottom_intersection_simple_ls_top2pct_*_full_data.pkl",
    "backtest_results_concentrated/v9_gat_avg_score_simple_ls_top1pct_*_full_data.pkl",
    "backtest_results_concentrated/v9_gat_top_union_bottom_intersection_simple_ls_top1pct_*_full_data.pkl",
    "backtest_results_concentrated/v9_gat_avg_score_optimizer_projected_run_top_3pct_*_full_data.pkl",
    "backtest_results_concentrated/v9_gat_top_union_bottom_intersection_optimizer_projected_run_top_3pct_*_full_data.pkl",
    "backtest_results_concentrated/v9_gat_avg_score_optimizer_projected_top2pct_*_full_data.pkl",
    "backtest_results_concentrated/v9_gat_top_union_bottom_intersection_optimizer_projected_top2pct_*_full_data.pkl",
    "backtest_results_concentrated/v9_gat_top_union_bottom_intersection_simple_ls_top10pct_*_full_data.pkl",
    "backtest_results_intersection/v9_gat_intersection_optimizer_*_full_data.pkl",
    "backtest_results_intersection/v9_gat_intersection_simple_ls_*_full_data.pkl",
    "backtest_results_ensemble/v9_gat_avg_score_simple_ls_*_full_data.pkl",
    "backtest_results_ensemble/v9_gat_top_union_bottom_intersection_simple_ls_*_full_data.pkl",
]

STRATEGY_NAME_REPLACEMENTS = [
    ("v9_gat_top_union_bottom_intersection", "top_union"),
    ("v9_gat_avg_score", "avg_score"),
    ("v9_gat_intersection", "intersection"),
    ("optimizer_projected", "optproj"),
    ("run_top_", "top"),
    ("pct", ""),
]

_tracking_api_call = SafeAPICaller(
    min_interval=1.5,
    max_retries=3,
    retry_base_delay=4.0,
    jitter=None,
    data_source="tushare",
)


def resolve_project_path(path, project_root=PROJECT_ROOT):
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def all_full_data_paths(project_root=PROJECT_ROOT):
    result_dirs = [
        "backtest_results",
        "backtest_results_layered",
        "backtest_results_intersection",
        "backtest_results_ensemble",
    ]
    candidates = []
    for dirname in result_dirs:
        candidates.extend((project_root / dirname).glob("*_full_data.pkl"))
    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def latest_full_data_path(project_root=PROJECT_ROOT):
    candidates = all_full_data_paths(project_root)
    if not candidates:
        raise FileNotFoundError("未找到 *_full_data.pkl，请先运行一次回测并保存结果")
    return candidates[-1]


def best_full_data_path(project_root=PROJECT_ROOT):
    preferred = sorted(
        (project_root / "backtest_results_ensemble").glob(
            "v9_gat_top_union_bottom_intersection_simple_ls_*_full_data.pkl"
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if preferred:
        return preferred[-1]
    return latest_full_data_path(project_root)


def latest_matching_path(pattern, project_root=PROJECT_ROOT):
    matches = sorted(project_root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def default_candidate_paths(project_root=PROJECT_ROOT):
    paths = []
    seen = set()
    for pattern in DEFAULT_CANDIDATE_PATTERNS:
        path = latest_matching_path(pattern, project_root)
        if path is not None and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def resolve_candidate_paths(args, project_root=PROJECT_ROOT):
    paths = []
    seen = set()

    def add_path(path):
        path = Path(path)
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)

    for item in getattr(args, "candidate_pkl", []):
        add_path(item)
    for pattern in getattr(args, "candidate_glob", []):
        for path in sorted(project_root.glob(pattern), key=lambda p: p.stat().st_mtime):
            add_path(path)
    if not paths:
        for path in default_candidate_paths(project_root):
            add_path(path)
    if not paths:
        raise FileNotFoundError("未找到候选 *_full_data.pkl")
    return paths


def strategy_label(path):
    stem = Path(path).name.replace("_full_data.pkl", "")
    for old, new in STRATEGY_NAME_REPLACEMENTS:
        stem = stem.replace(old, new)
    parts = stem.split("_")
    if len(parts) > 2 and parts[-2].isdigit() and parts[-1].isdigit():
        stem = "_".join(parts[:-2])
    return stem


def load_backtest_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    required = ["daily_weights", "dates", "codes"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path} 缺少字段: {missing}")
    return data


def load_prices(data_dir):
    cfg = DataConfig()
    cfg.data_dir = str(data_dir)
    price_dict, _ = load_price_volume(cfg)
    return price_dict


def safe_tushare_call(func, **kwargs):
    return _tracking_api_call(func, **kwargs)


def read_tracking_file(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "trade_date"
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def update_tracking_prices(codes, tracking_dir, hold_date, token=None, project_root=PROJECT_ROOT):
    tracking_dir = resolve_project_path(tracking_dir, project_root)
    raw_dir = (project_root / "data/raw").resolve()
    if tracking_dir.resolve() == raw_dir:
        raise ValueError("拒绝写入 data/raw：tracking 更新只能写入 data/tracking_raw 或其他隔离目录")
    tracking_dir.mkdir(parents=True, exist_ok=True)
    codes = sorted(set(codes))
    if not codes:
        return

    hold_next = pd.Timestamp(hold_date).normalize() + timedelta(days=1)
    end_date = pd.Timestamp.today().normalize()
    starts = {}
    for code in codes:
        existing = read_tracking_file(tracking_dir / f"{code}.csv")
        if existing.empty:
            starts[code] = hold_next
        else:
            starts[code] = max(hold_next, existing.index.max().normalize() + timedelta(days=1))

    min_start = min(starts.values())
    if min_start > end_date:
        print("tracking_raw 已覆盖当前日期，无需更新。")
        return

    import tushare as ts

    ts.set_token(resolve_tushare_token(token, context="tracking update"))
    pro = ts.pro_api()
    start_str = min_start.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    cal = safe_tushare_call(
        pro.trade_cal,
        exchange="",
        start_date=start_str,
        end_date=end_str,
        is_open="1",
        fields="cal_date",
    )
    if cal is None or cal.empty:
        print(f"没有可更新的交易日: {start_str}~{end_str}")
        return

    trade_dates = sorted(cal["cal_date"].astype(str).tolist())
    rows_by_code = {code: [] for code in codes}
    code_set = set(codes)
    print(f"自动更新 tracking_raw: {len(codes)} 只持仓股票，交易日 {trade_dates[0]}~{trade_dates[-1]}")

    for i, trade_date in enumerate(trade_dates, 1):
        daily = safe_tushare_call(
            pro.daily,
            trade_date=trade_date,
            fields="ts_code,trade_date,open,high,low,close,vol,amount",
        )
        if daily is None or daily.empty:
            continue
        daily = daily[daily["ts_code"].isin(code_set)]
        if daily.empty:
            continue
        for row in daily.itertuples(index=False):
            code = row.ts_code
            dt = pd.to_datetime(str(row.trade_date))
            if dt < starts[code]:
                continue
            rows_by_code[code].append({
                "trade_date": dt,
                "code": code,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.vol,
                "money": row.amount,
                "factor": 1.0,
            })
        print(f"  tracking_raw 更新进度: {i}/{len(trade_dates)} {trade_date}")

    updated_files = 0
    updated_rows = 0
    for code, rows in rows_by_code.items():
        if not rows:
            continue
        new_df = pd.DataFrame(rows).set_index("trade_date").sort_index()
        path = tracking_dir / f"{code}.csv"
        old_df = read_tracking_file(path)
        combined = pd.concat([old_df, new_df]) if not old_df.empty else new_df
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined.index.name = "trade_date"
        combined.to_csv(path)
        updated_files += 1
        updated_rows += len(new_df)
    print(f"tracking_raw 更新完成: 文件 {updated_files} 个，新增/覆盖行 {updated_rows} 行")


def last_nonzero_weights(data):
    codes = list(data["codes"])
    dates = pd.DatetimeIndex(data["dates"])
    weights = np.vstack([np.asarray(w, dtype=float) for w in data["daily_weights"]])
    leverage = np.abs(weights).sum(axis=1)
    active_idx = np.flatnonzero(leverage > 1e-10)
    if len(active_idx) == 0:
        raise ValueError("回测结果中没有非零持仓")
    idx = int(active_idx[-1])
    return dates[idx], pd.Series(weights[idx], index=codes).fillna(0.0)


def build_tracking_close(history_prices, tracking_prices, codes, hold_date, start_date, end_date):
    frames = []
    for code in codes:
        hist = history_prices.get(code)
        track = tracking_prices.get(code)
        parts = []
        if hist is not None:
            hist = hist[hist.index <= hold_date]
            if not hist.empty:
                parts.append(hist.iloc[[-1]])
        if track is not None:
            track = track[track.index > hold_date]
            if not track.empty:
                parts.append(track)
        if not parts:
            continue
        series = pd.concat(parts).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        frames.append(series.rename(code))

    if not frames:
        raise ValueError("没有可用的历史/跟踪行情。请先把日更数据写入 data/tracking_raw。")

    close = pd.concat(frames, axis=1).sort_index()
    base_candidates = close.index[close.index <= hold_date]
    if len(base_candidates) == 0:
        raise ValueError(f"没有找到持仓日 {hold_date.date()} 之前的基准收盘价")
    base_date = base_candidates.max()

    tracking_start = pd.Timestamp(start_date) if start_date else hold_date
    selected = (close.index == base_date) | (close.index > tracking_start)
    close = close.loc[selected]
    if end_date:
        close = close[close.index <= pd.Timestamp(end_date)]
    return close


def compute_tracking(close, weights):
    dates = close.dropna(how="all").index
    records = []
    nav = 1.0
    peak = 1.0
    active_weights = weights[np.abs(weights) > 1e-10]

    for i, dt in enumerate(dates):
        missing_count = 0
        missing_abs_weight = 0.0
        if i == 0:
            port_ret = 0.0
        else:
            prev_close = close.loc[dates[i - 1], active_weights.index]
            cur_close = close.loc[dt, active_weights.index]
            valid_price = prev_close.notna() & cur_close.notna() & (prev_close > 0)
            missing = ~valid_price
            missing_count = int(missing.sum())
            missing_abs_weight = float(np.abs(active_weights[missing]).sum())
            stock_ret = (cur_close / prev_close - 1.0).replace([np.inf, -np.inf], np.nan)
            stock_ret = stock_ret.where(valid_price, 0.0).fillna(0.0)
            port_ret = float((active_weights * stock_ret).sum())

        nav *= (1.0 + port_ret)
        peak = max(peak, nav)
        records.append({
            "date": dt,
            "portfolio_return": port_ret,
            "nav": nav,
            "drawdown": nav / peak - 1.0,
            "leverage": float(np.abs(active_weights).sum()),
            "long_exposure": float(active_weights[active_weights > 0].sum()),
            "short_exposure": float(active_weights[active_weights < 0].sum()),
            "n_positions": int(len(active_weights)),
            "missing_price_count": missing_count,
            "missing_abs_weight": missing_abs_weight,
        })

    daily = pd.DataFrame(records)
    if len(daily) <= 1:
        raise ValueError("tracking_raw 中还没有回测持仓日之后的有效收盘价，无法计算跟踪收益")
    return daily


def holdings_table(close, weights, capital, nav):
    active = weights[np.abs(weights) > 1e-10].sort_values(key=np.abs, ascending=False)
    entry_close = close.loc[close.index[0], active.index]
    latest_close = close.loc[close.index[-1], active.index]
    valid_entry = entry_close.notna() & (entry_close > 0)
    period_return = (latest_close / entry_close - 1.0).replace([np.inf, -np.inf], np.nan)
    period_return = period_return.where(valid_entry, np.nan)
    estimated_notional = capital * nav * active
    table = pd.DataFrame({
        "code": active.index,
        "side": np.where(active.values > 0, "LONG", "SHORT"),
        "weight": active.values,
        "estimated_notional": estimated_notional.values,
        "abs_notional": np.abs(estimated_notional.values),
        "entry_close": entry_close.values,
        "latest_close": latest_close.values,
        "period_return": period_return.values,
        "contribution_est": active.values * period_return.values,
    })
    return table.replace([np.inf, -np.inf], np.nan)


def parse_variants(raw_variants):
    variants = [item.strip() for item in raw_variants.split(",") if item.strip()]
    if not variants:
        raise ValueError("--variants 不能为空")
    for variant in variants:
        if variant in {"full", "long"}:
            continue
        if variant.startswith("top") and variant[3:].isdigit():
            continue
        raise ValueError(f"未知 variant: {variant}")
    return variants


def normalize_long_weights(weights, top_n=None):
    long_weights = weights[weights > 1e-10].sort_values(ascending=False)
    if top_n is not None:
        long_weights = long_weights.head(top_n)
    if long_weights.empty:
        return long_weights
    return long_weights / (long_weights.sum() + 1e-12)


def variant_weights(weights, variant):
    if variant == "full":
        return weights[np.abs(weights) > 1e-10]
    if variant == "long":
        return normalize_long_weights(weights)
    top_n = int(variant[3:])
    return normalize_long_weights(weights, top_n=top_n)


def limit_tracking_close(close, tracking_days):
    if tracking_days is None or tracking_days <= 0:
        return close
    dates = close.dropna(how="all").index
    if len(dates) <= tracking_days + 1:
        return close
    return close.loc[:dates[tracking_days]]


def batch_row(result_path, hold_date, close, weights, variant, window):
    daily = compute_tracking(close, weights)
    returns = daily["portfolio_return"].iloc[1:].values
    metrics = calc_extended_metrics(returns)
    nav = float(daily["nav"].iloc[-1])
    active = weights[np.abs(weights) > 1e-10]
    top_codes = active.sort_values(key=np.abs, ascending=False).head(10).index.tolist()
    return {
        "strategy": strategy_label(result_path),
        "variant": variant,
        "tracking_window": window,
        "hold_date": hold_date.date(),
        "start_date": daily["date"].iloc[1].date(),
        "end_date": daily["date"].iloc[-1].date(),
        "days": len(daily) - 1,
        "nav": nav,
        "return": nav - 1.0,
        "ann_return": metrics.get("ann_return", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "max_drawdown": daily["drawdown"].min(),
        "win_rate": metrics.get("win_rate", 0.0),
        "positions": int(len(active)),
        "leverage": float(np.abs(active).sum()),
        "long_exposure": float(active[active > 0].sum()),
        "short_exposure": float(active[active < 0].sum()),
        "avg_missing_abs_weight": float(daily["missing_abs_weight"].iloc[1:].mean()),
        "latest_missing_abs_weight": float(daily["missing_abs_weight"].iloc[-1]),
        "top_codes": ",".join(top_codes),
        "result_path": str(Path(result_path).relative_to(PROJECT_ROOT)),
    }
