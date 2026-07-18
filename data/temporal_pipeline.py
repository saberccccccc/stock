"""Temporal cross-section dataset builder.

This module is intentionally a sidecar to data.pipeline. The existing V7/V9
pipeline keeps working unchanged; this builder adds a true per-stock time
sequence memmap for temporal-tower experiments.
"""

import hashlib
import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from data.market_features import N_MARKET, build_market_features_index_only, compute_breadth_from_close_matrix
from core.research_protocol import assert_research_end_date, cached_dates_within_research
from data.labels import PHYSICAL_LABEL_FAMILIES
from data.pipeline import (
    CACHE_VERSION,
    INDUSTRY_REL_FEATURES,
    MACRO_COLS,
    N_AGGS,
    N_STOCK_RISK,
    TECH_FEATURES,
    _compute_base_features,
    _load_extra_features,
    _load_industry_map,
    _precompute_all,
    add_technical_features,
)

warnings.filterwarnings("ignore")

SCALE = 1000
SENTINEL = np.int16(-32768)
TEMPORAL_CACHE_VERSION = "temporal_v1"


def _as_timestamp(value):
    if value is None or value == "":
        return None
    return pd.Timestamp(value)


def _date_token(value):
    ts = _as_timestamp(value)
    return "none" if ts is None else ts.strftime("%Y%m%d")


def _temporal_cache_key(config, stock_universe, seq_feature_cols=None):
    data_dir = str(getattr(config, "data_dir", "data/raw"))
    universe_tag = "all"
    if stock_universe:
        digest = hashlib.md5("|".join(sorted(stock_universe)).encode()).hexdigest()[:8]
        universe_tag = f"u{len(stock_universe)}_{digest}"
    elif getattr(config, "test_mode", False):
        universe_tag = f"test{getattr(config, 'test_stocks', 'n')}"
    elif getattr(config, "max_stocks", None):
        universe_tag = f"max{getattr(config, 'max_stocks')}"

    feature_flags = []
    for name in ("technical", "market", "macro", "fundamental", "shareholder", "restricted"):
        attr = f"use_{name}_features"
        if getattr(config, attr, False):
            if name == "fundamental" and getattr(config, "use_fundamental_quality_features", False):
                feature_flags.append("fundaq")
            else:
                feature_flags.append(name[:5])
    feat_tag = "_".join(feature_flags) if feature_flags else "basic"

    date_tag = (
        f"tr{_date_token(getattr(config, 'temporal_train_start', None))}"
        f"_va{_date_token(getattr(config, 'temporal_val_start', None))}"
        f"_te{_date_token(getattr(config, 'temporal_test_start', None))}"
        f"_end{_date_token(getattr(config, 'temporal_end_date', None))}"
    )
    seq_cols_tag = "seqauto"
    if seq_feature_cols is None:
        seq_feature_cols = getattr(config, "temporal_seq_feature_cols", None)
    if seq_feature_cols:
        seq_cols_tag = hashlib.md5("|".join(seq_feature_cols).encode()).hexdigest()[:8]

    seq_len = int(getattr(config, "seq_len", 40))
    lookback = int(getattr(config, "temporal_lookback", seq_len))
    default_buffer = max(365, int(max(seq_len, lookback, 80) * 2.2) + 90)
    hist_days = int(getattr(config, "temporal_history_calendar_days", default_buffer))
    hist_tag = f"_hist{hist_days}"
    raw = (
        f"{TEMPORAL_CACHE_VERSION}_{CACHE_VERSION}_{universe_tag}_{feat_tag}"
        f"_s{seq_len}_lb{lookback}"
        f"_h{getattr(config, 'max_horizon', 10)}_t{getattr(config, 'target_horizon', 5)}"
        f"_min{getattr(config, 'min_stocks_per_time', 30)}{hist_tag}_{date_tag}_{seq_cols_tag}"
        f"_{hashlib.md5(data_dir.encode()).hexdigest()[:6]}"
    )
    return "temporal_cross_section_" + raw


def _load_price_frames(config, stock_universe=None):
    data_dir = getattr(config, "data_dir", "data/raw")
    seq_len = getattr(config, "seq_len", 40)
    lookback = getattr(config, "temporal_lookback", seq_len)
    max_horizon = getattr(config, "max_horizon", 10)

    train_start = _as_timestamp(getattr(config, "temporal_train_start", None))
    explicit_raw_start = _as_timestamp(getattr(config, "temporal_raw_start_date", None))
    raw_start = explicit_raw_start
    if raw_start is None and train_start is not None:
        default_buffer = max(365, int(max(seq_len, lookback, 80) * 2.2) + 90)
        buffer_days = int(getattr(config, "temporal_history_calendar_days", default_buffer))
        raw_start = train_start - timedelta(days=buffer_days)

    end_date = assert_research_end_date(
        getattr(config, "temporal_end_date", None),
        context="temporal dataset",
    )

    excluded = {"all_data_jq.csv", "stable_stocks.csv", "stable_stocks_industry.csv"}
    csv_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") and f not in excluded and f[0].isdigit()
    )
    if getattr(config, "max_stocks", None):
        csv_files = csv_files[: int(config.max_stocks)]
    if getattr(config, "test_mode", False):
        csv_files = csv_files[: int(getattr(config, "test_stocks", 1000))]

    min_len = max(seq_len, lookback, 80) + max_horizon + 20

    def _load_one(fname):
        code = fname.replace(".csv", "")
        if stock_universe and code not in stock_universe:
            return None, None
        try:
            df = pd.read_csv(os.path.join(data_dir, fname))
            df.columns = df.columns.str.strip().str.lower()
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)
            if "code" in df.columns:
                df.drop(columns=["code"], inplace=True)
        except Exception:
            return None, None

        required = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            return None, None

        df = df.sort_index()
        if raw_start is not None:
            df = df[df.index >= raw_start]
        if end_date is not None:
            df = df[df.index <= end_date]
        if len(df) < min_len:
            return None, None
        if getattr(config, "use_technical_features", False):
            df = add_technical_features(df, config)
        return code, df

    df_dict = {}
    n_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_one, fname): fname for fname in csv_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="load temporal CSV", mininterval=10):
            code, df = fut.result()
            if code is not None:
                df_dict[code] = df

    if not df_dict:
        raise ValueError("No valid stock CSVs for temporal dataset")
    return df_dict


def _split_time_indices(all_dates, valid_times, config):
    train_start = _as_timestamp(getattr(config, "temporal_train_start", None))
    val_start = _as_timestamp(getattr(config, "temporal_val_start", None))
    test_start = _as_timestamp(getattr(config, "temporal_test_start", None))
    end_date = _as_timestamp(getattr(config, "temporal_end_date", None))

    filtered = []
    for t in valid_times:
        d = pd.Timestamp(all_dates[t])
        if train_start is not None and d < train_start:
            continue
        if end_date is not None and d > end_date:
            continue
        filtered.append(t)

    if val_start is not None:
        train = [t for t in filtered if pd.Timestamp(all_dates[t]) < val_start]
        if test_start is not None:
            val = [t for t in filtered if val_start <= pd.Timestamp(all_dates[t]) < test_start]
            test = [t for t in filtered if pd.Timestamp(all_dates[t]) >= test_start]
        else:
            val = [t for t in filtered if pd.Timestamp(all_dates[t]) >= val_start]
            test = []
        return train, val, test, filtered

    n = len(filtered)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    return filtered[:n_train], filtered[n_train:n_val], filtered[n_val:], filtered


def _normalize_daily_sequence(daily_raw, seq_norm_path, valid_times, min_stocks):
    num_stocks, num_dates, seq_dim = daily_raw.shape
    seq_norm = np.memmap(seq_norm_path, dtype=np.int16, mode="w+", shape=(num_stocks, num_dates, seq_dim))
    seq_norm[:] = SENTINEL

    def _norm_date(t):
        x = np.asarray(daily_raw[:, t, :], dtype=np.float32)
        valid = np.isfinite(x).all(axis=1)
        if valid.sum() < min_stocks:
            return None
        xv = x[valid]
        lo, hi = np.nanpercentile(xv, [1, 99], axis=0)
        xw = np.clip(xv, lo, hi)
        mean = np.nanmean(xw, axis=0, keepdims=True)
        std = np.nanstd(xw, axis=0, keepdims=True) + 1e-8
        zn = np.clip((xw - mean) / std, -4.0, 4.0)
        out = (zn * SCALE).clip(-32767, 32767).astype(np.int16)
        return t, np.where(valid)[0], out

    n_done = 0
    n_workers = min(12, (os.cpu_count() or 4) + 4)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_norm_date, t): t for t in valid_times}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                t, valid_idx, out = result
                seq_norm[valid_idx, t, :] = out
            n_done += 1
            if n_done % 500 == 0:
                print(f"  temporal sequence normalized {n_done}/{len(valid_times)}")

    seq_norm.flush()
    del seq_norm


def build_temporal_cross_section_dataset(config, stock_universe=None, use_cache=True):
    """Build temporal metadata and memmaps.

    Returns metadata only. Use core.temporal_train_utils.TemporalMemmapDataset
    to materialize train/val/test samples lazily.
    """

    end_date = assert_research_end_date(
        getattr(config, "temporal_end_date", None),
        context="temporal dataset",
    )
    cache_dir = getattr(config, "temporal_cache_dir", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = _temporal_cache_key(config, stock_universe)
    meta_path = os.path.join(cache_dir, cache_key + "_meta.pkl")
    legacy_meta_path = meta_path.replace(
        f"_end{_date_token(getattr(config, 'temporal_end_date', None))}_",
        "_endnone_",
    )
    cache_candidates = [meta_path]
    if legacy_meta_path != meta_path:
        cache_candidates.append(legacy_meta_path)

    if use_cache and not getattr(config, "force_rebuild", False):
        for candidate in cache_candidates:
            if not os.path.exists(candidate):
                continue
            with open(candidate, "rb") as f:
                cached = pickle.load(f)
            dat_keys = (
                "feat_path", "risk_path", "ret_path", "seq_raw_path",
                "x_norm_path", "risk_full_path", "y_norm_path",
                "y_seq_norm_path", "seq_norm_path",
            )
            missing = [cached[k] for k in dat_keys if k in cached and not os.path.exists(cached[k])]
            if missing:
                print(f"Temporal cache incomplete, rebuilding: {missing[0]}")
                continue
            if not cached_dates_within_research(cached, end_date):
                print(f"Temporal cache exceeds research cutoff, refusing: {candidate}")
                continue
            print(f"Loaded temporal metadata cache: {candidate}")
            return cached

    data_dir = getattr(config, "data_dir", "data/raw")
    seq_len = getattr(config, "seq_len", 40)
    lookback = int(getattr(config, "temporal_lookback", seq_len))
    max_horizon = int(getattr(config, "max_horizon", 10))
    min_stocks = int(getattr(config, "min_stocks_per_time", 30))

    print("Loading temporal stock data...")
    df_dict = _load_price_frames(config, stock_universe)
    print(f"Valid stocks: {len(df_dict)}")

    print("Computing base features...")
    base_features = _compute_base_features(df_dict)
    if getattr(config, "use_technical_features", False):
        feature_cols = list(dict.fromkeys(base_features + TECH_FEATURES))
    else:
        feature_cols = list(base_features)

    all_dates = sorted(set().union(*[df.index for df in df_dict.values()]))
    extra_feat_cols = _load_extra_features(config, df_dict, all_dates)
    feature_cols = list(dict.fromkeys(feature_cols + extra_feat_cols))

    high_freq_count = len(feature_cols) - len(extra_feat_cols)
    high_agg_dim = high_freq_count * N_AGGS
    low_agg_dim = len(extra_feat_cols) * 2
    agg_feat_dim = high_agg_dim + low_agg_dim

    seq_feature_cols = getattr(config, "temporal_seq_feature_cols", None)
    if seq_feature_cols is None:
        seq_feature_cols = feature_cols[:high_freq_count]
    else:
        seq_feature_cols = [c for c in seq_feature_cols if c in feature_cols]
    if not seq_feature_cols:
        raise ValueError("temporal_seq_feature_cols resolved to an empty list")

    num_dates = len(all_dates)
    all_codes = list(df_dict.keys())
    num_stocks = len(all_codes)
    code_to_idx = {code: i for i, code in enumerate(all_codes)}
    date_to_idx = {date: i for i, date in enumerate(all_dates)}

    industry_dict, all_industries, industry_to_idx, n_industries = _load_industry_map(data_dir)
    print(
        f"Dates={num_dates}, agg_dim={agg_feat_dim}, seq_dim={len(seq_feature_cols)}, "
        f"industries={n_industries}"
    )

    feat_path = os.path.join(cache_dir, cache_key + "_feat.dat")
    risk_path = os.path.join(cache_dir, cache_key + "_risk.dat")
    ret_path = os.path.join(cache_dir, cache_key + "_ret.dat")
    seq_raw_path = os.path.join(cache_dir, cache_key + "_seq_raw.dat")

    feat_array = np.memmap(feat_path, dtype=np.float32, mode="w+", shape=(num_stocks, num_dates, agg_feat_dim))
    risk_dim = N_STOCK_RISK + N_MARKET + (len(MACRO_COLS) if getattr(config, "use_macro_features", False) else 0)
    risk_raw_array = np.memmap(risk_path, dtype=np.float32, mode="w+", shape=(num_stocks, num_dates, risk_dim))
    ret_seq_array = np.memmap(ret_path, dtype=np.float32, mode="w+", shape=(num_stocks, num_dates, max_horizon))
    daily_seq_raw = np.memmap(seq_raw_path, dtype=np.float32, mode="w+", shape=(num_stocks, num_dates, len(seq_feature_cols)))
    industry_array = np.full((num_stocks, num_dates), -1, dtype=np.int16)

    feat_array[:] = np.nan
    risk_raw_array[:] = 0.0
    ret_seq_array[:] = np.nan
    daily_seq_raw[:] = np.nan

    high_freq_slice = slice(0, high_freq_count)
    low_freq_slice = slice(high_freq_count, len(feature_cols))

    print("Filling temporal arrays...")

    def _fill_one_stock(args):
        code, df, sidx = args
        stock_dates = df.index
        stock_idx = np.array([date_to_idx[d] for d in stock_dates], dtype=np.int32)
        raw_feat = df.reindex(columns=feature_cols).values.astype(np.float32)
        seq_feat = df.reindex(columns=seq_feature_cols).values.astype(np.float32)
        T = len(stock_dates)

        daily_seq_raw[sidx, stock_idx, :] = seq_feat

        if T >= seq_len:
            windows = sliding_window_view(raw_feat, seq_len, axis=0)
            if windows.shape[1] != seq_len:
                windows = windows.transpose(0, 2, 1)
            n_win = windows.shape[0]

            last_high = windows[:, -1, high_freq_slice]
            sma5 = windows[:, -5:, high_freq_slice].mean(axis=1) if seq_len >= 5 else last_high
            sma20 = windows[:, -20:, high_freq_slice].mean(axis=1) if seq_len >= 20 else sma5
            vol5 = windows[:, -5:, high_freq_slice].std(axis=1) if seq_len >= 5 else np.zeros_like(last_high)
            vol20 = windows[:, -20:, high_freq_slice].std(axis=1) if seq_len >= 20 else vol5
            high_agg = np.concatenate([last_high, sma5, sma20, vol5, vol20], axis=1)

            last_low = windows[:, -1, low_freq_slice]
            qoq_lb = min(seq_len // 4, 10)
            qoq_low = last_low - windows[:, -qoq_lb, low_freq_slice]
            agg_feat = np.concatenate([high_agg, last_low, qoq_low], axis=1)
            agg_idx = stock_idx[seq_len - 1: seq_len - 1 + n_win]
            feat_array[sidx, agg_idx, :] = agg_feat

        risk_raw_array[sidx, stock_idx, 0] = df["log_volume"].values.astype(np.float32)
        risk_raw_array[sidx, stock_idx, 1] = df["vol_60d"].fillna(0).values.astype(np.float32)
        risk_raw_array[sidx, stock_idx, 2] = df["ret_20d"].fillna(0).values.astype(np.float32)
        risk_raw_array[sidx, stock_idx, 3] = df["ret_5d"].fillna(0).values.astype(np.float32)
        turnover_proxy = df["volume_ratio"] if "volume_ratio" in df.columns else df["volume_spike"]
        risk_raw_array[sidx, stock_idx, 4] = turnover_proxy.fillna(0).values.astype(np.float32)
        risk_raw_array[sidx, stock_idx, 5] = df["amplitude"].fillna(0).values.astype(np.float32)

        raw_ind = industry_dict.get(code)
        ind_id = industry_to_idx.get(raw_ind, -1) if raw_ind else -1
        industry_array[sidx, stock_idx] = ind_id

        close_vals = df["close"].values.astype(np.float32)
        if T >= max_horizon + 1:
            windows_close = sliding_window_view(close_vals, max_horizon + 1, axis=0)
            n_ret = windows_close.shape[0]
            ret_idx = stock_idx[:n_ret]
            base = np.where(windows_close[:, 0] > 0, windows_close[:, 0], np.nan)
            for h in range(1, max_horizon + 1):
                ret_seq_array[sidx, ret_idx, h - 1] = (windows_close[:, h] - base) / base

    tasks = [(code, df, code_to_idx[code]) for code, df in df_dict.items()]
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        futures = [pool.submit(_fill_one_stock, task) for task in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="fill temporal arrays", mininterval=10):
            pass

    if getattr(config, "use_market_features", True):
        print("Computing market regime features...")
        close_matrix = np.full((num_stocks, num_dates), np.nan, dtype=np.float32)
        for code, df in df_dict.items():
            sidx = code_to_idx[code]
            idx = np.array([date_to_idx[d] for d in df.index], dtype=np.int32)
            close_matrix[sidx, idx] = df["close"].values.astype(np.float32)
        breadth = compute_breadth_from_close_matrix(close_matrix)
        idx_feat = build_market_features_index_only(data_dir, all_dates)
        for t_idx, date in enumerate(all_dates):
            if date in idx_feat.index:
                row = idx_feat.loc[date].values.astype(np.float32)
                risk_raw_array[:, t_idx, N_STOCK_RISK:N_STOCK_RISK + 16] = row[:16]
                risk_raw_array[:, t_idx, N_STOCK_RISK + 16:N_STOCK_RISK + 19] = breadth[t_idx]
                risk_raw_array[:, t_idx, N_STOCK_RISK + 19:N_STOCK_RISK + N_MARKET] = row[16:]
        del close_matrix, breadth

    if getattr(config, "use_macro_features", False):
        print("Adding macro regime features...")
        try:
            from data.macro_factors import build_macro_features

            macro_df = build_macro_features(all_dates)
            macro_start = N_STOCK_RISK + N_MARKET
            for j, col in enumerate(MACRO_COLS):
                if col in macro_df.columns:
                    vals = macro_df[col].reindex(all_dates).fillna(0).values.astype(np.float32)
                    risk_raw_array[:, :, macro_start + j] = vals[None, :]
        except Exception as exc:
            print(f"Macro features failed, using zeros: {exc}")

    feat_array.flush()
    risk_raw_array.flush()
    ret_seq_array.flush()
    daily_seq_raw.flush()

    min_history = max(seq_len, lookback, 80)
    valid_times_all = list(range(min_history, num_dates - max_horizon))
    train_indices, val_indices, test_indices, valid_times = _split_time_indices(all_dates, valid_times_all, config)
    if not train_indices or not val_indices:
        raise ValueError(
            f"Invalid temporal split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
        )

    print(
        f"Split dates: train={len(train_indices)}, val={len(val_indices)}, "
        f"test={len(test_indices)}, total={len(valid_times)}"
    )

    print("Precomputing cross-section X/risk/labels...")
    # Temporal sidecar currently stores one close-to-close return sequence.
    # Wrap it for the v14 multi-label precompute API; dedicated open labels
    # should be generated through data.pipeline for production M0 training.
    label_arrays = {family: ret_seq_array for family in PHYSICAL_LABEL_FAMILIES}
    x_norm_path, risk_full_path, y_norm_path, y_seq_norm_path, x_dim, risk_full_dim = _precompute_all(
        feat_array,
        risk_raw_array,
        industry_array,
        label_arrays,
        valid_times,
        high_agg_dim,
        n_industries,
        feature_cols,
        max_horizon,
        getattr(config, "target_horizon", 5),
        getattr(config, "residualize_labels", False),
        cache_key,
        cache_dir,
        min_stocks=min_stocks,
    )

    print("Precomputing temporal sequence normalization...")
    seq_norm_path = os.path.join(cache_dir, cache_key + "_seq_norm.dat")
    _normalize_daily_sequence(daily_seq_raw, seq_norm_path, range(num_dates), min_stocks)

    metadata = {
        "builder": "build_temporal_cross_section_dataset",
        "cache_key": cache_key,
        "meta_path": meta_path,
        "feat_path": feat_path,
        "risk_path": risk_path,
        "ret_path": ret_path,
        "seq_raw_path": seq_raw_path,
        "x_norm_path": x_norm_path,
        "risk_full_path": risk_full_path,
        "y_norm_path": y_norm_path,
        "y_seq_norm_path": y_seq_norm_path,
        "seq_norm_path": seq_norm_path,
        "x_dim": x_dim,
        "risk_full_dim": risk_full_dim,
        "seq_dim": len(seq_feature_cols),
        "seq_lookback": lookback,
        "risk_cont_dim": risk_dim,
        "industry_array": industry_array,
        "all_codes": all_codes,
        "all_dates": all_dates,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "valid_indices": valid_times,
        "n_industries": n_industries,
        "feature_cols": feature_cols,
        "seq_feature_cols": seq_feature_cols,
        "high_agg_dim": high_agg_dim,
        "low_agg_dim": low_agg_dim,
        "high_feat_dim": high_freq_count,
        "target_horizon": getattr(config, "target_horizon", 5),
        "max_horizon": max_horizon,
        "min_stocks": min_stocks,
        "date_split": {
            "train_start": str(_as_timestamp(getattr(config, "temporal_train_start", None))),
            "val_start": str(_as_timestamp(getattr(config, "temporal_val_start", None))),
            "test_start": str(_as_timestamp(getattr(config, "temporal_test_start", None))),
            "end_date": str(_as_timestamp(getattr(config, "temporal_end_date", None))),
        },
        "scale": SCALE,
        "sentinel": int(SENTINEL),
    }

    if use_cache:
        print(f"Saving temporal metadata: {meta_path}")
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    return metadata
