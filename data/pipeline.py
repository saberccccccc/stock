# data_pipeline.py - 截面多因子数据流水线（内存优化版）
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from collections import defaultdict
import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core.research_protocol import assert_research_end_date, cached_dates_within_research
from data.labels import PHYSICAL_LABEL_FAMILIES, build_forward_return_labels

warnings.filterwarnings('ignore')

# 宏观特征列名
MACRO_COLS = ['north_net_zscore', 'margin_balance_change', 'pmi_zscore']
# 基本面特征列名（akshare下载，不含pe_percentile）
FUNDAMENTAL_COLS = ['roe', 'revenue_yoy']
FUNDAMENTAL_QUALITY_COLS = [
    'has_value',
    'days_since_effective',
    'is_fresh_quarter',
    'notice_is_estimated',
]
# 股东户数特征列名
SHAREHOLDER_COLS = ['sh_conc_ratio', 'sh_per_capita_ratio']
# 限售解禁特征列名
RESTRICTED_COLS = ['restricted_next_inv_days', 'restricted_next_ratio', 'restricted_mv_ratio_90d']
# 市场整体属性
from data.market_features import (
    build_market_features_index_only,
    compute_breadth_from_close_matrix,
    MARKET_COLS, N_MARKET,
)

# V7 聚合方式（模块级常量，供 train.py 引用）
AGG_NAMES = ['last', 'sma5', 'sma20', 'vol5', 'vol20']
N_AGGS = len(AGG_NAMES)
INDUSTRY_REL_FEATURES = ['ret_5d', 'ret_20d', 'vol_10d', 'vol_60d', 'price_momentum', 'log_volume']
TECH_FEATURES = [
    'sma5_gap', 'sma10_gap', 'sma20_gap', 'ema12_gap', 'ema26_gap',
    'rsi_norm', 'macd_pct', 'macd_signal_pct', 'macd_diff_pct',
    'atr_pct', 'volume_ratio'
]
CACHE_VERSION = "v14_multilabel_open"
N_STOCK_RISK = 6  # 个股风险因子数: size, vol, momentum, reversal, turnover, amplitude


def _bool_tag(name, enabled):
    return name if enabled else f"no{name}"


def _cache_config_tag(config, data_dir, stock_universe):
    norm_tag = "mad" if getattr(config, 'normalize_features', True) else "raw"
    min_stocks = getattr(config, 'min_stocks_per_time', 30)
    res_tag = "res" if getattr(config, 'residualize_labels', False) else "rawlab"
    universe_tag = "all"
    if stock_universe:
        universe_digest = hashlib.md5("|".join(sorted(stock_universe)).encode()).hexdigest()[:6]
        universe_tag = f"u{len(stock_universe)}_{universe_digest}"
    elif getattr(config, 'test_mode', False):
        universe_tag = f"test{getattr(config, 'test_stocks', 'n')}"
    elif getattr(config, 'max_stocks', None):
        universe_tag = f"max{getattr(config, 'max_stocks')}"
    lb_tag = f"_lb{config.max_lookback}" if getattr(config, 'max_lookback', None) else ""
    research_end = getattr(config, 'research_end_date', None)
    end_tag = f"_end{pd.Timestamp(research_end).strftime('%Y%m%d')}" if research_end else ""
    return f"{universe_tag}_s{config.seq_len}_t{config.target_horizon}_h{getattr(config, 'max_horizon', 10)}_min{min_stocks}_{norm_tag}_{res_tag}{lb_tag}{end_tag}"


def _compute_base_features(df_dict):
    """Compute 12 base features in-place for every stock DataFrame."""
    base_features = [
        'ret_5d', 'ret_20d', 'vol_10d', 'vol_60d',
        'price_momentum', 'log_volume', 'volume_spike',
        'upper_shadow', 'lower_shadow', 'body_size', 'gap', 'amplitude'
    ]
    for code, df in df_dict.items():
        close_safe = df['close'].where(df['close'] > 0)
        prev_close_safe = df['close'].shift(1).where(df['close'].shift(1) > 0)
        df['log_close'] = np.log(close_safe).replace([np.inf, -np.inf], np.nan)
        df['log_volume'] = np.log(df['volume'].clip(lower=0) + 1)

        daily_ret = close_safe.pct_change().replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
        df['ret_5d'] = close_safe.pct_change(5).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        df['ret_20d'] = close_safe.pct_change(20).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        df['vol_10d'] = daily_ret.rolling(10).std().clip(0, 1.0)
        df['vol_60d'] = daily_ret.rolling(60).std().clip(0, 1.0)
        df['price_momentum'] = (close_safe / close_safe.rolling(20).mean() - 1).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        df['volume_spike'] = df['log_volume'].pct_change(1).abs().replace([np.inf, -np.inf], np.nan).clip(0, 10)

        hl_range_raw = df['high'] - df['low']
        valid_range = hl_range_raw > (close_safe * 1e-4)
        hl_range = hl_range_raw.where(valid_range)
        df['upper_shadow'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / hl_range).clip(0, 1)
        df['lower_shadow'] = ((df[['open', 'close']].min(axis=1) - df['low']) / hl_range).clip(0, 1)
        df['body_size'] = (abs(df['close'] - df['open']) / hl_range).clip(0, 1)
        df['gap'] = ((df['open'] - df['close'].shift(1)) / prev_close_safe).replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
        df['amplitude'] = (hl_range_raw / close_safe).replace([np.inf, -np.inf], np.nan).clip(0, 1)
    return base_features


def _load_industry_map(data_dir):
    """Load stock industry CSV, return (industry_dict, all_industries, industry_to_idx, n_industries)."""
    industry_file = os.path.join(os.path.dirname(str(data_dir)), "stock_industry.csv")
    if not os.path.exists(industry_file):
        industry_file = "stock_industry.csv"

    industry_dict = {}
    all_industries = []
    if os.path.exists(industry_file):
        industry_df = pd.read_csv(industry_file)
        if 'code' in industry_df.columns:
            industry_df['code_norm'] = industry_df['code'].apply(_normalize_ts_code)
            industry_df = industry_df.dropna(subset=['industry', 'code_norm'])
            industry_dict = dict(zip(industry_df['code_norm'], industry_df['industry']))
            all_industries = sorted(set(industry_dict.values()))
    n_industries = len(all_industries)
    industry_to_idx = {ind: i for i, ind in enumerate(all_industries)}
    return industry_dict, all_industries, industry_to_idx, n_industries


def _residualize_labels(y_seq_t, ind_ids, size_proxy):
    """
    标签残差化：剥离行业均值 + 规模效应，保留纯个股 alpha 信号。
    y_seq_t: (N, H) 原始未来收益率
    ind_ids: (N,) 行业 ID，-1 表示未知
    size_proxy: (N,) 规模代理变量（log_volume）
    Returns: (N, H) 残差化后的收益率
    """
    residual = y_seq_t.copy()
    n_stocks, n_horizons = residual.shape

    for h in range(n_horizons):
        y_h = residual[:, h]

        # 1. 行业内去均值
        valid = ind_ids >= 0
        unique_inds = np.unique(ind_ids[valid])
        for ind in unique_inds:
            mask = ind_ids == ind
            if mask.sum() >= 3:
                y_h[mask] -= np.mean(y_h[mask])
        if (~valid).any():
            y_h[~valid] -= np.mean(y_h[valid]) if valid.any() else 0.0

        # 2. 市值回归去趋势：残差 = y - (alpha + beta * size)
        size_valid = np.isfinite(size_proxy) & (np.abs(size_proxy) < 50)
        if size_valid.sum() >= 20:
            A = np.column_stack([np.ones(size_valid.sum()), size_proxy[size_valid]])
            beta = np.linalg.lstsq(A, y_h[size_valid], rcond=None)[0]
            y_h[size_valid] -= A @ beta
            y_h[~size_valid] -= beta[0]  # 无 size 数据时至少去掉截距

        residual[:, h] = y_h
    return residual


def _normalize_and_assemble(X_t, X_rank, industry_relative, risk_vals, ind_ids, n_industries):
    """分特征组归一化：
    - agg 特征(129): winsorize(1%/99%) + z-score + clip(±4) —— 解决长尾压缩问题
    - rank 特征(115): z-score + clip(±4) —— 已均匀[0,1]，不 winsorize（会抹掉极端排序）
    - ind_rel 特征(6): z-score + clip(±4) —— 已均值中心化，分布良好
    """
    def _winz(x):
        """winsorize + z-score + clip"""
        pl, ph = np.percentile(x, [1, 99], axis=0)
        xw = np.clip(x, pl, ph)
        m, s = xw.mean(axis=0, keepdims=True), xw.std(axis=0, keepdims=True) + 1e-8
        return np.clip((xw - m) / s, -4.0, 4.0)

    def _zscore(x):
        """z-score + clip (no winsorize)"""
        m, s = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + 1e-8
        return np.clip((x - m) / s, -4.0, 4.0)

    X_agg_norm = _winz(X_t)                           # agg(129): winsorize+z-score
    X_rank_norm = _zscore(X_rank)                     # rank(115): z-score only
    X_indrel_norm = _zscore(industry_relative)        # ind_rel(6): z-score only
    X_norm = np.nan_to_num(
        np.concatenate([X_agg_norm, X_rank_norm, X_indrel_norm], axis=1),
        nan=0.0, posinf=0.0, neginf=0.0)

    # Risk: 前3列 z-score + clip, 其余不变（已有归一化或 one-hot）
    risk_cont_norm = risk_vals.copy()
    risk_mean = risk_vals[:, :6].mean(axis=0, keepdims=True)
    risk_std = risk_vals[:, :6].std(axis=0, keepdims=True) + 1e-8
    risk_cont_norm[:, :6] = (risk_vals[:, :6] - risk_mean) / risk_std
    risk_cont_norm = np.nan_to_num(risk_cont_norm, nan=0.0, posinf=0.0, neginf=0.0)
    risk_cont_norm[:, :6] = np.clip(risk_cont_norm[:, :6], -4.0, 4.0)

    # 行业信息已由 industry_ids 单独传递（embedding + GAT 边），不放 risk 里
    return X_norm, risk_cont_norm


def _normalize_ts_code(code):
    code = str(code).strip()
    if not code or code.lower() == 'nan':
        return None
    lower = code.lower()
    if lower.startswith('sh.') or lower.startswith('sz.'):
        return f"{code[3:9]}.{lower[:2].upper()}"
    if '.' in code:
        left, right = code.split('.', 1)
        if left.isdigit():
            return f"{left.zfill(6)}.{right.upper()}"
    digits = ''.join(ch for ch in code if ch.isdigit())
    if len(digits) >= 6:
        suffix = 'SH' if digits[:1] in {'5', '6', '9'} else 'SZ'
        return f"{digits[-6:]}.{suffix}"
    return code


def _load_extra_features(config, df_dict, all_dates):
    """加载并合并个股扩展因子（基本面等）。"""
    extra_feat_cols = []

    # ----- 基本面因子（PIT对齐，从akshare parquet读取）-----
    if getattr(config, 'use_fundamental_features', False):
        try:
            from data.fundamental_factors import merge_to_daily_akshare
            codes = list(df_dict.keys())
            funda_path = os.path.join("cache", "fundamental_features_akshare.parquet")
            if os.path.exists(funda_path):
                funda_df = pd.read_parquet(funda_path)
                if not funda_df.empty:
                    include_quality = getattr(config, 'use_fundamental_quality_features', False)
                    funda_daily = merge_to_daily_akshare(
                        funda_df, codes, all_dates, include_quality=include_quality
                    )
                    fundamental_cols = list(FUNDAMENTAL_COLS)
                    if include_quality:
                        fundamental_cols += FUNDAMENTAL_QUALITY_COLS
                    for code in codes:
                        for col in fundamental_cols:
                            citem = f'{code}_{col}'
                            if citem in funda_daily.columns:
                                df_dict[code][f'fund_{col}'] = funda_daily[citem].reindex(
                                    df_dict[code].index, method='ffill').fillna(0).values
                            else:
                                df_dict[code][f'fund_{col}'] = 0.0
                    extra_feat_cols += [f'fund_{c}' for c in fundamental_cols]
                    print(f"已加载基本面特征: {[f'fund_{c}' for c in fundamental_cols]}")
            else:
                print("提示: fundamental_features_akshare.parquet 不存在，跳过基本面因子")
        except Exception as e:
            print(f"基本面特征加载失败 {e}")

    # ----- 股东户数（PIT对齐的筹码集中度）-----
    if getattr(config, 'use_shareholder_features', False):
        try:
            from data.shareholder_features import download_all, merge_to_daily
            codes = list(df_dict.keys())
            sh_df = download_all()
            if not sh_df.empty:
                sh_daily = merge_to_daily(sh_df, codes, all_dates)
                for code in codes:
                    for col in SHAREHOLDER_COLS:
                        citem = f'{code}_{col}'
                        if citem in sh_daily.columns:
                            df_dict[code][f'sh_{col}'] = sh_daily[citem].reindex(
                                df_dict[code].index, method='ffill').fillna(0).values
                        else:
                            df_dict[code][f'sh_{col}'] = 0.0
                extra_feat_cols += [f'sh_{c}' for c in SHAREHOLDER_COLS]
                print(f"已加载股东户数特征: {[f'sh_{c}' for c in SHAREHOLDER_COLS]}")
        except Exception as e:
            print(f"股东户数特征加载失败 {e}")

    # ----- 限售解禁（未来解禁压力，天然无前视偏差）-----
    if getattr(config, 'use_restricted_features', False):
        try:
            from data.restricted_features import download_all, merge_to_daily
            codes = list(df_dict.keys())
            restr_df = download_all()
            if not restr_df.empty:
                restr_daily = merge_to_daily(restr_df, codes, all_dates)
                for code in codes:
                    for col in RESTRICTED_COLS:
                        citem = f'{code}_{col}'
                        if citem in restr_daily.columns:
                            df_dict[code][f'restr_{col}'] = restr_daily[citem].reindex(
                                df_dict[code].index, method='ffill').fillna(0).values
                        else:
                            df_dict[code][f'restr_{col}'] = 0.0
                extra_feat_cols += [f'restr_{c}' for c in RESTRICTED_COLS]
                print(f"已加载限售解禁特征: {[f'restr_{c}' for c in RESTRICTED_COLS]}")
        except Exception as e:
            print(f"限售解禁特征加载失败 {e}")

    return extra_feat_cols


def build_cross_section_dataset(config, stock_universe=None, use_cache=True):
    """Build cross-section dataset, return (train_samples, val_samples).

    相较v6版本的内存优化：
    - 不缓存全局特征矩阵 (num_stocks × num_dates × feat_dim)
    - 改用窗口索引预计算 + 按需切片
    - 保留截面rank特征和行业相对特征
    """
    data_dir = config.data_dir
    research_end = assert_research_end_date(
        getattr(config, 'research_end_date', None),
        context="cross-section dataset",
    )
    seq_len = config.seq_len
    future_len = getattr(config, 'future_len', 5)
    max_horizon = getattr(config, 'max_horizon', 10)

    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    test_n = getattr(config, 'test_stocks', None) if getattr(config, 'test_mode', False) else None

    # 可读的缓存文件名
    n_stocks = test_n if test_n else config.max_stocks if config.max_stocks else "all"
    features = []
    if config.use_technical_features:
        features.append("tech")
    if config.use_market_features:
        features.append("market")
    if config.use_fundamental_features:
        features.append("fundaq" if getattr(config, 'use_fundamental_quality_features', False) else "funda")
    if config.use_shareholder_features:
        features.append("shareh")
    if config.use_restricted_features:
        features.append("restr")
    if config.use_macro_features:
        features.append("macro")
    feat_str = "_".join(features) if features else "basic"

    universe_tag = ""
    if stock_universe:
        universe_digest = hashlib.md5("|".join(sorted(stock_universe)).encode()).hexdigest()[:8]
        universe_tag = f"_universe{len(stock_universe)}_{universe_digest}"
    config_tag = _cache_config_tag(config, data_dir, stock_universe)
    cache_key = f"cross_section_{CACHE_VERSION}_{feat_str}_{config_tag}"
    meta_path = os.path.join(cache_dir, cache_key + "_meta.pkl")
    legacy_meta_path = meta_path.replace(
        f"_end{pd.Timestamp(config.research_end_date).strftime('%Y%m%d')}_meta.pkl",
        "_meta.pkl",
    )
    cache_candidates = [meta_path]
    if legacy_meta_path != meta_path:
        cache_candidates.append(legacy_meta_path)
    if getattr(config, 'allow_cache_superset', False):
        target_token = f"_end{pd.Timestamp(config.research_end_date).strftime('%Y%m%d')}_meta.pkl"
        superset_pattern = os.path.basename(meta_path).replace(target_token, "_end*_meta.pkl")
        cache_candidates.extend(
            str(path)
            for path in sorted(Path(cache_dir).glob(superset_pattern), key=lambda item: item.name)
            if str(path) not in cache_candidates
        )
    if use_cache and not config.force_rebuild:
        # 验证所有 .dat 文件存在，防止手动删文件后缓存静默失败
        for candidate in cache_candidates:
            if not os.path.exists(candidate):
                continue
            with open(candidate, 'rb') as f:
                cached = pickle.load(f)
            dat_keys = ('feat_path', 'risk_path', 'ret_path',
                        'x_norm_path', 'risk_full_path', 'y_norm_path', 'y_seq_norm_path')
            dat_files = [cached[k] for k in dat_keys if k in cached]
            for family_meta in cached.get('label_families', {}).values():
                if family_meta.get('alias_of'):
                    continue
                dat_files.extend(
                    family_meta[k] for k in ('raw_path', 'norm_path') if k in family_meta
                )
            missing = [p for p in dat_files if not os.path.exists(p)]
            if missing:
                print(f"缓存不完整（{len(missing)} 个 .dat 文件缺失），重建: {missing[0]}")
                continue
            if (
                not cached_dates_within_research(cached, research_end)
                and getattr(config, 'allow_cache_superset', False)
            ):
                dates = pd.DatetimeIndex(pd.to_datetime(cached.get('all_dates', []))).normalize()
                if not dates.empty and dates.min() <= research_end <= dates.max():
                    cached = dict(cached)
                    cached['physical_data_start'] = str(dates.min().date())
                    cached['physical_data_end'] = str(dates.max().date())
                    cached['effective_data_end'] = str(research_end.date())
                    cached['cache_view_kind'] = 'physical_superset_logical_cutoff'
                    cached['meta_path'] = str(Path(candidate).resolve())
                    print(
                        f"loaded physical superset cache with logical cutoff "
                        f"{research_end.date()}: {candidate}"
                    )
                    return cached
            if not cached_dates_within_research(cached, research_end):
                print(f"缓存超过研究截止日，拒绝使用: {candidate}")
                continue
            print(f"加载缓存元数据: {candidate}")
            return cached

    # ========== 1. 读取股票数据 ==========
    print("读取股票数据...")
    excluded = {'all_data_jq.csv', 'stable_stocks.csv', 'stable_stocks_industry.csv'}
    csv_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and f not in excluded and f[0].isdigit()
    )
    if config.max_stocks:
        csv_files = csv_files[:config.max_stocks]
    if getattr(config, 'test_mode', False):
        test_n = getattr(config, 'test_stocks', 1000)
        csv_files = csv_files[:test_n]
        print(f"test mode: loading only {len(csv_files)} stocks")

    def _load_one(fname):
        code = fname.replace('.csv', '')
        if stock_universe and code not in stock_universe:
            return None, None
        file_path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
            if 'code' in df.columns:
                df.drop(columns=['code'], inplace=True)
        except Exception:
            return None, None
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            return None, None
        df = df.sort_index()
        df = df[df.index <= research_end]
        if config.use_technical_features:
            df = add_technical_features(df, config)
        if len(df) >= seq_len + max_horizon + 50:
            return code, df
        return None, None

    df_dict = {}
    n_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_one, fname): fname for fname in csv_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="加载CSV", mininterval=10):
            code, df = fut.result()
            if code is not None:
                df_dict[code] = df

    if not df_dict:
        raise ValueError("没有有效股票数据")
    print(f"有效股票数 {len(df_dict)}")

    # ========== 2. 构造多尺度特征 ==========
    print("构造多尺度特征...")
    BASE_FEATURES = _compute_base_features(df_dict)

    if config.use_technical_features:
        FEATURE_COLS = BASE_FEATURES + TECH_FEATURES
    else:
        FEATURE_COLS = BASE_FEATURES

    FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))  # 去重保序
    base_feat_dim = len(FEATURE_COLS)
    print(f"基础特征数 {base_feat_dim}, 聚合方式: {N_AGGS}种")

    # ========== 3. 加载扩展因子 ==========
    all_dates = sorted(set().union(*[df.index for df in df_dict.values()]))
    extra_feat_cols = _load_extra_features(config, df_dict, all_dates)
    FEATURE_COLS = FEATURE_COLS + extra_feat_cols
    base_feat_dim = len(FEATURE_COLS)
    high_agg_dim = (base_feat_dim - len(extra_feat_cols)) * N_AGGS  # 高频 23×5=115
    low_agg_dim = len(extra_feat_cols) * 2  # 低频 last+qoq
    agg_feat_dim = high_agg_dim + low_agg_dim  # 129
    print(f"最终特征列数 {base_feat_dim}, 聚合特征数 {agg_feat_dim} (高频{high_agg_dim}+低频{low_agg_dim})")

    # ========== 4. 日期截断（控制内存）==========
    max_lookback = getattr(config, 'max_lookback', None)
    num_dates = len(all_dates)
    if max_lookback is not None and num_dates > max_lookback:
        cutoff_date = all_dates[-max_lookback]
        for code in list(df_dict.keys()):
            df = df_dict[code]
            df = df[df.index >= cutoff_date]
            if len(df) < seq_len:
                del df_dict[code]
            else:
                df_dict[code] = df
        all_dates = sorted(set().union(*[df.index for df in df_dict.values()]))
        num_dates = len(all_dates)
        print(f"全局日期数 {num_dates} (截断最近 {max_lookback} 天)")
    else:
        print(f"全局日期数 {num_dates}")

    # ========== 5. 行业数据 ==========
    industry_dict, all_industries, industry_to_idx, n_industries = _load_industry_map(data_dir)
    print(f"行业数 {n_industries}")

    # ========== 6. 填充特征矩阵 ==========
    # 构建 (num_stocks, num_dates, agg_feat_dim) 矩阵
    # 这是唯一的大矩阵，但必须存在以供后续截面rank等操作
    all_codes = list(df_dict.keys())
    num_stocks = len(all_codes)
    code_to_idx = {code: i for i, code in enumerate(all_codes)}
    date_to_idx = {date: i for i, date in enumerate(all_dates)}

    feat_array = np.full((num_stocks, num_dates, agg_feat_dim), np.nan, dtype=np.float32)
    macro_dim = len(MACRO_COLS) if getattr(config, 'use_macro_features', False) else 0
    risk_cont_dim = 6 + N_MARKET + macro_dim  # 6 stock risk factors
    risk_raw_array = np.zeros((num_stocks, num_dates, risk_cont_dim), dtype=np.float32)
    industry_array = np.full((num_stocks, num_dates), -1, dtype=np.int16)
    raw_label_paths = {
        family: os.path.join(cache_dir, cache_key + f"_label_{family}_raw.dat")
        for family in PHYSICAL_LABEL_FAMILIES
    }
    label_arrays = {
        family: np.memmap(
            raw_label_paths[family], dtype=np.float32, mode='w+',
            shape=(num_stocks, num_dates, max_horizon),
        )
        for family in PHYSICAL_LABEL_FAMILIES
    }

    # 高频特征索引（前23个：base12 + tech11），低频特征索引（后7个：extra）
    HIGH_FREQ_COUNT = len(FEATURE_COLS) - len(extra_feat_cols)  # 23
    high_freq_slice = slice(0, HIGH_FREQ_COUNT)
    low_freq_slice = slice(HIGH_FREQ_COUNT, len(FEATURE_COLS))

    print("填充特征矩阵...")

    def _fill_one_stock(args):
        code, df, sidx, feat_cols, high_slice, low_slice = args
        stock_dates = df.index
        stock_idx_inner = np.array([date_to_idx[d] for d in stock_dates], dtype=np.int32)
        T = len(stock_dates)

        raw_feat = df.reindex(columns=feat_cols).values
        if T >= seq_len:
            windows = sliding_window_view(raw_feat, seq_len, axis=0)
            if windows.shape[1] != seq_len:
                windows = windows.transpose(0, 2, 1)
            n_win = windows.shape[0]

            last_high = windows[:, -1, high_slice]
            sma5 = windows[:, -5:, high_slice].mean(axis=1) if seq_len >= 5 else last_high
            sma20 = windows[:, -20:, high_slice].mean(axis=1) if seq_len >= 20 else sma5
            vol5 = windows[:, -5:, high_slice].std(axis=1) if seq_len >= 5 else np.zeros_like(last_high)
            vol20 = windows[:, -20:, high_slice].std(axis=1) if seq_len >= 20 else vol5
            high_agg = np.concatenate([last_high, sma5, sma20, vol5, vol20], axis=1)

            last_low = windows[:, -1, low_slice]
            qoq_lb = min(seq_len // 4, 10)
            qoq_low = last_low - windows[:, -qoq_lb, low_slice]
            agg_feat = np.concatenate([high_agg, last_low, qoq_low], axis=1)

            agg_idx = stock_idx_inner[seq_len - 1: seq_len - 1 + n_win]
            feat_array[sidx, agg_idx, :] = agg_feat

        # 连续风险因子 (6维): 规模/波动/动量/反转/换手率/振幅
        risk_raw_array[sidx, stock_idx_inner, 0] = df['log_volume'].values.astype(np.float32)
        risk_raw_array[sidx, stock_idx_inner, 1] = df['vol_60d'].fillna(0).values.astype(np.float32)
        risk_raw_array[sidx, stock_idx_inner, 2] = df['ret_20d'].fillna(0).values.astype(np.float32)
        risk_raw_array[sidx, stock_idx_inner, 3] = df['ret_5d'].fillna(0).values.astype(np.float32)
        risk_raw_array[sidx, stock_idx_inner, 4] = df['volume_ratio'].fillna(0).values.astype(np.float32)
        risk_raw_array[sidx, stock_idx_inner, 5] = df['amplitude'].fillna(0).values.astype(np.float32)

        # 行业ID
        raw_ind = industry_dict.get(code)
        ind_id = industry_to_idx.get(raw_ind, -1) if raw_ind else -1
        industry_array[sidx, stock_idx_inner] = ind_id

        # 多期收益率
        aligned = df.reindex(all_dates)
        stock_labels = build_forward_return_labels(
            aligned['open'].values, aligned['close'].values, max_horizon
        )
        for family, values in stock_labels.items():
            label_arrays[family][sidx, :, :] = values

    # 准备任务
    fill_tasks = [(code, df, code_to_idx[code], FEATURE_COLS, high_freq_slice, low_freq_slice)
                  for code, df in df_dict.items()]
    n_fill_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_fill_workers) as pool:
        futs = [pool.submit(_fill_one_stock, task) for task in fill_tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="填充数组", mininterval=10):
            fut.result()

    # 填充市场整体属性（向量化计算，避免O(N*D)循环）
    if getattr(config, 'use_market_features', True):
        print("计算市场整体属性...")
        # 1. 构建收盘价矩阵(num_stocks, num_dates) 用于向量化计算宽度特征
        close_matrix = np.full((num_stocks, num_dates), np.nan, dtype=np.float32)
        def _fill_close(args):
            code, df, sidx = args
            stock_idx_inner = np.array([date_to_idx[d] for d in df.index], dtype=np.int32)
            close_matrix[sidx, stock_idx_inner] = df['close'].values.astype(np.float32)

        close_tasks = [(code, df, code_to_idx[code]) for code, df in df_dict.items()]
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            for _ in tqdm(pool.map(_fill_close, close_tasks), total=len(close_tasks),
                          desc="收盘价矩阵", mininterval=10):
                pass

        # 2. 向量化计算宽度特征(advance_decline, new_high_ratio, return_dispersion)
        breadth = compute_breadth_from_close_matrix(close_matrix)  # (num_dates, 3)
        del close_matrix  # 释放内存

        # 3. 指数特征
        idx_feat = build_market_features_index_only(config.data_dir, all_dates)

        # 4. 合并并填充市场状态特征
        # MARKET_COLS顺序: 16宽基 + 3宽度 + 31行业收益 = 50
        # idx_feat列顺序: 16宽基 + 31行业收益 = 47
        idx_feat_cols = list(idx_feat.columns)
        for t_idx, date in enumerate(all_dates):
            if date in idx_feat.index:
                row = idx_feat.loc[date].values
                # 前16列 宽基指数特征
                risk_raw_array[:, t_idx, N_STOCK_RISK:N_STOCK_RISK+16] = row[:16]
                # 中间3列 宽度特征
                risk_raw_array[:, t_idx, N_STOCK_RISK+16:N_STOCK_RISK+19] = breadth[t_idx]
                # 后31列 行业指数收益
                risk_raw_array[:, t_idx, N_STOCK_RISK+19:N_STOCK_RISK+N_MARKET] = row[16:]
        del breadth
        print(f"已加载市场整体属性 {MARKET_COLS}")

    if getattr(config, 'use_macro_features', False):
        print("加载宏观/资金流特征到市场状态...")
        try:
            from data.macro_factors import build_macro_features
            macro_df = build_macro_features(all_dates)
            macro_start = N_STOCK_RISK + N_MARKET
            for j, col in enumerate(MACRO_COLS):
                if col in macro_df.columns:
                    vals = macro_df[col].reindex(all_dates).fillna(0).values.astype(np.float32)
                    risk_raw_array[:, :, macro_start + j] = vals[None, :]
            print(f"已加载宏观/资金流特征 {MACRO_COLS}")
        except Exception as e:
            print(f"宏观/资金流特征加载失败，使用0填充: {e}")

    # 释放不再需要的大对象，为截面构建腾出内存
    import gc
    del df_dict
    gc.collect()

    # ========== 7. 预计算截面特征 + 保存矩阵到磁盘 memmap ==========
    min_stocks = getattr(config, 'min_stocks_per_time', 30)
    # vol_60d needs 60d history + sma20 aggregation needs 20d → min 80d
    min_history = max(seq_len, 80)
    valid_times = list(range(min_history, num_dates))
    print(f"写入特征矩阵到磁盘 memmap ({num_stocks}只 × {num_dates}天)...")

    feat_path = os.path.join(cache_dir, cache_key + "_feat.dat")
    risk_path = os.path.join(cache_dir, cache_key + "_risk.dat")
    _write_memmap(feat_path, feat_array)
    _write_memmap(risk_path, risk_raw_array)
    for values in label_arrays.values():
        values.flush()

    # 预计算全部截面特征（X_norm + risk_factors + labels），存 int16
    print(f"预计算截面特征 ({len(valid_times)} 个截面)...")
    _risk_cont_dim = N_STOCK_RISK + N_MARKET + (len(MACRO_COLS) if getattr(config, 'use_macro_features', False) else 0)
    x_norm_path, risk_full_path, y_norm_path, norm_label_paths, x_dim, risk_full_dim = \
        _precompute_all(
            feat_array, risk_raw_array, industry_array, label_arrays,
            valid_times, high_agg_dim, n_industries, FEATURE_COLS,
            max_horizon, getattr(config, 'target_horizon', 5),
            getattr(config, 'residualize_labels', False),
            cache_key, cache_dir,
            min_stocks=getattr(config, 'min_stocks_per_time', 30),
        )
    print(f"预计算完成: X_norm={x_dim}维, risk_full={risk_full_dim}维")

    del feat_array, risk_raw_array, label_arrays
    gc.collect()

    split = int(len(valid_times) * 0.8)
    train_indices = valid_times[:split]
    val_indices = valid_times[split:]

    label_families = {
        family: {
            'raw_path': raw_label_paths[family],
            'norm_path': norm_label_paths[family],
            'date_shift': 0,
        }
        for family in PHYSICAL_LABEL_FAMILIES
    }
    label_families['oo_lag1'] = {'alias_of': 'oo', 'date_shift': 1}
    metadata = {
        'feat_path': feat_path, 'risk_path': risk_path, 'ret_path': raw_label_paths['cc'],
        'x_norm_path': x_norm_path, 'risk_full_path': risk_full_path,
        'y_norm_path': y_norm_path, 'y_seq_norm_path': norm_label_paths['cc'],
        'label_schema_version': 1, 'label_families': label_families,
        'x_dim': x_dim, 'risk_full_dim': risk_full_dim,
        'risk_cont_dim': _risk_cont_dim,
        'industry_array': industry_array,
        'all_codes': all_codes, 'all_dates': all_dates,
        'train_indices': train_indices, 'val_indices': val_indices,
        'n_industries': n_industries, 'feature_cols': FEATURE_COLS,
        'high_agg_dim': high_agg_dim, 'low_agg_dim': low_agg_dim,
        'high_feat_dim': HIGH_FREQ_COUNT,
        'target_horizon': getattr(config, 'target_horizon', 5),
        'max_horizon': max_horizon, 'min_stocks': min_stocks,
        'residualize': getattr(config, 'residualize_labels', False),
    }

    if use_cache:
        meta_path = os.path.join(cache_dir, cache_key + "_meta.pkl")
        print(f"保存元数据: {meta_path}")
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    return metadata


def _normalize_label_family(values, ind_ids, size_proxy, residualize, min_stocks):
    """Normalize each horizon independently while preserving missing cells."""
    result = np.full(values.shape, np.int16(-32768), dtype=np.int16)
    for h in range(values.shape[1]):
        valid = np.isfinite(values[:, h])
        if int(valid.sum()) < min_stocks:
            continue
        y_h = values[valid, h].astype(np.float32, copy=True)
        if residualize:
            y_h = _residualize_labels(
                y_h[:, None], ind_ids[valid], size_proxy[valid]
            )[:, 0]
        p_low, p_high = np.percentile(y_h, [1, 99])
        y_h = np.clip(y_h, p_low, p_high)
        y_h = (y_h - np.mean(y_h)) / (np.std(y_h) + 1e-8)
        result[valid, h] = (y_h * 1000).clip(-32767, 32767).astype(np.int16)
    return result


def _precompute_one_date(args):
    """Precompute one feature cross-section and all label families."""
    (t, feat_array, risk_raw_array, industry_array, label_arrays,
     high_agg_dim, n_industries, relative_indices, max_horizon,
     target_horizon, residualize, min_stocks) = args

    X_t_all = feat_array[:, t, :]
    risk_all = risk_raw_array[:, t, :]
    ind_all = industry_array[:, t]

    valid = (~np.isnan(X_t_all).any(axis=1)
             & ~np.isnan(risk_all).any(axis=1))
    if valid.sum() < min_stocks:
        return None

    valid_idx = np.where(valid)[0]
    X_t = X_t_all[valid_idx]
    risk_vals = risk_all[valid_idx]
    ind_ids = ind_all[valid_idx]

    # Rank
    denom = max(X_t.shape[0] - 1, 1)
    X_rank = np.argsort(np.argsort(X_t[:, :high_agg_dim], axis=0), axis=0).astype(np.float32) / denom

    # Industry relative
    n_relative = len(relative_indices)
    industry_relative = np.zeros((X_t.shape[0], n_relative), dtype=np.float32)
    if n_industries > 0:
        for j, feat_idx in enumerate(relative_indices):
            feat_vals = X_t[:, feat_idx].copy()
            for ind in range(n_industries):
                mask_ind = ind_ids == ind
                if mask_ind.sum() > 1:
                    feat_vals[mask_ind] -= np.mean(feat_vals[mask_ind])
            unknown_mask = ind_ids == -1
            if unknown_mask.sum() > 1:
                feat_vals[unknown_mask] -= np.mean(feat_vals[unknown_mask])
            industry_relative[:, j] = feat_vals

    # Normalize and assemble
    X_norm_t, risk_factors_t = _normalize_and_assemble(
        X_t, X_rank, industry_relative, risk_vals, ind_ids, n_industries)

    normalized_labels = {
        family: _normalize_label_family(
            values[valid_idx, t, :], ind_ids, risk_vals[:, 0],
            residualize, min_stocks,
        )
        for family, values in label_arrays.items()
    }
    target_h = min(target_horizon - 1, max_horizon - 1)
    y_label = normalized_labels['cc'][:, target_h]

    scale = 1000
    return (t, valid_idx,
            (X_norm_t * scale).clip(-32767, 32767).astype(np.int16),
            (risk_factors_t * scale).clip(-32767, 32767).astype(np.int16),
            y_label, normalized_labels)


def _precompute_all(feat_array, risk_raw_array, industry_array, label_arrays,
                    valid_times, high_agg_dim, n_industries, feature_cols,
                    max_horizon, target_horizon, residualize, cache_key, cache_dir,
                    min_stocks=30):
    """预计算所有截面的 X_norm, risk_factors, y_label, y_seq_norm，存为 int16 memmap。

    使用 ThreadPoolExecutor 并行处理日期——每个日期截面独立，NumPy 释放 GIL。

    Returns (x_path, r_path, y_path, ys_path, x_dim, risk_full_dim).
    """
    num_stocks, num_dates, agg_dim = feat_array.shape
    risk_cont_dim = risk_raw_array.shape[2]
    x_dim = agg_dim + high_agg_dim + len(INDUSTRY_REL_FEATURES)
    risk_full_dim = risk_cont_dim  # 行业信息由 industry_ids 单独传递

    x_path = os.path.join(cache_dir, cache_key + "_X_norm.dat")
    r_path = os.path.join(cache_dir, cache_key + "_risk_full.dat")
    y_path = os.path.join(cache_dir, cache_key + "_y_norm.dat")
    norm_paths = {
        family: os.path.join(cache_dir, cache_key + f"_label_{family}_norm.dat")
        for family in PHYSICAL_LABEL_FAMILIES
    }

    X_mm = np.memmap(x_path, dtype=np.int16, mode='w+', shape=(num_stocks, num_dates, x_dim))
    R_mm = np.memmap(r_path, dtype=np.int16, mode='w+', shape=(num_stocks, num_dates, risk_full_dim))
    Y_mm = np.memmap(y_path, dtype=np.int16, mode='w+', shape=(num_stocks, num_dates))
    label_mmaps = {
        family: np.memmap(path, dtype=np.int16, mode='w+',
                          shape=(num_stocks, num_dates, max_horizon))
        for family, path in norm_paths.items()
    }
    INVALID = np.int16(-32768)
    X_mm[:] = INVALID
    R_mm[:] = INVALID
    Y_mm[:] = INVALID
    for mm in label_mmaps.values():
        mm[:] = INVALID

    relative_indices = [feature_cols.index(name) for name in INDUSTRY_REL_FEATURES if name in feature_cols]

    # 构建参数列表（共享数组只传引用，不复制）
    base_args = (feat_array, risk_raw_array, industry_array, label_arrays,
                 high_agg_dim, n_industries, relative_indices, max_horizon,
                 target_horizon, residualize, min_stocks)
    tasks = [(t,) + base_args for t in valid_times]

    n_workers = min(12, (os.cpu_count() or 4) + 4)
    n_done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_precompute_one_date, task): task[0] for task in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                t, valid_idx, X_t, R_t, Y_t, labels_t = result
                X_mm[valid_idx, t, :] = X_t
                R_mm[valid_idx, t, :] = R_t
                Y_mm[valid_idx, t] = Y_t
                for family, values in labels_t.items():
                    label_mmaps[family][valid_idx, t, :] = values
            n_done += 1
            if n_done % 500 == 0:
                print(f"  预计算进度: {n_done}/{len(valid_times)}")

    X_mm.flush(); R_mm.flush(); Y_mm.flush()
    for mm in label_mmaps.values():
        mm.flush()
    del X_mm, R_mm, Y_mm, label_mmaps

    return x_path, r_path, y_path, norm_paths, x_dim, risk_full_dim


def _write_memmap(path, arr):
    """将 numpy 数组写入 memmap 文件（float32/int16）。"""
    mm = np.memmap(path, dtype=arr.dtype, mode='w+', shape=arr.shape)
    mm[:] = arr[:]
    mm.flush()
    del mm


def _open_memmap(path, dtype, shape):
    """以只读模式打开 memmap 文件。"""
    return np.memmap(path, dtype=dtype, mode='r', shape=shape)


def samples_from_precomputed_metadata(meta, split='val', *, time_indices=None, require_labels=True):
    """Convert precomputed memmap metadata back to legacy cross-section samples.

    This is a compatibility adapter for backtest/recommendation code that still
    expects a list of dict samples instead of the lightweight metadata returned
    by build_cross_section_dataset().
    """
    n_stocks = len(meta['all_codes'])
    n_dates = len(meta['all_dates'])
    x_mm = _open_memmap(meta['x_norm_path'], np.int16, (n_stocks, n_dates, meta['x_dim']))
    r_mm = _open_memmap(meta['risk_full_path'], np.int16, (n_stocks, n_dates, meta['risk_full_dim']))
    y_mm = _open_memmap(meta['y_norm_path'], np.int16, (n_stocks, n_dates))
    ys_mm = _open_memmap(meta['y_seq_norm_path'], np.int16, (n_stocks, n_dates, meta['max_horizon']))

    if time_indices is None:
        if split == 'train':
            time_indices = meta['train_indices']
        elif split == 'val':
            time_indices = meta['val_indices']
        elif split == 'all':
            time_indices = list(meta['train_indices']) + list(meta['val_indices'])
        else:
            raise ValueError(f"unknown split: {split}")

    all_codes = np.asarray(meta['all_codes'])
    all_dates = list(meta['all_dates'])
    industry_array = meta['industry_array']
    samples = []
    invalid = np.int16(-32768)
    scale = 1000.0

    for t in time_indices:
        # Inference must never use future-label availability as a universe
        # filter. Feature/risk sentinels describe what was knowable on date t.
        valid = (
            y_mm[:, t] != invalid
            if require_labels
            else ((x_mm[:, t, 0] != invalid) & (r_mm[:, t, 0] != invalid))
        )
        if int(valid.sum()) < meta.get('min_stocks', 30):
            continue
        valid_idx = np.where(valid)[0]
        if require_labels:
            y = y_mm[valid_idx, t].astype(np.float32) / scale
            y_seq = ys_mm[valid_idx, t, :].astype(np.float32) / scale
        else:
            y = np.zeros(len(valid_idx), dtype=np.float32)
            y_seq = np.zeros((len(valid_idx), meta['max_horizon']), dtype=np.float32)
        samples.append({
            'date': all_dates[t],
            'X': x_mm[valid_idx, t, :].astype(np.float32) / scale,
            'y': y,
            'y_seq': y_seq,
            'codes': all_codes[valid_idx].tolist(),
            'raw_y': y.copy(),
            'risk': r_mm[valid_idx, t, :].astype(np.float32) / scale,
            'industry_ids': industry_array[valid_idx, t].astype(np.int64),
        })
    return samples


def add_technical_features(df: "pd.DataFrame", config) -> "pd.DataFrame":
    """Add normalized technical indicators for cross-stock comparison"""
    close = df['close'].where(df['close'] > 0)
    high = df['high']
    low = df['low']
    volume = df['volume'].clip(lower=0)

    for period in config.sma_periods:
        sma = close.rolling(window=period).mean()
        df[f'sma{period}_gap'] = (close / sma - 1).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['ema12_gap'] = (close / ema_12 - 1).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
    df['ema26_gap'] = (close / ema_26 - 1).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=config.rsi_period).mean()
    avg_loss = loss.rolling(window=config.rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    df['rsi_norm'] = (rsi / 100 - 0.5).clip(-0.5, 0.5)

    ema_fast = close.ewm(span=config.macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=config.macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=config.macd_signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    df['macd_pct'] = (macd / close).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
    df['macd_signal_pct'] = (macd_signal / close).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
    df['macd_diff_pct'] = (macd_diff / close).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    df['atr_pct'] = (atr / close).replace([np.inf, -np.inf], np.nan).clip(0, 1)

    df['volume_ratio'] = (volume / volume.rolling(20).mean()).replace([np.inf, -np.inf], np.nan).clip(0, 20)

    return df


def _inference_cache_path_for(config, max_lookback=None, stock_universe=None, data_end_date=None):
    data_dir = Path(config.data_dir).resolve()
    dependencies = list(data_dir.rglob("*.csv"))
    dependencies.extend(
        path
        for path in (
            Path("cache/fundamental_features_akshare.parquet"),
            Path("cache/shareholder_features.parquet"),
            Path("cache/restricted_features.parquet"),
            Path("cache/north_flow.csv"),
            Path("cache/margin_balance.csv"),
            Path("cache/pmi_pit_v2.csv"),
        )
        if path.exists()
    )
    dependency_signature = "|".join(
        f"{path.resolve()}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
        for path in sorted(dependencies, key=lambda item: str(item))
    )
    feature_signature = "|".join(
        f"{name}={int(bool(getattr(config, name, False)))}"
        for name in (
            "use_technical_features",
            "use_market_features",
            "use_macro_features",
            "use_fundamental_features",
            "use_fundamental_quality_features",
            "use_shareholder_features",
            "use_restricted_features",
        )
    )
    data_tag = hashlib.sha1(
        f"{data_dir}|{feature_signature}|{dependency_signature}".encode("utf-8")
    ).hexdigest()[:10]
    lookback_tag = "all" if max_lookback is None else str(int(max_lookback))
    end_tag = "latest" if data_end_date is None else pd.Timestamp(data_end_date).strftime("%Y%m%d")
    if stock_universe:
        universe_text = "|".join(sorted(str(code) for code in stock_universe))
        universe_tag = hashlib.sha1(universe_text.encode("utf-8")).hexdigest()[:8]
    else:
        universe_tag = "all"
    return os.path.join(
        "cache",
        f"inference_matrices_{data_tag}_lb{lookback_tag}_u{universe_tag}_end{end_tag}.pkl",
    )


def _save_inference_cache(matrices, cache_path):
    import pickle
    try:
        os.makedirs("cache", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(matrices, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _load_inference_cache(cache_path):
    import pickle
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


def _build_inference_matrices(
    config,
    stock_universe=None,
    max_lookback=None,
    data_end_date=None,
):
    """Load CSVs, compute features, and build the full inference matrices once.

    Returns a dict with all shared data used to produce cross-section samples.
    If max_lookback is provided, only keep the most recent N dates to save memory.
    """
    data_end_date = pd.Timestamp(data_end_date) if data_end_date is not None else None
    cache_path = _inference_cache_path_for(
        config,
        max_lookback,
        stock_universe,
        data_end_date=data_end_date,
    )
    cached = _load_inference_cache(cache_path)
    if cached is not None:
        print(f"加载推理矩阵缓存: {cache_path}")
        return cached

    data_dir = config.data_dir
    seq_len = config.seq_len
    max_horizon = getattr(config, 'max_horizon', 10)
    test_n = getattr(config, 'test_stocks', None) if getattr(config, 'test_mode', False) else None

    print("读取股票数据...")
    excluded = {'all_data_jq.csv', 'stable_stocks.csv', 'stable_stocks_industry.csv'}
    csv_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and f not in excluded and f[0].isdigit()
    )
    if config.max_stocks:
        csv_files = csv_files[:config.max_stocks]
    if test_n:
        csv_files = csv_files[:test_n]
        print(f"test mode: loading only {len(csv_files)} stocks")

    df_dict = {}
    for fname in tqdm(csv_files, desc="加载CSV", mininterval=10):
        code = fname.replace('.csv', '')
        if stock_universe and code not in stock_universe:
            continue
        file_path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
            if 'code' in df.columns:
                df.drop(columns=['code'], inplace=True)
        except Exception:
            continue
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            continue
        df = df.sort_index()
        if data_end_date is not None:
            df = df[df.index <= data_end_date]
        if len(df) >= seq_len + 50:
            df_dict[code] = df

    if not df_dict:
        raise ValueError("没有有效股票数据")
    print(f"有效股票数 {len(df_dict)}")

    # 早期截断：max_lookback 时仅保留近期数据再计算特征（大幅加速推理）
    all_dates_raw = sorted(set().union(*[df.index for df in df_dict.values()]))
    num_dates_raw = len(all_dates_raw)
    if max_lookback is not None and num_dates_raw > max_lookback:
        buffer = 80
        cutoff_idx = max(0, num_dates_raw - max_lookback - buffer)
        cutoff_date = all_dates_raw[cutoff_idx]
        for code in list(df_dict.keys()):
            df = df_dict[code]
            df = df[df.index >= cutoff_date]
            if len(df) < seq_len:
                del df_dict[code]
            else:
                df_dict[code] = df
        print(f"推理截断: 仅保留 {cutoff_date.date()} 之后数据 ({len(df_dict)} 只股票)")

    # 在截断后的数据上计算技术特征
    if config.use_technical_features:
        for code in list(df_dict.keys()):
            df_dict[code] = add_technical_features(df_dict[code], config)

    print("构造多尺度特征...")
    base_features = _compute_base_features(df_dict)

    feature_cols = base_features + TECH_FEATURES if config.use_technical_features else base_features
    feature_cols = list(dict.fromkeys(feature_cols))
    all_dates = sorted(set().union(*[df.index for df in df_dict.values()]))
    extra_feat_cols = _load_extra_features(config, df_dict, all_dates)
    feature_cols = feature_cols + extra_feat_cols
    base_feat_dim = len(feature_cols)
    high_freq_count = base_feat_dim - len(extra_feat_cols)  # 23
    high_agg_dim = high_freq_count * N_AGGS
    low_agg_dim = len(extra_feat_cols) * 2
    agg_feat_dim = high_agg_dim + low_agg_dim
    print(f"最终特征列数 {base_feat_dim}, 聚合特征数 {agg_feat_dim} (高频{high_agg_dim}+低频{low_agg_dim})")

    # 去除缓冲区
    num_dates = len(all_dates)
    if max_lookback is not None and num_dates > max_lookback:
        cutoff_date = all_dates[-max_lookback]
        for code in list(df_dict.keys()):
            df = df_dict[code]
            df = df[df.index >= cutoff_date]
            if len(df) < seq_len:
                del df_dict[code]
            else:
                df_dict[code] = df
        all_dates = sorted(set().union(*[df.index for df in df_dict.values()]))
        num_dates = len(all_dates)
        print(f"全局日期数 {num_dates} (截断最近 {max_lookback} 天)")
    else:
        print(f"全局日期数 {num_dates}")

    industry_dict, all_industries, industry_to_idx, n_industries = _load_industry_map(data_dir)
    print(f"行业数 {n_industries}")

    all_codes = list(df_dict.keys())
    num_stocks = len(all_codes)
    code_to_idx = {code: i for i, code in enumerate(all_codes)}
    date_to_idx = {date: i for i, date in enumerate(all_dates)}

    feat_array = np.full((num_stocks, num_dates, agg_feat_dim), np.nan, dtype=np.float32)
    macro_dim = len(MACRO_COLS) if getattr(config, 'use_macro_features', False) else 0
    risk_cont_dim = 6 + N_MARKET + macro_dim  # 6 stock risk factors
    risk_raw_array = np.zeros((num_stocks, num_dates, risk_cont_dim), dtype=np.float32)
    industry_array = np.full((num_stocks, num_dates), -1, dtype=np.int16)

    print("填充特征矩阵...")
    for code, df in tqdm(df_dict.items(), desc="填充数组", mininterval=10):
        sidx = code_to_idx[code]
        stock_dates = df.index
        stock_idx = np.array([date_to_idx[d] for d in stock_dates], dtype=np.int32)
        raw_feat = df.reindex(columns=feature_cols).values
        if len(stock_dates) >= seq_len:
            windows = sliding_window_view(raw_feat, seq_len, axis=0)
            if windows.shape[1] != seq_len:
                windows = windows.transpose(0, 2, 1)
            n_windows = windows.shape[0]
            # 高频量价：5种聚合
            last_high = windows[:, -1, :high_freq_count]
            sma5 = windows[:, -5:, :high_freq_count].mean(axis=1) if seq_len >= 5 else last_high
            sma20 = windows[:, -20:, :high_freq_count].mean(axis=1) if seq_len >= 20 else sma5
            vol5 = windows[:, -5:, :high_freq_count].std(axis=1) if seq_len >= 5 else np.zeros_like(last_high)
            vol20 = windows[:, -20:, :high_freq_count].std(axis=1) if seq_len >= 20 else vol5
            high_agg = np.concatenate([last_high, sma5, sma20, vol5, vol20], axis=1)
            # 低频基本面：仅 last + qoq
            last_low = windows[:, -1, high_freq_count:]
            qoq_lookback = min(seq_len // 4, 10)
            qoq_low = last_low - windows[:, -qoq_lookback, high_freq_count:]
            agg_feat = np.concatenate([high_agg, last_low, qoq_low], axis=1)
            agg_dates_idx = stock_idx[seq_len - 1: seq_len - 1 + n_windows]
            feat_array[sidx, agg_dates_idx, :] = agg_feat

        size_vals = df['log_volume'].values.astype(np.float32)
        vol_vals = df['vol_60d'].fillna(0).values.astype(np.float32)
        mom_vals = df['ret_20d'].fillna(0).values.astype(np.float32)
        rev_vals = df['ret_5d'].fillna(0).values.astype(np.float32)
        turn_vals = df['volume_ratio'].fillna(0).values.astype(np.float32)
        amp_vals = df['amplitude'].fillna(0).values.astype(np.float32)
        risk_raw_array[sidx, stock_idx, :N_STOCK_RISK] = np.column_stack(
            [size_vals, vol_vals, mom_vals, rev_vals, turn_vals, amp_vals])

        raw_ind = industry_dict.get(code)
        ind_id = industry_to_idx.get(raw_ind, -1) if raw_ind else -1
        industry_array[sidx, stock_idx] = ind_id

    if getattr(config, 'use_market_features', True):
        print("计算市场整体属性...")
        close_matrix = np.full((num_stocks, num_dates), np.nan, dtype=np.float32)
        for code, df in df_dict.items():
            sidx = code_to_idx[code]
            stock_idx = np.array([date_to_idx[d] for d in df.index], dtype=np.int32)
            close_matrix[sidx, stock_idx] = df['close'].values.astype(np.float32)
        breadth = compute_breadth_from_close_matrix(close_matrix)
        del close_matrix
        idx_feat = build_market_features_index_only(config.data_dir, all_dates)
        for t_idx, date in enumerate(all_dates):
            if date in idx_feat.index:
                row = idx_feat.loc[date].values
                risk_raw_array[:, t_idx, N_STOCK_RISK:N_STOCK_RISK+16] = row[:16]
                risk_raw_array[:, t_idx, 19:22] = breadth[t_idx]
                risk_raw_array[:, t_idx, 22:3 + N_MARKET] = row[16:]
        del breadth
        print(f"已加载市场整体属性 {MARKET_COLS}")

    if getattr(config, 'use_macro_features', False):
        print("加载宏观/资金流特征到市场状态...")
        try:
            from data.macro_factors import build_macro_features
            macro_df = build_macro_features(all_dates)
            macro_start = N_STOCK_RISK + N_MARKET
            for j, col in enumerate(MACRO_COLS):
                if col in macro_df.columns:
                    vals = macro_df[col].reindex(all_dates).fillna(0).values.astype(np.float32)
                    risk_raw_array[:, :, macro_start + j] = vals[None, :]
            print(f"已加载宏观/资金流特征 {MACRO_COLS}")
        except Exception as e:
            print(f"宏观/资金流特征加载失败，使用0填充: {e}")

    all_codes_np = np.array(all_codes)
    matrices = {
        'feat_array': feat_array,
        'risk_raw_array': risk_raw_array,
        'industry_array': industry_array,
        'all_dates': all_dates,
        'all_codes': all_codes_np,
        'n_industries': n_industries,
        'industry_to_idx': industry_to_idx,
        'feature_cols': feature_cols,
        'high_agg_dim': high_agg_dim,
        'seq_len': seq_len,
        'max_horizon': max_horizon,
        'min_stocks': getattr(config, 'min_stocks_per_time', 30),
    }
    _save_inference_cache(matrices, cache_path)
    return matrices


def _sample_from_matrices(m, t_idx):
    """Extract a single cross-section sample at time index t_idx from pre-built matrices m."""
    X_t_all = m['feat_array'][:, t_idx, :]
    risk_all = m['risk_raw_array'][:, t_idx, :]
    ind_all = m['industry_array'][:, t_idx]
    feature_cols = m['feature_cols']
    n_industries = m['n_industries']

    valid_feat = ~np.isnan(X_t_all).any(axis=1)
    valid_risk = ~np.isnan(risk_all).any(axis=1)
    valid = valid_feat & valid_risk
    valid_count = int(valid.sum())
    if valid_count < m['min_stocks']:
        return None

    X_t = X_t_all[valid]
    risk_vals = risk_all[valid]
    ind_ids = ind_all[valid]
    # rank 仅对高频聚合特征
    denom = max(X_t.shape[0] - 1, 1)
    X_rank = np.argsort(np.argsort(X_t[:, :m['high_agg_dim']], axis=0), axis=0).astype(np.float32) / denom

    relative_indices = [feature_cols.index(name) for name in INDUSTRY_REL_FEATURES if name in feature_cols]
    industry_relative = np.zeros((X_t.shape[0], len(relative_indices)), dtype=np.float32)
    if n_industries > 0:
        for j, feat_idx in enumerate(relative_indices):
            feat_vals = X_t[:, feat_idx].copy()
            for ind in range(n_industries):
                mask_ind = ind_ids == ind
                if mask_ind.sum() > 1:
                    feat_vals[mask_ind] -= np.mean(feat_vals[mask_ind])
            unknown_mask = ind_ids == -1
            if unknown_mask.sum() > 1:
                feat_vals[unknown_mask] -= np.mean(feat_vals[unknown_mask])
            industry_relative[:, j] = feat_vals

    X_norm, risk_factors = _normalize_and_assemble(X_t, X_rank, industry_relative, risk_vals, ind_ids, n_industries)

    n = X_norm.shape[0]
    return {
        'date': m['all_dates'][t_idx],
        'X': X_norm,
        'y': np.zeros(n, dtype=np.float32),
        'y_seq': np.zeros((n, m['max_horizon']), dtype=np.float32),
        'codes': m['all_codes'][valid].tolist(),
        'raw_y': np.zeros(n, dtype=np.float32),
        'risk': risk_factors,
        'industry_ids': ind_ids,
    }


def build_inference_sample(config, stock_universe=None, as_of_date=None):
    """Build a single label-free inference cross-section sample. Backward compatible."""
    max_lookback = config.seq_len + 50  # ~90天，只加载近期数据
    explicit_as_of = as_of_date is not None and str(as_of_date).lower() != "latest"
    data_end_date = pd.Timestamp(as_of_date) if explicit_as_of else None
    matrices = _build_inference_matrices(
        config,
        stock_universe,
        max_lookback=max_lookback,
        data_end_date=data_end_date,
    )
    all_dates = matrices['all_dates']
    seq_len = matrices['seq_len']
    num_dates = len(all_dates)

    if as_of_date is None or str(as_of_date).lower() == "latest":
        candidate_indices = range(num_dates - 1, seq_len - 1, -1)
    else:
        cutoff = pd.to_datetime(as_of_date)
        candidate_indices = [i for i, date in enumerate(all_dates) if i >= seq_len and date <= cutoff]
        candidate_indices = reversed(candidate_indices)

    last_error = None
    for t in candidate_indices:
        sample = _sample_from_matrices(matrices, t)
        if sample is None:
            last_error = f"{all_dates[t].date()} 有效股票数不足 < {matrices['min_stocks']}"
            continue
        print(f"推理截面日期 {all_dates[t].date()}，有效股票数 {len(sample['codes'])}")
        return sample

    if as_of_date is None or str(as_of_date).lower() == "latest":
        raise ValueError(last_error or "无法构造最新推理截面")
    raise ValueError(last_error or f"无法在 {as_of_date} 及之前构造推理截面")


def build_inference_samples(config, as_of_dates, stock_universe=None):
    """Build inference samples for multiple dates efficiently (CSVs read once)."""
    as_of_dates = list(as_of_dates)
    if not as_of_dates:
        return []
    n_requested = max(len(as_of_dates), 1)
    max_lookback = int(config.seq_len) + n_requested + 80
    data_end_date = max(pd.Timestamp(date) for date in as_of_dates)
    matrices = _build_inference_matrices(
        config,
        stock_universe,
        max_lookback=max_lookback,
        data_end_date=data_end_date,
    )
    all_dates = matrices['all_dates']
    seq_len = matrices['seq_len']
    all_dates_list = list(all_dates)

    results = []
    for as_of in as_of_dates:
        cutoff = pd.to_datetime(as_of)
        candidates = [i for i, d in enumerate(all_dates_list) if i >= seq_len and d <= cutoff]
        found = None
        for t in reversed(candidates):
            sample = _sample_from_matrices(matrices, t)
            if sample is not None:
                found = sample
                break
        if found is not None:
            actual_date = pd.Timestamp(found['date']).strftime('%Y-%m-%d')
            print(f"  {as_of} -> 实际 {actual_date}, 股票数 {len(found['codes'])}")
            found['_requested_date'] = as_of
            results.append(found)
        else:
            print(f"  {as_of}: SKIP (不足最小股票数)")

    return results
