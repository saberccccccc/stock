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

warnings.filterwarnings('ignore')

# 宏观特征列名
MACRO_COLS = ['north_net_zscore', 'margin_balance_change', 'pmi_zscore']
# 基本面特征列名
FUNDAMENTAL_COLS = ['roe', 'revenue_yoy', 'pe_percentile']
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
CACHE_VERSION = "v13_config_key"


def _cache_config_digest(config, data_dir, stock_universe):
    cache_config = {
        'data_dir': os.path.abspath(data_dir),
        'max_stocks': getattr(config, 'max_stocks', None),
        'test_mode': getattr(config, 'test_mode', False),
        'test_stocks': getattr(config, 'test_stocks', None),
        'seq_len': getattr(config, 'seq_len', None),
        'target_horizon': getattr(config, 'target_horizon', None),
        'max_horizon': getattr(config, 'max_horizon', None),
        'min_stocks_per_time': getattr(config, 'min_stocks_per_time', None),
        'normalize_features': getattr(config, 'normalize_features', None),
        'use_multi_horizon': getattr(config, 'use_multi_horizon', None),
        'horizon_indices': tuple(getattr(config, 'horizon_indices', ())),
        'horizon_weights': tuple(getattr(config, 'horizon_weights', ())),
        'use_technical_features': getattr(config, 'use_technical_features', False),
        'use_market_features': getattr(config, 'use_market_features', False),
        'use_fundamental_features': getattr(config, 'use_fundamental_features', False),
        'use_macro_features': getattr(config, 'use_macro_features', False),
        'stock_universe': tuple(sorted(stock_universe)) if stock_universe else None,
        'agg_names': tuple(AGG_NAMES),
        'industry_rel_features': tuple(INDUSTRY_REL_FEATURES),
        'tech_features': tuple(TECH_FEATURES),
        'fundamental_cols': tuple(FUNDAMENTAL_COLS),
        'macro_cols': tuple(MACRO_COLS),
    }
    payload = repr(sorted(cache_config.items())).encode('utf-8')
    return hashlib.md5(payload).hexdigest()[:10]


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

    # ----- 基本面因子（季频前向填充到日频）-----
    if getattr(config, 'use_fundamental_features', False):
        try:
            from data.fundamental_factors import fetch_fundamentals, merge_to_daily
            codes = list(df_dict.keys())
            # 注意：需要用户Token，未设置时跳过
            token = getattr(config, 'tushare_token', None) or os.environ.get('TUSHARE_TOKEN')
            if token:
                funda_df = fetch_fundamentals(codes, token)
                if not funda_df.empty:
                    funda_daily = merge_to_daily(funda_df, codes, all_dates)
                    for code in codes:
                        clean = code.replace('.SH', '').replace('.SZ', '')
                        for col in FUNDAMENTAL_COLS:
                            citem = f'{code}_{col}'
                            alt_item = f'{clean}_{col}'
                            if citem in funda_daily.columns:
                                df_dict[code][f'fund_{col}'] = funda_daily[citem].reindex(
                                    df_dict[code].index, method='ffill').fillna(0).values
                            elif alt_item in funda_daily.columns:
                                df_dict[code][f'fund_{col}'] = funda_daily[alt_item].reindex(
                                    df_dict[code].index, method='ffill').fillna(0).values
                            else:
                                df_dict[code][f'fund_{col}'] = 0.0
                    extra_feat_cols += [f'fund_{c}' for c in FUNDAMENTAL_COLS]
                    print(f"已加载基本面特征: {[f'fund_{c}' for c in FUNDAMENTAL_COLS]}")
            else:
                print("提示: 未设置tushare_token，跳过基本面因子加载")
        except Exception as e:
            print(f"基本面特征加载失败 {e}")

    return extra_feat_cols


def build_cross_section_dataset(config, stock_universe=None, use_cache=True):
    """Build cross-section dataset, return (train_samples, val_samples).

    相较v6版本的内存优化：
    - 不缓存全局特征矩阵 (num_stocks × num_dates × feat_dim)
    - 改用窗口索引预计算 + 按需切片
    - 保留截面rank特征和行业相对特征
    """
    data_dir = config.data_dir
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
        features.append("funda")
    if config.use_macro_features:
        features.append("macro")
    feat_str = "_".join(features) if features else "basic"

    universe_tag = ""
    if stock_universe:
        universe_digest = hashlib.md5("|".join(sorted(stock_universe)).encode()).hexdigest()[:8]
        universe_tag = f"_universe{len(stock_universe)}_{universe_digest}"
    config_digest = _cache_config_digest(config, data_dir, stock_universe)
    cache_key = f"cross_section_{CACHE_VERSION}_{n_stocks}stocks{universe_tag}_{feat_str}_seq{seq_len}_target{config.target_horizon}_maxh{max_horizon}_cfg{config_digest}"
    cache_file = os.path.join(cache_dir, cache_key + ".pkl")
    if use_cache and os.path.exists(cache_file) and not config.force_rebuild:
        print(f"加载缓存: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

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
        if config.use_technical_features:
            # add_technical_features defined in this module
            df = add_technical_features(df, config)
        if len(df) >= seq_len + max_horizon + 50:
            df_dict[code] = df

    if not df_dict:
        raise ValueError("没有有效股票数据")
    print(f"有效股票数 {len(df_dict)}")

    # ========== 2. 构造多尺度特征 ==========
    print("构造多尺度特征...")
    # V8: 减少冗余，增加微观结构特征
    BASE_FEATURES = [
        'ret_5d', 'ret_20d', 'vol_10d', 'vol_60d',
        'price_momentum', 'log_volume', 'volume_spike',
        'upper_shadow', 'lower_shadow', 'body_size', 'gap', 'amplitude'
    ]

    for code, df in df_dict.items():
        close_safe = df['close'].where(df['close'] > 0)
        prev_close_safe = df['close'].shift(1).where(df['close'].shift(1) > 0)
        df['log_close'] = np.log(close_safe).replace([np.inf, -np.inf], np.nan)
        df['log_volume'] = np.log(df['volume'].clip(lower=0) + 1)

        # 收益率和波动率
        daily_ret = close_safe.pct_change().replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
        df['ret_5d'] = close_safe.pct_change(5).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        df['ret_20d'] = close_safe.pct_change(20).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        df['vol_10d'] = daily_ret.rolling(10).std().clip(0, 1.0)
        df['vol_60d'] = daily_ret.rolling(60).std().clip(0, 1.0)
        df['price_momentum'] = (close_safe / close_safe.rolling(20).mean() - 1).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        df['volume_spike'] = df['log_volume'].pct_change(1).abs().replace([np.inf, -np.inf], np.nan).clip(0, 10)

        # 微观结构特征
        hl_range_raw = df['high'] - df['low']
        valid_range = hl_range_raw > (close_safe * 1e-4)
        hl_range = hl_range_raw.where(valid_range)
        df['upper_shadow'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / hl_range).clip(0, 1)
        df['lower_shadow'] = ((df[['open', 'close']].min(axis=1) - df['low']) / hl_range).clip(0, 1)
        df['body_size'] = (abs(df['close'] - df['open']) / hl_range).clip(0, 1)
        df['gap'] = ((df['open'] - df['close'].shift(1)) / prev_close_safe).replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
        df['amplitude'] = (hl_range_raw / close_safe).replace([np.inf, -np.inf], np.nan).clip(0, 1)

    if config.use_technical_features:
        FEATURE_COLS = BASE_FEATURES + TECH_FEATURES
    else:
        FEATURE_COLS = BASE_FEATURES

    FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))  # 去重保序
    base_feat_dim = len(FEATURE_COLS)
    agg_feat_dim = base_feat_dim * N_AGGS
    print(f"基础特征数 {base_feat_dim}, 聚合方式: {N_AGGS}种，聚合特征数 {agg_feat_dim}")

    # ========== 3. 加载扩展因子 ==========
    all_dates = sorted(set().union(*[df.index for df in df_dict.values()]))
    extra_feat_cols = _load_extra_features(config, df_dict, all_dates)
    FEATURE_COLS = FEATURE_COLS + extra_feat_cols
    base_feat_dim = len(FEATURE_COLS)
    agg_feat_dim = base_feat_dim * N_AGGS
    print(f"最终特征列数 {base_feat_dim}, 聚合特征数 {agg_feat_dim}")

    # ========== 4. 全局日期并集 ==========
    num_dates = len(all_dates)
    print(f"全局日期数 {num_dates}")

    # ========== 5. 行业数据 ==========
    industry_file = os.path.join(os.path.dirname(data_dir), "stock_industry.csv")
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
    risk_cont_dim = 3 + N_MARKET + macro_dim
    risk_raw_array = np.zeros((num_stocks, num_dates, risk_cont_dim), dtype=np.float32)
    industry_array = np.full((num_stocks, num_dates), -1, dtype=np.int16)
    ret_seq_array = np.full((num_stocks, num_dates, max_horizon), np.nan, dtype=np.float32)

    print("填充特征矩阵...")
    for code, df in tqdm(df_dict.items(), desc="填充数组", mininterval=10):
        sidx = code_to_idx[code]
        stock_dates = df.index
        stock_idx = np.array([date_to_idx[d] for d in stock_dates], dtype=np.int32)
        T_stock = len(stock_dates)

        # V7特征聚合：5种聚合替代原3种last/trend/vol
        raw_feat = df.reindex(columns=FEATURE_COLS).values
        if T_stock >= seq_len:
            windows = sliding_window_view(raw_feat, seq_len, axis=0)
            if windows.shape[1] != seq_len:
                windows = windows.transpose(0, 2, 1)
            n_windows = windows.shape[0]

            # 5种多尺度聚合
            last_val = windows[:, -1, :]                          # 鏈EUR鏂板EUR?
            sma5 = windows[:, -5:, :].mean(axis=1) if seq_len >= 5 else last_val
            sma20 = windows[:, -20:, :].mean(axis=1) if seq_len >= 20 else sma5
            vol5 = windows[:, -5:, :].std(axis=1) if seq_len >= 5 else np.zeros_like(last_val)
            vol20 = windows[:, -20:, :].std(axis=1) if seq_len >= 20 else vol5
            agg_feat = np.concatenate([last_val, sma5, sma20, vol5, vol20], axis=1)

            agg_dates_idx = stock_idx[seq_len - 1: seq_len - 1 + n_windows]
            feat_array[sidx, agg_dates_idx, :] = agg_feat

        # 连续风险因子
        size_vals = df['log_volume'].values.astype(np.float32)
        vol_vals = df['vol_60d'].fillna(0).values.astype(np.float32)
        mom_vals = df['ret_20d'].fillna(0).values.astype(np.float32)
        risk_raw_array[sidx, stock_idx, :3] = np.column_stack([size_vals, vol_vals, mom_vals])

        # 行业ID
        raw_ind = industry_dict.get(code)
        ind_id = industry_to_idx.get(raw_ind, -1) if raw_ind else -1
        industry_array[sidx, stock_idx] = ind_id

        # 多期收益率
        close_vals = df['close'].values
        if T_stock >= max_horizon + 1:
            price_windows = sliding_window_view(close_vals, max_horizon + 1, axis=0)
            n_ret = price_windows.shape[0]
            ret_dates_idx = stock_idx[:n_ret]
            for h in range(1, max_horizon + 1):
                ret = (price_windows[:, h] - price_windows[:, 0]) / price_windows[:, 0]
                ret_seq_array[sidx, ret_dates_idx, h - 1] = ret

    # 填充市场整体属性（向量化计算，避免O(N*D)循环）
    if getattr(config, 'use_market_features', True):
        print("计算市场整体属性...")
        # 1. 构建收盘价矩阵(num_stocks, num_dates) 用于向量化计算宽度特征
        close_matrix = np.full((num_stocks, num_dates), np.nan, dtype=np.float32)
        for code, df in df_dict.items():
            sidx = code_to_idx[code]
            stock_dates = df.index
            stock_idx = np.array([date_to_idx[d] for d in stock_dates], dtype=np.int32)
            close_matrix[sidx, stock_idx] = df['close'].values.astype(np.float32)

        # 2. 向量化计算宽度特征(advance_decline, new_high_ratio, return_dispersion)
        breadth = compute_breadth_from_close_matrix(close_matrix)  # (num_dates, 3)
        del close_matrix  # 释放内存

        # 3. 指数特征
        idx_feat = build_market_features_index_only(config.data_dir, all_dates)

        # 4. 合并并填充市场状态特征
        # MARKET_COLS顺序: 16个宽基指数+ 3个宽度特征+ 31个行业收益+ 31个行业可用性mask
        # idx_feat列顺序 16个宽基指数+ 31个行业收益+ 31个行业可用性mask
        idx_feat_cols = list(idx_feat.columns)
        for t_idx, date in enumerate(all_dates):
            if date in idx_feat.index:
                row = idx_feat.loc[date].values
                # 前16列 宽基指数特征
                risk_raw_array[:, t_idx, 3:19] = row[:16]
                # 中间3列 宽度特征
                risk_raw_array[:, t_idx, 19:22] = breadth[t_idx]
                # 后62列 行业指数收益 + 可用性mask
                risk_raw_array[:, t_idx, 22:3 + N_MARKET] = row[16:]
        del breadth
        print(f"已加载市场整体属性 {MARKET_COLS}")

    if getattr(config, 'use_macro_features', False):
        print("加载宏观/资金流特征到市场状态...")
        try:
            from data.macro_factors import build_macro_features
            macro_df = build_macro_features(all_dates)
            macro_start = 3 + N_MARKET
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

    # ========== 7. 构建截面样本 ==========
    print("构建截面样本...")
    X_samples = []
    min_stocks = getattr(config, 'min_stocks_per_time', 30)
    all_codes_np = np.array(all_codes)

    for t in tqdm(range(seq_len, num_dates - max_horizon), desc="构建截面", mininterval=10):
        X_t_all = feat_array[:, t, :]
        y_seq_all = ret_seq_array[:, t, :]
        risk_all = risk_raw_array[:, t, :]
        ind_all = industry_array[:, t]

        valid_feat = ~np.isnan(X_t_all).any(axis=1)
        valid_ret = ~np.isnan(y_seq_all).any(axis=1)
        valid_risk = ~np.isnan(risk_all).any(axis=1)
        valid = valid_feat & valid_ret & valid_risk
        if valid.sum() < min_stocks:
            continue

        X_t = X_t_all[valid]
        y_seq_t = y_seq_all[valid]
        risk_vals = risk_all[valid]
        ind_ids = ind_all[valid]

        # V7: 多周期加权标签（而非单周期future_len-1）
        target_h = getattr(config, 'target_horizon', 5)
        h_idx = min(target_h - 1, max_horizon - 1)
        y_t = y_seq_t[:, h_idx]           # 主标签用 target_horizon 日收益
        # y_seq_t 保留全部10个horizon用于多任务训练

        # 截面rank特征
        X_rank = np.argsort(np.argsort(X_t, axis=0), axis=0).astype(np.float32) / (X_t.shape[0] - 1)

        # 行业相对特征：仅对last聚合的核心因子计算，避免维度过高
        relative_indices = [FEATURE_COLS.index(name) for name in INDUSTRY_REL_FEATURES if name in FEATURE_COLS]
        n_relative = len(relative_indices)
        industry_relative = np.zeros((X_t.shape[0], n_relative), dtype=np.float32)
        if n_industries > 0:
            for j, feat_idx in enumerate(relative_indices):
                feat_vals = X_t[:, feat_idx].copy()
                for ind in range(n_industries):
                    mask_ind = (ind_ids == ind)
                    if mask_ind.sum() > 1:
                        feat_vals[mask_ind] -= np.mean(feat_vals[mask_ind])
                unknown_mask = (ind_ids == -1)
                if unknown_mask.sum() > 1:
                    feat_vals[unknown_mask] -= np.mean(feat_vals[unknown_mask])
                industry_relative[:, j] = feat_vals
        else:
            industry_relative = X_t[:, relative_indices]

        # 拼接: 聚合特征 + rank特征 + 行业相对特征
        X_t = np.concatenate([X_t, X_rank, industry_relative], axis=1)

        # MAD鏍囧噯鍖?
        median = np.median(X_t, axis=0, keepdims=True)
        mad = np.median(np.abs(X_t - median), axis=0, keepdims=True) + 1e-8
        X_norm = (X_t - median) / mad
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
        X_norm = np.clip(X_norm, -10.0, 10.0)

        # 风险因子标准化（仅前3个股票级风险因子做截面标准化，市场特征保持原值
        risk_cont_norm = risk_vals.copy()
        risk_mean = risk_vals[:, :3].mean(axis=0, keepdims=True)
        risk_std = risk_vals[:, :3].std(axis=0, keepdims=True) + 1e-8
        risk_cont_norm[:, :3] = (risk_vals[:, :3] - risk_mean) / risk_std
        risk_cont_norm = np.nan_to_num(risk_cont_norm, nan=0.0, posinf=0.0, neginf=0.0)
        risk_cont_norm[:, :3] = np.clip(risk_cont_norm[:, :3], -10.0, 10.0)

        # 行业one-hot
        if n_industries > 0:
            industry_onehot = np.zeros((len(ind_ids), n_industries), dtype=np.float32)
            valid_ind = ind_ids >= 0
            if valid_ind.any():
                industry_onehot[valid_ind, ind_ids[valid_ind]] = 1.0
            risk_factors = np.concatenate([risk_cont_norm, industry_onehot], axis=1)
        else:
            risk_factors = risk_cont_norm

        # 标签：稳健缩放
        p_low, p_high = np.percentile(y_t, [1, 99])
        y_clipped = np.clip(y_t, p_low, p_high)
        y_label = (y_clipped - np.mean(y_clipped)) / (np.std(y_clipped) + 1e-8)

        # 多期标签
        y_seq_norm = np.zeros_like(y_seq_t)
        for h in range(max_horizon):
            y_h = y_seq_t[:, h]
            p_l, p_h = np.percentile(y_h, [1, 99])
            y_h_c = np.clip(y_h, p_l, p_h)
            y_seq_norm[:, h] = (y_h_c - np.mean(y_h_c)) / (np.std(y_h_c) + 1e-8)

        X_samples.append({
            'date': all_dates[t],
            'X': X_norm,
            'y': y_label,
            'y_seq': y_seq_norm,
            'codes': all_codes_np[valid].tolist(),
            'raw_y': y_t,
            'risk': risk_factors,
            'industry_ids': ind_ids,
        })

    print(f"截面样本数 {len(X_samples)}")

    # 显式释放已不再需要的特大数组，节省约340 MB
    del feat_array, risk_raw_array, industry_array, ret_seq_array
    gc.collect()

    split = int(len(X_samples) * 0.8)
    train_samples = X_samples[:split]
    val_samples = X_samples[split:]

    if use_cache:
        print(f"保存缓存: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump((train_samples, val_samples), f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_samples, val_samples


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
