# data_pipeline.py 鈥?鎴?潰澶氬洜瀛愭暟鎹??绾匡紙鍐呭瓨浼樺寲鐗堬級
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

# 瀹忚?鐗瑰緛鍒楀悕
MACRO_COLS = ['north_net_zscore', 'margin_balance_change', 'pmi_zscore']
# 鍩烘湰闈㈢壒寰佸垪鍚?
FUNDAMENTAL_COLS = ['roe', 'revenue_yoy', 'pe_percentile']
# 甯傚満鏁翠綋灞炴EUR?
from data.market_features import (
    build_market_features_index_only,
    compute_breadth_from_close_matrix,
    MARKET_COLS, N_MARKET,
)

# V7 鑱氬悎鏂瑰紡锛堟ā鍧楃骇甯搁噺锛屼緵 train.py 寮曠敤锛?
AGG_NAMES = ['last', 'sma5', 'sma20', 'vol5', 'vol20']
N_AGGS = len(AGG_NAMES)
INDUSTRY_REL_FEATURES = ['ret_5d', 'ret_20d', 'vol_10d', 'vol_60d', 'price_momentum', 'log_volume']
TECH_FEATURES = [
    'sma5_gap', 'sma10_gap', 'sma20_gap', 'ema12_gap', 'ema26_gap',
    'rsi_norm', 'macd_pct', 'macd_signal_pct', 'macd_diff_pct',
    'atr_pct', 'volume_ratio'
]
CACHE_VERSION = "v11_clipped_features"


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
    """鍔犺浇骞跺悎骞朵釜鑲℃墿灞曞洜瀛愶紙鍩烘湰闈㈢瓑锛夈EUR?""
    extra_feat_cols = []

    # ----- 鍩烘湰闈㈠洜瀛愶紙瀛ｉ?鍓嶅悜濉?厖鍒版棩棰戯級-----
    if getattr(config, 'use_fundamental_features', False):
        try:
            from data.fundamental_factors import fetch_fundamentals, merge_to_daily
            codes = list(df_dict.keys())
            # 娉ㄦ剰锛氶渶瑕佺敤鎴稵oken锛屾湭璁剧疆鏃惰烦杩?
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
                    print(f"宸插姞杞藉熀鏈?潰鐗瑰緛: {[f'fund_{c}' for c in FUNDAMENTAL_COLS]}")
            else:
                print("鎻愮ず: 鏈??缃?tushare_token锛岃烦杩囧熀鏈?潰鍥犲瓙鍔犺浇")
        except Exception as e:
            print(f"鍩烘湰闈㈢壒寰佸姞杞藉け璐? {e}")

    return extra_feat_cols


def build_cross_section_dataset(config, stock_universe=None, use_cache=True):
    """
    """Build cross-section dataset, return (train_samples, val_samples).

    鐩歌緝v6鐗堟湰鐨勫唴瀛樹紭鍖栵細
    - 涓嶉?瀛樺叏灞EUR鐗瑰緛鐭╅樀 (num_stocks 脳 num_dates 脳 feat_dim)
    - 鏀圭敤绐楀彛绱㈠紩棰勮?绠?+ 鎸夐渶鍒囩墖
    - 淇濈暀浜嗘埅闈?ank鐗瑰緛鍜岃?涓氱浉瀵圭壒寰?
    """
    data_dir = config.data_dir
    seq_len = config.seq_len
    future_len = getattr(config, 'future_len', 5)
    max_horizon = getattr(config, 'max_horizon', 10)

    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    test_n = getattr(config, 'test_stocks', None) if getattr(config, 'test_mode', False) else None

    # 鍙??鐨勭紦瀛樻枃浠跺悕
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
    cache_key = f"cross_section_{CACHE_VERSION}_{n_stocks}stocks{universe_tag}_{feat_str}_seq{seq_len}_h{config.target_horizon}"
    cache_file = os.path.join(cache_dir, cache_key + ".pkl")
    if use_cache and os.path.exists(cache_file) and not config.force_rebuild:
        print(f"鍔犺浇缂撳瓨: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # ========== 1. 璇诲彇鑲＄エ鏁版嵁 ==========
    print("璇诲彇鑲＄エ鏁版嵁...")
    excluded = {'all_data_jq.csv', 'stable_stocks.csv', 'stable_stocks_industry.csv'}
    csv_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and f not in excluded and f[0].isdigit()
    )
    if config.max_stocks:
        csv_files = csv_files[:config.max_stocks]
    if getattr(config, 'test_mode', False):
        test_n = getattr(config, 'test_stocks', 1000)
        print(f"test mode: loading only {len(csv_files)} stocks")
        print(f"test mode: loading only {len(csv_files)} stocks")

    df_dict = {}
    for fname in tqdm(csv_files, desc="鍔犺浇CSV", mininterval=10):
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
        raise ValueError("娌℃湁鏈夋晥鑲＄エ鏁版嵁")
    print(f"鏈夋晥鑲＄エ鏁? {len(df_dict)}")

    # ========== 2. 鏋勯EUR犲?灏哄害鐗瑰緛 ==========
    print("鏋勯EUR犲?灏哄害鐗瑰緛...")
    # V8: 鍑忓皯鍐椾綑锛屽?鍔犲井瑙傜粨鏋勭壒寰?
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

        # 鏀剁泭鐜囧拰娉㈠姩鐜?
        daily_ret = close_safe.pct_change().replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
        df['ret_5d'] = close_safe.pct_change(5).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        df['ret_20d'] = close_safe.pct_change(20).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        df['vol_10d'] = daily_ret.rolling(10).std().clip(0, 1.0)
        df['vol_60d'] = daily_ret.rolling(60).std().clip(0, 1.0)
        df['price_momentum'] = (close_safe / close_safe.rolling(20).mean() - 1).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        df['volume_spike'] = df['log_volume'].pct_change(1).abs().replace([np.inf, -np.inf], np.nan).clip(0, 10)

        # 寰??缁撴瀯鐗瑰緛
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

    FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))  # 鍘婚噸淇濆簭
    base_feat_dim = len(FEATURE_COLS)
    agg_feat_dim = base_feat_dim * N_AGGS
    print(f"鍩虹?鐗瑰緛鏁? {base_feat_dim}, 鑱氬悎鏂瑰紡: {N_AGGS}绉? 鑱氬悎鐗瑰緛鏁? {agg_feat_dim}")

    # ========== 3. 鍔犺浇鎵╁睍鍥犲瓙 ==========
    all_dates = sorted(set().union(*[df.index for df in df_dict.values()]))
    extra_feat_cols = _load_extra_features(config, df_dict, all_dates)
    FEATURE_COLS = FEATURE_COLS + extra_feat_cols
    base_feat_dim = len(FEATURE_COLS)
    agg_feat_dim = base_feat_dim * N_AGGS
    print(f"鏈EUR缁堢壒寰佸垪鏁? {base_feat_dim}, 鑱氬悎鐗瑰緛鏁? {agg_feat_dim}")

    # ========== 4. 鍏ㄥ眬鏃ユ湡骞堕泦 ==========
    num_dates = len(all_dates)
    print(f"鍏ㄥ眬鏃ユ湡鏁? {num_dates}")

    # ========== 5. 琛屼笟鏁版嵁 ==========
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
    print(f"琛屼笟鏁? {n_industries}")

    # ========== 6. 濉?厖鐗瑰緛鐭╅樀 ==========
    # 鏋勫缓 (num_stocks, num_dates, agg_feat_dim) 鐭╅樀
    # 杩欐槸鍞?竴鐨勫ぇ鐭╅樀锛屼絾蹇呴』瀛樺湪浠ヤ緵鍚庣画鎴?潰rank绛夋搷浣?
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

    print("濉?厖鐗瑰緛鐭╅樀...")
    for code, df in tqdm(df_dict.items(), desc="濉?厖鏁扮粍", mininterval=10):
        sidx = code_to_idx[code]
        stock_dates = df.index
        stock_idx = np.array([date_to_idx[d] for d in stock_dates], dtype=np.int32)
        T_stock = len(stock_dates)

        # V7鐗瑰緛鑱氬悎锛?绉嶈仛鍚堟浛浠ｅ師3绉峫ast/trend/vol锛?
        raw_feat = df.reindex(columns=FEATURE_COLS).values
        if T_stock >= seq_len:
            windows = sliding_window_view(raw_feat, seq_len, axis=0)
            if windows.shape[1] != seq_len:
                windows = windows.transpose(0, 2, 1)
            n_windows = windows.shape[0]

            # 5绉嶅?灏哄害鑱氬悎
            last_val = windows[:, -1, :]                          # 鏈EUR鏂板EUR?
            sma5 = windows[:, -5:, :].mean(axis=1) if seq_len >= 5 else last_val
            sma20 = windows[:, -20:, :].mean(axis=1) if seq_len >= 20 else sma5
            vol5 = windows[:, -5:, :].std(axis=1) if seq_len >= 5 else np.zeros_like(last_val)
            vol20 = windows[:, -20:, :].std(axis=1) if seq_len >= 20 else vol5
            agg_feat = np.concatenate([last_val, sma5, sma20, vol5, vol20], axis=1)

            agg_dates_idx = stock_idx[seq_len - 1: seq_len - 1 + n_windows]
            feat_array[sidx, agg_dates_idx, :] = agg_feat

        # 杩炵画椋庨櫓鍥犲瓙
        size_vals = df['log_volume'].values.astype(np.float32)
        vol_vals = df['vol_60d'].fillna(0).values.astype(np.float32)
        mom_vals = df['ret_20d'].fillna(0).values.astype(np.float32)
        risk_raw_array[sidx, stock_idx, :3] = np.column_stack([size_vals, vol_vals, mom_vals])

        # 琛屼笟ID
        raw_ind = industry_dict.get(code)
        ind_id = industry_to_idx.get(raw_ind, -1) if raw_ind else -1
        industry_array[sidx, stock_idx] = ind_id

        # 澶氭湡鏀剁泭鐜?
        close_vals = df['close'].values
        if T_stock >= max_horizon + 1:
            price_windows = sliding_window_view(close_vals, max_horizon + 1, axis=0)
            n_ret = price_windows.shape[0]
            ret_dates_idx = stock_idx[:n_ret]
            for h in range(1, max_horizon + 1):
                ret = (price_windows[:, h] - price_windows[:, 0]) / price_windows[:, 0]
                ret_seq_array[sidx, ret_dates_idx, h - 1] = ret

    # 濉?厖甯傚満鏁翠綋灞炴EURэ紙鍚戦噺鍖栬?绠楋紝閬垮厤O(N*D)寰?幆锛?
    if getattr(config, 'use_market_features', True):
        print("璁＄畻甯傚満鏁翠綋灞炴EUR?..")
        # 1. 鏋勫缓鏀剁洏浠风煩闃?(num_stocks, num_dates) 鐢ㄤ簬鍚戦噺鍖栬?绠楀?搴︾壒寰?
        close_matrix = np.full((num_stocks, num_dates), np.nan, dtype=np.float32)
        for code, df in df_dict.items():
            sidx = code_to_idx[code]
            stock_dates = df.index
            stock_idx = np.array([date_to_idx[d] for d in stock_dates], dtype=np.int32)
            close_matrix[sidx, stock_idx] = df['close'].values.astype(np.float32)

        # 2. 鍚戦噺鍖栬?绠楀?搴︾壒寰?(advance_decline, new_high_ratio, return_dispersion)
        breadth = compute_breadth_from_close_matrix(close_matrix)  # (num_dates, 3)
        del close_matrix  # 閲婃斁鍐呭瓨

        # 3. 鎸囨暟鐗瑰緛
        idx_feat = build_market_features_index_only(config.data_dir, all_dates)

        # 4. 鍚堝苟骞跺～鍏呭競鍦虹姸鎬佺壒寰?
        # MARKET_COLS椤哄簭: 16涓??鍩烘寚鏁?+ 3涓??搴︾壒寰?+ 31涓??涓氭敹鐩?+ 31涓??涓氬彲鐢ㄦEUR?ask
        # idx_feat鍒楅『搴? 16涓??鍩烘寚鏁?+ 31涓??涓氭敹鐩?+ 31涓??涓氬彲鐢ㄦEUR?ask
        idx_feat_cols = list(idx_feat.columns)
        for t_idx, date in enumerate(all_dates):
            if date in idx_feat.index:
                row = idx_feat.loc[date].values
                # 鍓?6鍒? 瀹藉熀鎸囨暟鐗瑰緛
                risk_raw_array[:, t_idx, 3:19] = row[:16]
                # 涓?棿3鍒? 瀹藉害鐗瑰緛
                risk_raw_array[:, t_idx, 19:22] = breadth[t_idx]
                # 鍚?2鍒? 琛屼笟鎸囨暟鏀剁泭 + 鍙?敤鎬?ask
                risk_raw_array[:, t_idx, 22:3 + N_MARKET] = row[16:]
        del breadth
        print(f"宸插姞杞藉競鍦烘暣浣撳睘鎬? {MARKET_COLS}")

    if getattr(config, 'use_macro_features', False):
        print("鍔犺浇瀹忚?/璧勯噾娴佺壒寰佸埌甯傚満鐘舵EUR?..")
        try:
            from data.macro_factors import build_macro_features
            macro_df = build_macro_features(all_dates)
            macro_start = 3 + N_MARKET
            for j, col in enumerate(MACRO_COLS):
                if col in macro_df.columns:
                    vals = macro_df[col].reindex(all_dates).fillna(0).values.astype(np.float32)
                    risk_raw_array[:, :, macro_start + j] = vals[None, :]
            print(f"宸插姞杞藉畯瑙?璧勯噾娴佺壒寰? {MACRO_COLS}")
        except Exception as e:
            print(f"瀹忚?/璧勯噾娴佺壒寰佸姞杞藉け璐ワ紝浣跨敤0濉?厖: {e}")

    # 閲婃斁涓嶅啀闇EUR瑕佺殑澶у?璞★紝涓烘埅闈㈡瀯寤鸿吘鍑哄唴瀛?
    import gc
    del df_dict
    gc.collect()

    # ========== 7. 鏋勫缓鎴?潰鏍锋湰 ==========
    print("鏋勫缓鎴?潰鏍锋湰...")
    X_samples = []
    min_stocks = getattr(config, 'min_stocks_per_time', 30)
    all_codes_np = np.array(all_codes)

    for t in tqdm(range(seq_len, num_dates - max_horizon), desc="鏋勫缓鎴?潰", mininterval=10):
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

        # V7: 澶氬懆鏈熷姞鏉冩爣绛撅紙鑰岄潪鍗曞懆鏈?future_len-1锛?
        target_h = getattr(config, 'target_horizon', 5)
        h_idx = min(target_h - 1, max_horizon - 1)
        y_t = y_seq_t[:, h_idx]           # 涓绘爣绛剧敤 target_horizon 鏃ユ敹鐩?
        # y_seq_t 淇濈暀鍏ㄩ儴10涓猦orizon鐢ㄤ簬澶氫换鍔¤?缁?

        # 鎴?潰rank鐗瑰緛
        X_rank = np.argsort(np.argsort(X_t, axis=0), axis=0).astype(np.float32) / (X_t.shape[0] - 1)

        # 琛屼笟鐩稿?鐗瑰緛锛氫粎瀵筶ast鑱氬悎鐨勬牳蹇冨洜瀛愯?绠楋紝閬垮厤缁村害杩囬珮
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

        # 鎷兼帴: 鑱氬悎鐗瑰緛 + rank鐗瑰緛 + 琛屼笟鐩稿?鐗瑰緛
        X_t = np.concatenate([X_t, X_rank, industry_relative], axis=1)

        # MAD鏍囧噯鍖?
        median = np.median(X_t, axis=0, keepdims=True)
        mad = np.median(np.abs(X_t - median), axis=0, keepdims=True) + 1e-8
        X_norm = (X_t - median) / mad
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
        X_norm = np.clip(X_norm, -10.0, 10.0)

        # 椋庨櫓鍥犲瓙鏍囧噯鍖栵紙浠呭墠3涓?釜鑲＄骇椋庨櫓鍥犲瓙鍋氭埅闈㈡爣鍑嗗寲锛屽競鍦虹壒寰佷繚鎸佸師鍊硷級
        risk_cont_norm = risk_vals.copy()
        risk_mean = risk_vals[:, :3].mean(axis=0, keepdims=True)
        risk_std = risk_vals[:, :3].std(axis=0, keepdims=True) + 1e-8
        risk_cont_norm[:, :3] = (risk_vals[:, :3] - risk_mean) / risk_std
        risk_cont_norm = np.nan_to_num(risk_cont_norm, nan=0.0, posinf=0.0, neginf=0.0)
        risk_cont_norm[:, :3] = np.clip(risk_cont_norm[:, :3], -10.0, 10.0)

        # 琛屼笟one-hot
        if n_industries > 0:
            industry_onehot = np.zeros((len(ind_ids), n_industries), dtype=np.float32)
            valid_ind = ind_ids >= 0
            if valid_ind.any():
                industry_onehot[valid_ind, ind_ids[valid_ind]] = 1.0
            risk_factors = np.concatenate([risk_cont_norm, industry_onehot], axis=1)
        else:
            risk_factors = risk_cont_norm

        # 鏍囩?锛氱ǔ鍋ョ缉鏀?
        p_low, p_high = np.percentile(y_t, [1, 99])
        y_clipped = np.clip(y_t, p_low, p_high)
        y_label = (y_clipped - np.mean(y_clipped)) / (np.std(y_clipped) + 1e-8)

        # 澶氭湡鏍囩?
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

    print(f"鎴?潰鏍锋湰鏁? {len(X_samples)}")

    # 鏄惧紡閲婃斁宸蹭笉鍐嶉渶瑕佺殑鐗瑰ぇ鏁扮粍锛岃妭鐪?~340 MB
    del feat_array, risk_raw_array, industry_array, ret_seq_array
    gc.collect()

    split = int(len(X_samples) * 0.8)
    train_samples = X_samples[:split]
    val_samples = X_samples[split:]

    if use_cache:
        print(f"淇濆瓨缂撳瓨: {cache_file}")
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
