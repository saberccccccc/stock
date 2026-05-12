# market_features.py 鈥?甯傚満鏁翠綋灞炴€ц?绠楋紙鎸囨暟銆佸?搴︺€佺?鏁ｅ害锛?import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 鐢充竾涓€绾ц?涓氫唬鐮佸拰鍚嶇О
SW_INDUSTRIES = [
    ('801010', 'agriculture'), ('801030', 'chemical'), ('801040', 'steel'),
    ('801050', 'nonferrous'), ('801080', 'electronics'), ('801110', 'home_appliances'),
    ('801120', 'food_beverage'), ('801130', 'textile'), ('801140', 'light_manufacturing'),
    ('801150', 'pharma'), ('801160', 'utilities'), ('801170', 'transport'),
    ('801180', 'realestate'), ('801200', 'retail'), ('801210', 'social_service'),
    ('801230', 'comprehensive'), ('801710', 'building_materials'), ('801720', 'building_decoration'),
    ('801730', 'power_equipment'), ('801740', 'defense'), ('801750', 'computer'),
    ('801760', 'media'), ('801770', 'telecom'), ('801780', 'banking'),
    ('801790', 'nonbank_finance'), ('801880', 'auto'), ('801890', 'machinery'),
    ('801950', 'coal'), ('801960', 'petrochemical'), ('801970', 'environmental'),
    ('801980', 'beauty_care'),
]

# 甯傚満鐗瑰緛鍒楀悕
MARKET_COLS = [
    # 娌?繁300
    'idx_ret_1d', 'idx_ret_5d', 'idx_ret_20d', 'idx_vol_20d',
    # 涓婅瘉50
    'sz50_ret_1d', 'sz50_ret_5d', 'sz50_ret_20d', 'sz50_vol_20d',
    # 涓?瘉500
    'zz500_ret_1d', 'zz500_ret_5d', 'zz500_ret_20d', 'zz500_vol_20d',
    # 鍒涗笟鏉挎寚
    'cyb_ret_1d', 'cyb_ret_5d', 'cyb_ret_20d', 'cyb_vol_20d',
    # 甯傚満瀹藉害
    'advance_decline', 'new_high_ratio', 'return_dispersion',
]

# 娣诲姞琛屼笟鎸囨暟鏀剁泭涓庡彲鐢ㄦ€?ask鍒楀悕
for code, name in SW_INDUSTRIES:
    MARKET_COLS.append(f'sw_{code}_ret')
for code, name in SW_INDUSTRIES:
    MARKET_COLS.append(f'sw_{code}_available')

N_MARKET = len(MARKET_COLS)


def _compute_index_features(data_dir):
    """浠?hs300_index.csv 璁＄畻鎸囨暟鐗瑰緛"""
    idx_path = os.path.join(data_dir, "hs300_index.csv")
    if not os.path.exists(idx_path):
        print(f"璀﹀憡: 鏈?壘鍒版寚鏁版枃浠?{idx_path}锛屾寚鏁扮壒寰佸皢濉?")
        return None

    df = pd.read_csv(idx_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    result = pd.DataFrame(index=df.index)
    result['idx_ret_1d'] = df['close'].pct_change(1)
    result['idx_ret_5d'] = df['close'].pct_change(5)
    result['idx_ret_20d'] = df['close'].pct_change(20)
    result['idx_vol_20d'] = df['close'].pct_change().rolling(20).std()
    return result.fillna(0)


def compute_breadth_from_close_matrix(close_matrix):
    """
    浠?(num_stocks, num_dates) 鏀剁洏浠风煩闃佃?绠楀競鍦哄?搴︾壒寰併€?    浣跨敤numpy鍚戦噺鍖栵紝閬垮厤Python閫愭棩鏈?閫愯偂绁ㄥ惊鐜?€?
    Args:
        close_matrix: (num_stocks, num_dates) float32, NaN琛ㄧず缂哄け

    Returns:
        breadth: (num_dates, 3) float32 鈥?advance_decline, new_high_ratio, return_dispersion
    """
    num_stocks, num_dates = close_matrix.shape
    breadth = np.zeros((num_dates, 3), dtype=np.float32)

    # 鏃ユ敹鐩婄煩闃?    ret_matrix = np.full_like(close_matrix, np.nan)
    ret_matrix[:, 1:] = (close_matrix[:, 1:] / close_matrix[:, :-1]) - 1

    for t in range(num_dates):
        col_ret = ret_matrix[:, t]
        valid = ~np.isnan(col_ret)
        n = valid.sum()
        if n < 10:
            continue

        rets = col_ret[valid]
        up = (rets > 0).sum()
        down = (rets < 0).sum()
        ad_ratio = (up + 1) / (down + 1)
        breadth[t, 0] = np.clip(np.log(ad_ratio), -2, 2)

        # 鏂伴珮姣斾緥: 褰撴棩鏀剁洏 >= 鍓?0鏃ユ渶楂?0.995
        if t >= 20:
            col_close = close_matrix[:, t]
            valid_c = ~np.isnan(col_close)
            if valid_c.sum() >= 10:
                high_20 = np.nanmax(close_matrix[valid_c, t-19:t+1], axis=1)
                new_high = (col_close[valid_c] >= high_20 * 0.995).sum()
                breadth[t, 1] = new_high / max(valid_c.sum(), 1)

        # 鎴?潰鏀剁泭绂绘暎搴?        breadth[t, 2] = np.std(rets)

    return breadth


def _compute_index_features_full(idx_path, prefix):
    """
    浠庢寚鏁版枃浠惰?绠楀畬鏁寸壒寰侊紙1d/5d/20d鏀剁泭 + 20d娉㈠姩鐜囷級

    Args:
        idx_path: 鎸囨暟鏂囦欢璺?緞
        prefix: 鍒楀悕鍓嶇紑锛堝? 'idx', 'sz50', 'zz500', 'cyb'锛?
    Returns:
        DataFrame with 4 columns: {prefix}_ret_1d/5d/20d, {prefix}_vol_20d
    """
    if not os.path.exists(idx_path):
        return None

    try:
        df = pd.read_csv(idx_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

        result = pd.DataFrame(index=df.index)
        result[f'{prefix}_ret_1d'] = df['close'].pct_change(1)
        result[f'{prefix}_ret_5d'] = df['close'].pct_change(5)
        result[f'{prefix}_ret_20d'] = df['close'].pct_change(20)
        result[f'{prefix}_vol_20d'] = df['close'].pct_change().rolling(20).std()
        return result.fillna(0)
    except Exception as e:
        print(f"璀﹀憡: 璇诲彇{idx_path}澶辫触: {e}")
        return None


def build_market_features_index_only(data_dir, all_dates):
    """
    浠庢寚鏁版枃浠舵瀯寤烘寚鏁扮壒寰侊紙涓嶅寘鍚??搴︾壒寰侊級銆?
    Returns:
        DataFrame indexed by date with 16 + 31 + 31 = 78 index columns
    """
    all_dates = pd.DatetimeIndex(sorted(all_dates))

    # 鍒濆?鍖栫粨鏋淒ataFrame - 瀹藉熀鎸囨暟
    index_cols = []
    for prefix in ['idx', 'sz50', 'zz500', 'cyb']:
        for suffix in ['ret_1d', 'ret_5d', 'ret_20d', 'vol_20d']:
            index_cols.append(f'{prefix}_{suffix}')

    # 娣诲姞琛屼笟鎸囨暟鏀剁泭涓庡彲鐢ㄦ€?ask鍒楀悕
    for code, name in SW_INDUSTRIES:
        index_cols.append(f'sw_{code}_ret')
    for code, name in SW_INDUSTRIES:
        index_cols.append(f'sw_{code}_available')

    result = pd.DataFrame(index=all_dates, columns=index_cols, data=0.0)

    # 璇诲彇瀹藉熀鎸囨暟鏁版嵁
    indices = [
        ('hs300_index.csv', 'idx'),
        ('sz50_index.csv', 'sz50'),
        ('zz500_index.csv', 'zz500'),
        ('cyb_index.csv', 'cyb'),
    ]

    missing_index_files = []
    for filename, prefix in indices:
        idx_path = os.path.join(data_dir, filename)
        idx_feat = _compute_index_features_full(idx_path, prefix)
        if idx_feat is not None:
            for col in idx_feat.columns:
                result[col] = idx_feat[col].reindex(all_dates).fillna(0).values
        else:
            missing_index_files.append(filename)
    if missing_index_files:
        print(f"璀﹀憡: 缂哄皯瀹藉熀鎸囨暟鏂囦欢锛屽皢浠?濉?厖: {missing_index_files}")

    # 璇诲彇琛屼笟鎸囨暟鏁版嵁
    sw_dir = os.path.join(data_dir, 'sw_industry')
    missing_sw = []
    if os.path.exists(sw_dir):
        for code, name in SW_INDUSTRIES:
            sw_file = os.path.join(sw_dir, f'{code}_{name}.csv')
            if os.path.exists(sw_file):
                try:
                    df = pd.read_csv(sw_file)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').set_index('date')
                    ret_1d = df['close'].pct_change(1).fillna(0)
                    ret_col = f'sw_{code}_ret'
                    avail_col = f'sw_{code}_available'
                    result[ret_col] = ret_1d.reindex(all_dates).fillna(0).values
                    result[avail_col] = df['close'].notna().astype(np.float32).reindex(all_dates).fillna(0).values
                except Exception as e:
                    print(f"璀﹀憡: 璇诲彇琛屼笟{name}澶辫触: {e}")
            else:
                missing_sw.append(code)
    else:
        missing_sw = [code for code, _ in SW_INDUSTRIES]
    if missing_sw:
        print(f"璀﹀憡: 缂哄皯{len(missing_sw)}涓?敵涓囪?涓氭寚鏁版枃浠讹紝灏嗕互0濉?厖")

    return result


if __name__ == "__main__":
    print("=== 甯傚満鏁翠綋灞炴€ф祴璇?===\n")
    data_dir = "data/raw"
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='B')
    result = build_market_features_index_only(data_dir, dates)
    print(f"鎸囨暟鐗瑰緛 shape: {result.shape}")
    print(result.head(10))
    print("\n缁熻?鎽樿?:")
    print(result.describe())
