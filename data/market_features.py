# market_features.py - 市场整体属性计算（指数、宽度、离散度）
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 申万一级行业代码和名称
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

# 市场特征列名
MARKET_COLS = [
    # 沪深300
    'idx_ret_1d', 'idx_ret_5d', 'idx_ret_20d', 'idx_vol_20d',
    # 上证50
    'sz50_ret_1d', 'sz50_ret_5d', 'sz50_ret_20d', 'sz50_vol_20d',
    # 中证500
    'zz500_ret_1d', 'zz500_ret_5d', 'zz500_ret_20d', 'zz500_vol_20d',
    # 创业板指
    'cyb_ret_1d', 'cyb_ret_5d', 'cyb_ret_20d', 'cyb_vol_20d',
    # 市场宽度
    'advance_decline', 'new_high_ratio', 'return_dispersion',
]

# 添加行业指数收益与可用性mask列名
for code, name in SW_INDUSTRIES:
    MARKET_COLS.append(f'sw_{code}_ret')
for code, name in SW_INDUSTRIES:
    MARKET_COLS.append(f'sw_{code}_available')

N_MARKET = len(MARKET_COLS)


def _compute_index_features(data_dir):
    """从 hs300_index.csv 计算指数特征"""
    idx_path = os.path.join(data_dir, "hs300_index.csv")
    if not os.path.exists(idx_path):
        print(f"警告: 未找到指数文件 {idx_path}，指数特征将填0")
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
    从 (num_stocks, num_dates) 收盘价矩阵计算市场宽度特征。
    使用numpy向量化，避免Python逐日/逐股票循环。
    Args:
        close_matrix: (num_stocks, num_dates) float32, NaN表示缺失

    Returns:
        breadth: (num_dates, 3) float32 - advance_decline, new_high_ratio, return_dispersion
    """
    num_stocks, num_dates = close_matrix.shape
    breadth = np.zeros((num_dates, 3), dtype=np.float32)

    # 日收益矩阵
    ret_matrix = np.full_like(close_matrix, np.nan)
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

        # 新高比例: 当日收盘 >= 前20日最高价 * 0.995
        if t >= 20:
            col_close = close_matrix[:, t]
            valid_c = ~np.isnan(col_close)
            if valid_c.sum() >= 10:
                high_20 = np.nanmax(close_matrix[valid_c, t-19:t+1], axis=1)
                new_high = (col_close[valid_c] >= high_20 * 0.995).sum()
                breadth[t, 1] = new_high / max(valid_c.sum(), 1)

        # 截面收益离散度
        breadth[t, 2] = np.std(rets)

    return breadth


def _compute_index_features_full(idx_path, prefix):
    """
    从指数文件计算完整特征（1d/5d/20d收益 + 20d波动率）

    Args:
        idx_path: 指数文件路径
        prefix: 列名前缀（如 'idx', 'sz50', 'zz500', 'cyb'）
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
        print(f"警告: 读取{idx_path}失败: {e}")
        return None


def build_market_features_index_only(data_dir, all_dates):
    """
    从指数文件构建指数特征（不包含宽度特征）。
    Returns:
        DataFrame indexed by date with 16 + 31 + 31 = 78 index columns
    """
    all_dates = pd.DatetimeIndex(sorted(all_dates))

    # 初始化结果DataFrame - 宽基指数
    index_cols = []
    for prefix in ['idx', 'sz50', 'zz500', 'cyb']:
        for suffix in ['ret_1d', 'ret_5d', 'ret_20d', 'vol_20d']:
            index_cols.append(f'{prefix}_{suffix}')

    # 添加行业指数收益与可用性mask列名
    for code, name in SW_INDUSTRIES:
        index_cols.append(f'sw_{code}_ret')
    for code, name in SW_INDUSTRIES:
        index_cols.append(f'sw_{code}_available')

    result = pd.DataFrame(index=all_dates, columns=index_cols, data=0.0)

    # 读取宽基指数数据
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
        print(f"警告: 缺少宽基指数文件，将以0填充: {missing_index_files}")

    # 读取行业指数数据
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
                    print(f"警告: 读取行业{name}失败: {e}")
            else:
                missing_sw.append(code)
    else:
        missing_sw = [code for code, _ in SW_INDUSTRIES]
    if missing_sw:
        print(f"警告: 缺少{len(missing_sw)}个申万行业指数文件，将以0填充")

    return result


if __name__ == "__main__":
    print("=== 市场整体属性测试===\n")
    data_dir = "data/raw"
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='B')
    result = build_market_features_index_only(data_dir, dates)
    print(f"指数特征 shape: {result.shape}")
    print(result.head(10))
    print("\n统计摘要:")
    print(result.describe())
