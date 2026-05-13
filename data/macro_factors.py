# macro_factors.py - 宏观与资金流因子获取（akshare免费数据源）
import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== 安全调用（参考TushareProLite限流逻辑===================
_last_request_time = 0
_min_interval = 1.5


def _safe_ak_call(func, *args, **kwargs):
    """带重试和限流的akshare调用"""
    global _last_request_time
    for attempt in range(3):
        try:
            elapsed = time.time() - _last_request_time
            if elapsed < _min_interval:
                time.sleep(_min_interval - elapsed)
            _last_request_time = time.time()
            time.sleep(random.uniform(0.2, 0.5))
            return func(*args, **kwargs)
        except Exception as e:
            print(f"  请求失败 ({attempt+1}/3): {e}, 等待 {4*(attempt+1)}s")
            time.sleep(4 * (attempt + 1))
    return None

# ==================== 北向资金净流入 ====================
def fetch_north_flow(save_path=None):
    """获取北向资金日频净流入数据"""
    import akshare as ak
    save_path = save_path or os.path.join(CACHE_DIR, "north_flow.csv")

    if os.path.exists(save_path):
        df = pd.read_csv(save_path, index_col=0, parse_dates=True)
        print(f"cache: loaded north flow {len(df)} rows")
        return df

    print("下载北向资金数据...")
    df = _safe_ak_call(ak.stock_hsgt_hist_em, symbol="hutong")
    df_sz = _safe_ak_call(ak.stock_hsgt_hist_em, symbol="shentong")

    if df is not None and not df.empty:
        # 列名：日期、当日成交净买额
        df['日期'] = pd.to_datetime(df['日期'])
        net = pd.to_numeric(df['当日成交净买额'], errors='coerce').fillna(0)
        result = pd.DataFrame({'net_flow': net.values}, index=df['日期'])

        if df_sz is not None and not df_sz.empty:
            df_sz['日期'] = pd.to_datetime(df_sz['日期'])
            net_sz = pd.to_numeric(df_sz['当日成交净买额'], errors='coerce').fillna(0)
            sz_series = pd.Series(net_sz.values, index=df_sz['日期'])
            result['net_flow'] = result['net_flow'].add(sz_series, fill_value=0)

        result = result.sort_index()
        rolling_mean = result['net_flow'].rolling(60, min_periods=20).mean().shift(1)
        rolling_std = result['net_flow'].rolling(60, min_periods=20).std().shift(1)
        result['north_net_zscore'] = (result['net_flow'] - rolling_mean) / (rolling_std + 1e-8)
        result = result[['north_net_zscore']]
        result.to_csv(save_path)
        print(f"north flow saved: {len(result)} rows")
        return result

    print("north flow download failed, generating placeholder data")
    empty = pd.DataFrame(columns=['north_net_zscore'])
    empty.to_csv(save_path)
    return empty


# ==================== 融资融券余额 ====================
def fetch_margin_balance(save_path=None):
    """获取沪深融资融券余额日频数据"""
    import akshare as ak
    save_path = save_path or os.path.join(CACHE_DIR, "margin_balance.csv")

    if os.path.exists(save_path):
        df = pd.read_csv(save_path, index_col=0, parse_dates=True)
        print("macro factor: PMI zscore range fixed")
        return df

    print("下载融资融券数据...")
    # 使用宏观接口：macro_china_market_margin_sh / sz
    df_sh = _safe_ak_call(ak.macro_china_market_margin_sh)
    df_sz = _safe_ak_call(ak.macro_china_market_margin_sz)

    frames = []
    for df, label in [(df_sh, 'sh'), (df_sz, 'sz')]:
        if df is not None and not df.empty and '日期' in df.columns and '融资融券余额' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            daily = df.groupby('日期')['融资融券余额'].last()
            frames.append(daily)

    if frames:
        combined = pd.DataFrame({'balance': pd.concat(frames).groupby(level=0).sum()})
        combined = combined.sort_index()
        combined['margin_balance_change'] = combined['balance'].pct_change(5).fillna(0)
        result = combined[['margin_balance_change']]
        result.to_csv(save_path)
        print(f"margin balance loaded: {len(result)} rows")
        return result

    print("融资融券下载失败，生成占位数据")
    empty = pd.DataFrame(columns=['margin_balance_change'])
    empty.to_csv(save_path)
    return empty

# ==================== PMI ====================
def fetch_pmi(save_path=None):
    """获取制造业PMI月频数据，按保守可获得日生效。"""
    import akshare as ak
    save_path = save_path or os.path.join(CACHE_DIR, "pmi_pit_v2.csv")

    if os.path.exists(save_path):
        df = pd.read_csv(save_path, index_col=0, parse_dates=True)
        print(f"从缓存加载PMI: {len(df)} 条")
        return df

    print("下载PMI数据...")
    df = _safe_ak_call(ak.macro_china_pmi)
    if df is not None and not df.empty:
        def parse_pmi_month(s):
            import re
            m = re.search(r"(\d{4})year(\d{2})month", str(s))
            if m:
                return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-01")
            return pd.NaT

        df['month_date'] = df['月份'].apply(parse_pmi_month)
        df['pmi'] = pd.to_numeric(df['制造业-指数'], errors='coerce')
        df = df.dropna(subset=['month_date', 'pmi']).sort_values('month_date')
        df['effective_date'] = df['month_date'] + pd.offsets.MonthBegin(1)
        expanding_mean = df['pmi'].expanding(min_periods=12).mean().shift(1)
        expanding_std = df['pmi'].expanding(min_periods=12).std().shift(1)
        df['pmi_zscore'] = (df['pmi'] - expanding_mean) / (expanding_std + 1e-8)
        result = pd.DataFrame({'pmi_zscore': df['pmi_zscore'].values}, index=df['effective_date'])
        result = result.replace([np.inf, -np.inf], np.nan).sort_index()
        result.to_csv(save_path)
        print(f"PMI已保存: {len(result)} 条")
        return result

    print("PMI下载失败，生成占位数据")
    empty = pd.DataFrame(columns=['pmi_zscore'])
    empty.to_csv(save_path)
    return empty


# ==================== 构建宏观特征 ====================
def build_macro_features(all_dates):
    """
    将宏观数据对齐到交易日历
    all_dates: list/pd.Index of trade dates
    返回: DataFrame index=dates, columns=[north_net_zscore, margin_balance_change, pmi_zscore]
    """
    north = fetch_north_flow()
    margin = fetch_margin_balance()
    pmi = fetch_pmi()

    all_dates = pd.DatetimeIndex(sorted(all_dates))
    macro = pd.DataFrame(index=all_dates)

    macro['north_net_zscore'] = 0.0
    macro['margin_balance_change'] = 0.0
    macro['pmi_zscore'] = 0.0

    # 对齐北向资金（日频）
    if not north.empty and 'north_net_zscore' in north.columns:
        macro['north_net_zscore'] = north['north_net_zscore'].reindex(all_dates).fillna(0)

    # 对齐融资融券（日频）
    if not margin.empty and 'margin_balance_change' in margin.columns:
        macro['margin_balance_change'] = margin['margin_balance_change'].reindex(all_dates).fillna(0)

    # 对齐PMI（月频 -> 前向填充到日频）
    if not pmi.empty and 'pmi_zscore' in pmi.columns:
        macro['pmi_zscore'] = pmi['pmi_zscore'].reindex(all_dates, method='ffill').fillna(0)

    return macro


if __name__ == "__main__":
    print("=== 宏观/资金流因子测试 ===\n")
    df = build_macro_features(pd.date_range('2020-01-01', '2025-12-31', freq='B'))
    print(f"\n宏观特征 shape: {df.shape}")
    print(df.head(10))
    print("\n统计摘要:")
    print(df.describe())
