# fundamental_factors.py - 基本面因子获取及PIT对齐
import hashlib
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
CACHE_VERSION = "pit_v3_schema"
FACTOR_SCHEMA = ('roe', 'revenue_yoy', 'pe_percentile')
FACTOR_SCHEMA_VERSION = "schema_v1"

_last_request_time = 0
_min_interval = 2.0


def resolve_tushare_token(token=None):
    resolved = token or os.getenv('TUSHARE_TOKEN')
    if not resolved:
        raise ValueError('缺少 TUSHARE_TOKEN，请设置环境变量或传入token')
    return resolved


def _cache_path(codes, start_date, end_date):
    joined = '|'.join(sorted(map(str, codes)))
    digest = hashlib.md5(joined.encode('utf-8')).hexdigest()[:12]
    schema_digest = hashlib.md5('|'.join(FACTOR_SCHEMA).encode('utf-8')).hexdigest()[:8]
    filename = f"fundamental_features_{CACHE_VERSION}_{FACTOR_SCHEMA_VERSION}_{schema_digest}_{digest}_{start_date}_{end_date}.parquet"
    return os.path.join(CACHE_DIR, filename)


def _safe_ts_call(pro, func_name, *args, **kwargs):
    global _last_request_time
    for attempt in range(3):
        try:
            elapsed = time.time() - _last_request_time
            if elapsed < _min_interval:
                time.sleep(_min_interval - elapsed)
            _last_request_time = time.time()
            time.sleep(random.uniform(0.3, 0.6))
            func = getattr(pro, func_name)
            return func(*args, **kwargs)
        except Exception as e:
            print(f"  请求失败 ({attempt+1}/3): {e}, 等待 {5*(attempt+1)}s")
            time.sleep(5 * (attempt + 1))
    return None


def _to_datetime(series):
    return pd.to_datetime(series, errors='coerce')


def _rolling_percentile(values, window=1250, min_periods=60):
    s = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)

    def pct_rank(x):
        cur = x[-1]
        hist = x[np.isfinite(x)]
        if not np.isfinite(cur) or len(hist) < min_periods:
            return np.nan
        return float(np.mean(hist <= cur))

    return s.rolling(window, min_periods=min_periods).apply(pct_rank, raw=True)


def fetch_fundamentals(codes, token=None, start_date='20100101', end_date='20261231'):
    """
    获取PIT基本面因子长表。
    Returns:
        DataFrame columns=[ts_code,effective_date,end_date,roe,revenue_yoy,pe_percentile]
    """
    cache_file = _cache_path(codes, start_date, end_date)
    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        print(f"从缓存加载PIT基本面因子 {df.shape}")
        return df

    token = resolve_tushare_token(token)
    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()

    print("获取利润表数据...")
    income_list = []
    for code in tqdm(codes, desc="income_table"):
        df = _safe_ts_call(
            pro, 'income', ts_code=code, start_date=start_date, end_date=end_date,
            fields='ts_code,ann_date,end_date,revenue,n_income'
        )
        if df is not None and not df.empty:
            income_list.append(df)

    print("获取资产负债表数据...")
    balance_list = []
    for code in tqdm(codes, desc="资产负债表"):
        df = _safe_ts_call(
            pro, 'balancesheet', ts_code=code, start_date=start_date, end_date=end_date,
            fields='ts_code,ann_date,end_date,total_hldr_eqy_exc_min_int'
        )
        if df is not None and not df.empty:
            balance_list.append(df)

    funda_frames = []
    if income_list and balance_list:
        income_all = pd.concat(income_list, ignore_index=True)
        balance_all = pd.concat(balance_list, ignore_index=True)

        for df in (income_all, balance_all):
            df['ann_date'] = _to_datetime(df['ann_date'])
            df['end_date'] = _to_datetime(df['end_date'])

        income_all = income_all.dropna(subset=['ts_code', 'ann_date', 'end_date'])
        balance_all = balance_all.dropna(subset=['ts_code', 'ann_date', 'end_date'])
        income_all = income_all.sort_values(['ts_code', 'end_date', 'ann_date'])
        balance_all = balance_all.sort_values(['ts_code', 'end_date', 'ann_date'])

        # 去重：每个(ts_code, end_date)保留最新ann_date，避免merge产生笛卡尔积
        income_dedup = income_all.groupby(['ts_code', 'end_date']).last().reset_index()
        balance_dedup = balance_all.groupby(['ts_code', 'end_date']).last().reset_index()

        income_dedup['revenue'] = pd.to_numeric(income_dedup['revenue'], errors='coerce')
        income_dedup['n_income'] = pd.to_numeric(income_dedup['n_income'], errors='coerce')
        income_dedup['revenue_yoy'] = income_dedup.groupby('ts_code')['revenue'].pct_change(4)

        merged = income_dedup.merge(
            balance_dedup[['ts_code', 'end_date', 'ann_date', 'total_hldr_eqy_exc_min_int']],
            on=['ts_code', 'end_date'], how='left', suffixes=('_income', '_balance')
        )
        merged['total_hldr_eqy_exc_min_int'] = pd.to_numeric(
            merged['total_hldr_eqy_exc_min_int'], errors='coerce'
        )
        ann_cols = merged[['ann_date_income', 'ann_date_balance']]
        merged['effective_date'] = ann_cols.max(axis=1)
        equity = merged['total_hldr_eqy_exc_min_int']
        merged['roe'] = np.where(equity > 0, merged['n_income'] / equity, np.nan)
        funda_frames.append(merged[[
            'ts_code', 'effective_date', 'end_date', 'roe', 'revenue_yoy'
        ]])

    print("获取估值数据...")
    daily_frames = []
    for code in tqdm(codes, desc="valuation_index"):
        df = _safe_ts_call(
            pro, 'daily_basic', ts_code=code, start_date=start_date, end_date=end_date,
            fields='ts_code,trade_date,pe,pe_ttm,pb'
        )
        if df is None or df.empty:
            continue
        df['trade_date'] = _to_datetime(df['trade_date'])
        df = df.dropna(subset=['trade_date']).sort_values('trade_date')
        pe = pd.to_numeric(df.get('pe_ttm'), errors='coerce')
        if pe.isna().all() and 'pe' in df.columns:
            pe = pd.to_numeric(df['pe'], errors='coerce')
        pe = pe.where(pe > 0)
        df['pe_percentile'] = _rolling_percentile(pe)
        daily_frames.append(pd.DataFrame({
            'ts_code': df['ts_code'],
            'effective_date': df['trade_date'],
            'end_date': pd.NaT,
            'pe_percentile': df['pe_percentile'],
        }))

    if daily_frames:
        funda_frames.append(pd.concat(daily_frames, ignore_index=True))

    if funda_frames:
        result = pd.concat(funda_frames, ignore_index=True, sort=False)
        result['effective_date'] = _to_datetime(result['effective_date'])
        result = result.dropna(subset=['ts_code', 'effective_date'])
        result = result.sort_values(['ts_code', 'effective_date'])
    else:
        result = pd.DataFrame(columns=[
            'ts_code', 'effective_date', 'end_date', 'roe', 'revenue_yoy', 'pe_percentile'
        ])

    result.to_parquet(cache_file)
    print(f"PIT基本面因子已保存: {result.shape}")
    return result


def merge_to_daily(funda_df, code_list, all_dates):
    """Merge fundamental data to daily frequency"""
    dates = pd.DatetimeIndex(sorted(all_dates))
    result = pd.DataFrame(index=dates)
    if funda_df is None or funda_df.empty:
        for code in code_list:
            result[f'{code}_roe'] = 0.0
            result[f'{code}_revenue_yoy'] = 0.0
            result[f'{code}_pe_percentile'] = 0.0
        return result

    df = funda_df.copy()
    df['effective_date'] = _to_datetime(df['effective_date'])
    df = df.dropna(subset=['ts_code', 'effective_date']).sort_values(['ts_code', 'effective_date'])

    for code in code_list:
        sub = df[df['ts_code'] == code]
        for col in ['roe', 'revenue_yoy', 'pe_percentile']:
            out_col = f'{code}_{col}'
            if sub.empty or col not in sub.columns:
                result[out_col] = 0.0
                continue
            series = sub[['effective_date', col]].dropna(subset=[col])
            if series.empty:
                result[out_col] = 0.0
                continue
            series = series.drop_duplicates('effective_date', keep='last').set_index('effective_date')[col]
            aligned = series.sort_index().reindex(dates, method='ffill')
            result[out_col] = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    return result


if __name__ == "__main__":
    print("=== 基本面因子测试===\n")
    test_codes = ['000001.SZ', '600519.SH', '600000.SH']
    df = fetch_fundamentals(test_codes)
    print(df.head())
