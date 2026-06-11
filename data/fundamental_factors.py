# fundamental_factors.py - 基本面因子获取及PIT对齐
import hashlib
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

from data.api_utils import SafeAPICaller, resolve_tushare_token

warnings.filterwarnings('ignore')

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_VERSION = "pit_v3_schema"
FACTOR_SCHEMA = ('roe', 'revenue_yoy', 'pe_percentile')
FACTOR_SCHEMA_VERSION = "schema_v1"

_ts_call = SafeAPICaller(
    min_interval=2.0,
    max_retries=3,
    retry_base_delay=5.0,
    jitter=(0.3, 0.6),
    data_source="tushare",
)


def _cache_path(codes, start_date, end_date):
    joined = '|'.join(sorted(map(str, codes)))
    digest = hashlib.md5(joined.encode('utf-8')).hexdigest()[:12]
    schema_digest = hashlib.md5('|'.join(FACTOR_SCHEMA).encode('utf-8')).hexdigest()[:8]
    filename = f"fundamental_features_{CACHE_VERSION}_{FACTOR_SCHEMA_VERSION}_{schema_digest}_{digest}_{start_date}_{end_date}.parquet"
    return os.path.join(CACHE_DIR, filename)


def _safe_ts_call(pro, func_name, *args, **kwargs):
    func = getattr(pro, func_name)
    return _ts_call(func, *args, **kwargs)


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

    def _batch_fetch(pro, api_name, fields, chunk_size=200, sleep_between=65):
        """批量拉取：每次200只股票，间隔65秒（基础用户1次/分钟限制）"""
        all_frames = []
        chunks = [codes[i:i+chunk_size] for i in range(0, len(codes), chunk_size)]
        n_chunks = len(chunks)
        for ci, chunk in enumerate(tqdm(chunks, desc=f"{api_name} (batch {chunk_size})")):
            code_str = ','.join(chunk)
            df = _safe_ts_call(
                pro, api_name, ts_code=code_str,
                start_date=start_date, end_date=end_date, fields=fields
            )
            if df is not None and not df.empty:
                all_frames.append(df)
            if ci < n_chunks - 1:
                time.sleep(sleep_between)
        return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    print("获取利润表数据（批量）...")
    income_all = _batch_fetch(pro, 'income',
        fields='ts_code,ann_date,end_date,revenue,n_income')

    print("获取资产负债表数据（批量）...")
    balance_all = _batch_fetch(pro, 'balancesheet',
        fields='ts_code,ann_date,end_date,total_hldr_eqy_exc_min_int')

    funda_frames = []
    if not income_all.empty and not balance_all.empty:

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


def merge_to_daily_akshare(funda_df, codes, all_dates):
    """
    PIT 对齐：akshare 下载的基本面数据合并到日频。
    funda_df columns: ts_code, effective_date, end_date, roe, revenue_yoy
    对每个交易日，取 effective_date <= 当日的最新 end_date 数据，前向填充。

    Returns: DataFrame index=all_dates, columns=[{code}_roe, {code}_revenue_yoy]
    """
    df = funda_df.copy()
    df = df.dropna(subset=['effective_date']).sort_values('effective_date')
    all_dates = pd.DatetimeIndex(all_dates)
    result = pd.DataFrame(index=all_dates, dtype=np.float32)

    cols = ['roe', 'revenue_yoy']

    for code in codes:
        pure = code.replace('.SH', '').replace('.SZ', '')
        code_data = df[df['ts_code'] == code].copy()

        if code_data.empty:
            for col in cols:
                result[f'{code}_{col}'] = 0.0
            continue

        code_data = code_data.sort_values(['effective_date', 'end_date'])
        code_data = code_data.drop_duplicates(subset=['effective_date'], keep='last')
        code_data = code_data.set_index('effective_date').sort_index()

        for col in cols:
            if col not in code_data.columns:
                result[f'{code}_{col}'] = 0.0
                continue
            series = code_data[col].dropna()
            if series.empty:
                result[f'{code}_{col}'] = 0.0
                continue
            daily = series.reindex(all_dates, method='ffill').fillna(0.0)
            result[f'{code}_{col}'] = daily.values

    return result


if __name__ == "__main__":
    print("=== 基本面因子测试===\n")
    test_codes = ['000001.SZ', '600519.SH', '600000.SH']
    df = fetch_fundamentals(test_codes)
    print(df.head())
