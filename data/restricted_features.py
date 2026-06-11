# restricted_features.py - 限售解禁PIT特征（未来解禁压力）
import io
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

CACHE_FILE = "cache/restricted_features.parquet"
RESTRICTED_COLS = ['restricted_next_inv_days', 'restricted_next_ratio', 'restricted_mv_ratio_90d']
DEFAULT_NOTICE_LAG_DAYS = 30


def _ensure_effective_date(df):
    df = df.copy()
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    if 'effective_date' not in df.columns:
        df['effective_date'] = df['release_date'] - pd.Timedelta(days=DEFAULT_NOTICE_LAG_DAYS)
    else:
        df['effective_date'] = pd.to_datetime(df['effective_date'], errors='coerce')
        missing = df['effective_date'].isna()
        df.loc[missing, 'effective_date'] = df.loc[missing, 'release_date'] - pd.Timedelta(days=DEFAULT_NOTICE_LAG_DAYS)
    return df


def _quiet_fetch(func, *args, **kwargs):
    f_out = io.StringIO()
    f_err = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
        return func(*args, **kwargs)


def download_all(force=False):
    """
    下载全部限售解禁历史数据（2010-01-01 ~ 当前+1年），存为 parquet。
    已存在时跳过（除非 force=True）。

    Returns: DataFrame with columns [code, release_date, actual_shares, actual_mv, float_ratio]
    """
    cache_path = Path(CACHE_FILE)
    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
        df = _ensure_effective_date(cached)
        if 'effective_date' not in cached.columns:
            df.to_parquet(cache_path, index=False)
        print(f"从缓存加载限售解禁: {len(df)} rows, {df['code'].nunique()} stocks")
        return df

    import akshare as ak

    # 分批下载：每年一批，避免单次请求超时
    end_date = (datetime.now() + timedelta(days=365)).strftime('%Y%m%d')
    all_frames = []
    for year in range(2010, 2027):
        start = f'{year}0101'
        end = f'{year}1231'
        if year == 2026:
            end = end_date

        for attempt in range(3):
            try:
                raw = _quiet_fetch(ak.stock_restricted_release_detail_em,
                                   start_date=start, end_date=end)
                break
            except Exception:
                if attempt < 2:
                    time.sleep(3)
                else:
                    raw = None

        if raw is None or raw.empty:
            print(f"  {year}: 无数据")
            continue

        # 位置映射：[1]股票代码 [3]解禁时间 [6]实际解禁数量 [7]实际解禁市值 [8]占流通市值比例
        cols = raw.columns.tolist()
        if len(cols) < 9:
            print(f"  {year}: 列数不足 ({len(cols)})，跳过")
            continue

        raw = raw.rename(columns={
            cols[1]: 'code',
            cols[3]: 'release_date',
            cols[6]: 'actual_shares',
            cols[7]: 'actual_mv',
            cols[8]: 'float_ratio',
        })
        raw = raw[['code', 'release_date', 'actual_shares', 'actual_mv', 'float_ratio']].copy()
        raw['code'] = raw['code'].astype(str).str.zfill(6)
        raw['release_date'] = pd.to_datetime(raw['release_date'], errors='coerce')
        raw['actual_shares'] = pd.to_numeric(raw['actual_shares'], errors='coerce')
        raw['actual_mv'] = pd.to_numeric(raw['actual_mv'], errors='coerce')
        raw['float_ratio'] = pd.to_numeric(raw['float_ratio'], errors='coerce')
        raw = raw.dropna(subset=['release_date'])
        raw = _ensure_effective_date(raw)
        all_frames.append(raw)
        print(f"  {year}: {len(raw)} events")

    if not all_frames:
        raise RuntimeError("未能下载任何限售解禁数据")

    df = pd.concat(all_frames, ignore_index=True)
    df = _ensure_effective_date(df)
    df = df.sort_values('release_date').reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"已保存: {cache_path} ({len(df)} rows, {df['code'].nunique()} stocks)")
    return df


def download_update():
    """
    增量更新：仅下载缓存中最新解禁日期之后的数据。
    如果缓存不存在则全量下载。
    """
    cache_path = Path(CACHE_FILE)
    if not cache_path.exists():
        return download_all()

    existing = _ensure_effective_date(pd.read_parquet(cache_path))
    latest_date = existing['release_date'].max()
    print(f"缓存最新解禁日期: {latest_date.date()}")

    import akshare as ak

    end_date = (datetime.now() + timedelta(days=365)).strftime('%Y%m%d')
    start_date = (latest_date - timedelta(days=30)).strftime('%Y%m%d')  # 30天重叠避免遗漏

    for attempt in range(3):
        try:
            raw = _quiet_fetch(ak.stock_restricted_release_detail_em,
                               start_date=start_date, end_date=end_date)
            break
        except Exception:
            if attempt < 2:
                time.sleep(3)
            else:
                raw = None

    if raw is None or raw.empty:
        print("增量更新: 无新数据")
        return existing

    cols = raw.columns.tolist()
    if len(cols) < 9:
        print(f"增量更新: 列数不足 ({len(cols)})")
        return existing

    raw = raw.rename(columns={
        cols[1]: 'code', cols[3]: 'release_date',
        cols[6]: 'actual_shares', cols[7]: 'actual_mv', cols[8]: 'float_ratio',
    })
    raw = raw[['code', 'release_date', 'actual_shares', 'actual_mv', 'float_ratio']].copy()
    raw['code'] = raw['code'].astype(str).str.zfill(6)
    raw['release_date'] = pd.to_datetime(raw['release_date'], errors='coerce')
    raw['actual_shares'] = pd.to_numeric(raw['actual_shares'], errors='coerce')
    raw['actual_mv'] = pd.to_numeric(raw['actual_mv'], errors='coerce')
    raw['float_ratio'] = pd.to_numeric(raw['float_ratio'], errors='coerce')
    raw = raw.dropna(subset=['release_date'])
    raw = _ensure_effective_date(raw)

    new_events = raw[raw['release_date'] > latest_date]
    if new_events.empty:
        print("增量更新: 无新事件")
        return existing

    df = pd.concat([existing, new_events], ignore_index=True)
    df = _ensure_effective_date(df)
    df = df.drop_duplicates(subset=['code', 'release_date'], keep='last')
    df = df.sort_values('release_date').reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    print(f"增量更新: +{len(new_events)} 事件 (总{len(df)} rows, {df['code'].nunique()} stocks)")
    return df


def merge_to_daily(restricted_df, codes, all_dates):
    """
    PIT 对齐：对每个交易日，计算已披露的未来解禁压力特征。
    无公告日字段时使用 release_date - DEFAULT_NOTICE_LAG_DAYS 作为保守可用日。
    """
    df = restricted_df.copy()
    all_dates = pd.DatetimeIndex(all_dates)
    result = pd.DataFrame(index=all_dates, dtype=np.float32)
    all_dates_d = all_dates.values.astype('datetime64[D]')
    n_dates = len(all_dates)

    for code in codes:
        pure = code.replace('.SH', '').replace('.SZ', '')
        code_data = df[df['code'] == pure]

        if code_data.empty:
            for col in RESTRICTED_COLS:
                result[f'{code}_{col}'] = 0.0
            continue

        if 'effective_date' not in code_data.columns:
            code_data = code_data.copy()
            code_data['effective_date'] = code_data['release_date'] - pd.Timedelta(days=DEFAULT_NOTICE_LAG_DAYS)
        releases = code_data[['effective_date', 'release_date', 'float_ratio']].dropna().sort_values('release_date')

        next_inv_days = np.zeros(n_dates, dtype=np.float32)
        next_ratio = np.zeros(n_dates, dtype=np.float32)
        ratio_90d = np.zeros(n_dates, dtype=np.float32)

        for row in releases.itertuples(index=False):
            eff = np.datetime64(pd.Timestamp(row.effective_date).date())
            rel = np.datetime64(pd.Timestamp(row.release_date).date())
            ratio = float(row.float_ratio)
            active = (all_dates_d >= eff) & (all_dates_d < rel)
            if not active.any():
                continue
            deltas = (rel - all_dates_d[active]).astype('timedelta64[D]').astype(int)
            current_days = np.where(next_inv_days[active] > 0, 1.0 / next_inv_days[active] - 1.0, np.inf)
            replace = deltas < current_days
            active_idx = np.where(active)[0]
            idx = active_idx[replace]
            next_inv_days[idx] = 1.0 / (1.0 + deltas[replace].astype(np.float32))
            next_ratio[idx] = ratio

            in_90d = active & (rel <= all_dates_d + np.timedelta64(90, 'D'))
            ratio_90d[in_90d] += ratio

        result[f'{code}_restricted_next_inv_days'] = next_inv_days
        result[f'{code}_restricted_next_ratio'] = next_ratio
        result[f'{code}_restricted_mv_ratio_90d'] = ratio_90d

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="限售解禁数据下载")
    parser.add_argument('--update', action='store_true', help='增量更新')
    parser.add_argument('--force', action='store_true', help='强制全量重下')
    args = parser.parse_args()

    if args.update:
        df = download_update()
    else:
        df = download_all(force=args.force)

    print(f"\n数据概览:")
    print(f"  日期范围: {df['release_date'].min()} ~ {df['release_date'].max()}")
    print(f"  总事件: {len(df)}, 股票数: {df['code'].nunique()}")
