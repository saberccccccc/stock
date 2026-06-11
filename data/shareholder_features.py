# shareholder_features.py - 股东户数PIT特征（筹码集中度）
import io
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

CACHE_FILE = "cache/shareholder_features.parquet"
SH_COLS = ['sh_conc_ratio', 'sh_per_capita_ratio']


def _quarter_dates():
    """生成 2013Q1 ~ 当前季度的季度末日期列表（akshare 最早到 2013Q1）"""
    dates = []
    for y in range(2013, 2027):
        for m, d in [('03', '31'), ('06', '30'), ('09', '30'), ('12', '31')]:
            dt = f'{y}{m}{d}'
            if dt > datetime.now().strftime('%Y%m%d'):
                break
            dates.append(dt)
    return dates


def _quiet_fetch(func, *args, **kwargs):
    """调用 akshare 函数，吞掉内部进度条输出"""
    f_out = io.StringIO()
    f_err = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
        return func(*args, **kwargs)


def download_all(force=False):
    """
    下载全部季度股东户数数据，存为 parquet。
    已存在时跳过（除非 force=True）。
    Returns: DataFrame
    """
    cache_path = Path(CACHE_FILE)
    if cache_path.exists() and not force:
        df = pd.read_parquet(cache_path)
        print(f"从缓存加载股东户数: {len(df)} rows, {df['code'].nunique()} stocks")
        return df

    import akshare as ak

    quarters = _quarter_dates()
    print(f"下载股东户数: {len(quarters)} 个季度 ({quarters[0]} ~ {quarters[-1]})")

    all_frames = []
    for q in quarters:
        for attempt in range(3):
            try:
                raw = _quiet_fetch(ak.stock_zh_a_gdhs, symbol=q)
                break
            except Exception:
                if attempt < 2:
                    time.sleep(3)
                else:
                    raw = None

        if raw is None or raw.empty:
            print(f"  {q}: 获取失败")
            continue

        raw = raw.copy()
        raw['quarter_end'] = q

        # akshare stock_zh_a_gdhs 列序固定，用位置映射避免中文编码问题
        # [0]股票代码 [7]股东户数-变化比例 [11]户均持股金额 [13]市值 [15]公告日期
        cols = raw.columns.tolist()
        if len(cols) < 16:
            print(f"  {q}: 列数不足 ({len(cols)})，跳过")
            continue

        raw = raw.rename(columns={
            cols[0]: 'code',
            cols[7]: 'sh_change_raw',
            cols[11]: 'per_capita_value',
            cols[13]: 'market_cap',
            cols[15]: 'announce_date',
        })
        raw = raw[['code', 'sh_change_raw', 'per_capita_value', 'market_cap', 'announce_date']].copy()
        raw['quarter_end'] = q
        all_frames.append(raw)
        print(f"  {q}: {len(raw)} stocks")

    if not all_frames:
        raise RuntimeError("未能下载任何股东户数数据")

    df = pd.concat(all_frames, ignore_index=True)

    # 清洗
    df['code'] = df['code'].astype(str).str.zfill(6)
    # 判断沪深：代码规则判断（简单处理，后续可扩展）
    # 目前保留纯数字代码，merge时再匹配
    df['announce_date'] = pd.to_datetime(df['announce_date'], errors='coerce')
    df['sh_change_raw'] = pd.to_numeric(df['sh_change_raw'], errors='coerce')
    df['per_capita_value'] = pd.to_numeric(df.get('per_capita_value', np.nan), errors='coerce')
    df['market_cap'] = pd.to_numeric(df.get('market_cap', np.nan), errors='coerce')

    # 构造特征
    # sh_conc_ratio: 负的股东户数变化比例，正值=筹码集中=看涨，clip ±200
    df['sh_conc_ratio'] = np.clip(-df['sh_change_raw'], -200, 200)
    # sh_per_capita_ratio: 户均持股市值 / 总市值 ×10000（放大避免float32精度丢失）
    if 'per_capita_value' in df.columns and 'market_cap' in df.columns:
        raw_ratio = np.where(
            df['market_cap'] > 0,
            df['per_capita_value'] / df['market_cap'],
            0.0
        )
        df['sh_per_capita_ratio'] = raw_ratio * 10000.0
    else:
        df['sh_per_capita_ratio'] = 0.0

    df = df.dropna(subset=['announce_date'])
    df = df.drop(columns=['sh_change_raw', 'per_capita_value', 'market_cap'], errors='ignore')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"已保存: {cache_path} ({len(df)} rows, {df['code'].nunique()} stocks)")
    return df


def merge_to_daily(shareholder_df, codes, all_dates):
    """
    PIT 对齐：对每个交易日，取公告日 <= 当日的最近季度数据，前向填充。

    Args:
        shareholder_df: download_all() 返回的 DataFrame
        codes: 股票代码列表（tushare格式 000001.SZ）
        all_dates: 交易日 DatetimeIndex

    Returns:
        DataFrame index=all_dates, columns=[{code}_{col} for code in codes for col in SH_COLS]
    """
    df = shareholder_df.copy()
    df = df.dropna(subset=['announce_date']).sort_values('announce_date')

    result = pd.DataFrame(index=all_dates, dtype=np.float32)

    # 获取所有唯一的公告日期作为"版本快照"
    announce_dates = sorted(df['announce_date'].unique())

    for code in codes:
        pure = code.replace('.SH', '').replace('.SZ', '')
        code_data = df[df['code'] == pure].copy()

        if code_data.empty:
            for col in SH_COLS:
                result[f'{code}_{col}'] = 0.0
            continue

        # 按公告日期排序，取每个公告日期对应的最新 quarter_end
        code_data = code_data.sort_values(['announce_date', 'quarter_end'])
        # 对每个公告日，取最新季度的值（去重保留最后）
        code_data = code_data.drop_duplicates(subset=['announce_date'], keep='last')
        code_data = code_data.set_index('announce_date').sort_index()

        for col in SH_COLS:
            if col not in code_data.columns:
                result[f'{code}_{col}'] = 0.0
                continue
            series = code_data[col].dropna()
            if series.empty:
                result[f'{code}_{col}'] = 0.0
                continue
            # Reindex 到所有交易日，前向填充
            daily = series.reindex(all_dates, method='ffill').fillna(0.0)
            result[f'{code}_{col}'] = daily.values

    return result


def download_update():
    """
    增量更新：仅下载缓存中最新季度之后的季度数据。
    如果缓存不存在则全量下载。
    """
    cache_path = Path(CACHE_FILE)
    if not cache_path.exists():
        return download_all()

    existing = pd.read_parquet(cache_path)
    latest_quarter = str(existing['quarter_end'].max())
    print(f"缓存最新季度: {latest_quarter}")

    all_quarters = _quarter_dates()
    # 找需要更新的季度（latest_quarter 之后的）
    try:
        idx = all_quarters.index(latest_quarter)
        new_quarters = all_quarters[idx + 1:]
    except ValueError:
        new_quarters = all_quarters

    if not new_quarters:
        print("增量更新: 已是最新")
        return existing

    print(f"增量更新: {len(new_quarters)} 个季度 ({new_quarters[0]} ~ {new_quarters[-1]})")

    import akshare as ak
    new_frames = []
    for q in new_quarters:
        for attempt in range(3):
            try:
                raw = _quiet_fetch(ak.stock_zh_a_gdhs, symbol=q)
                break
            except Exception:
                if attempt < 2:
                    time.sleep(3)
                else:
                    raw = None

        if raw is None or raw.empty:
            print(f"  {q}: 获取失败")
            continue

        raw = raw.copy()
        cols = raw.columns.tolist()
        if len(cols) < 16:
            print(f"  {q}: 列数不足 ({len(cols)})，跳过")
            continue

        raw = raw.rename(columns={
            cols[0]: 'code', cols[7]: 'sh_change_raw',
            cols[11]: 'per_capita_value', cols[13]: 'market_cap', cols[15]: 'announce_date',
        })
        raw = raw[['code', 'sh_change_raw', 'per_capita_value', 'market_cap', 'announce_date']].copy()
        raw['code'] = raw['code'].astype(str).str.zfill(6)
        raw['announce_date'] = pd.to_datetime(raw['announce_date'], errors='coerce')
        raw['sh_change_raw'] = pd.to_numeric(raw['sh_change_raw'], errors='coerce')
        raw['per_capita_value'] = pd.to_numeric(raw.get('per_capita_value', np.nan), errors='coerce')
        raw['market_cap'] = pd.to_numeric(raw.get('market_cap', np.nan), errors='coerce')
        raw['sh_conc_ratio'] = np.clip(-raw['sh_change_raw'], -200, 200)
        if 'per_capita_value' in raw.columns and 'market_cap' in raw.columns:
            raw_ratio = np.where(
                raw['market_cap'] > 0,
                raw['per_capita_value'] / raw['market_cap'],
                0.0
            )
            raw['sh_per_capita_ratio'] = raw_ratio * 10000.0
        else:
            raw['sh_per_capita_ratio'] = 0.0
        raw['quarter_end'] = q
        raw = raw.dropna(subset=['announce_date'])
        raw = raw.drop(columns=['sh_change_raw', 'per_capita_value', 'market_cap'], errors='ignore')
        new_frames.append(raw)
        print(f"  {q}: {len(raw)} stocks")

    if not new_frames:
        print("增量更新: 无新数据")
        return existing

    new_data = pd.concat(new_frames, ignore_index=True)
    df = pd.concat([existing, new_data], ignore_index=True)
    df = df.drop_duplicates(subset=['code', 'quarter_end'], keep='last')
    df = df.sort_values('announce_date').reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    print(f"增量更新完成: +{len(new_data)} rows (总{len(df)} rows, {df['code'].nunique()} stocks)")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="股东户数数据下载")
    parser.add_argument('--update', action='store_true', help='增量更新')
    parser.add_argument('--force', action='store_true', help='强制全量重下')
    args = parser.parse_args()

    if args.update:
        df = download_update()
    else:
        df = download_all(force=args.force)

    print(f"\n数据概览:")
    print(f"  日期范围: {df['announce_date'].min()} ~ {df['announce_date'].max()}")
    print(f"  季度数: {df['quarter_end'].nunique()}")
    print(f"  股票数: {df['code'].nunique()}")
    print(f"  sh_conc_ratio 分布:")
    print(df['sh_conc_ratio'].describe())
