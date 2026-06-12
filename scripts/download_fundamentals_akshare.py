#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立脚本：用 akshare 多线程下载基本面数据，缓存到 parquet。

用法:
  python scripts/download_fundamentals_akshare.py              # 全量下载
  python scripts/download_fundamentals_akshare.py --update      # 增量更新
  python scripts/download_fundamentals_akshare.py --workers 16
  python scripts/download_fundamentals_akshare.py --test 10

输出: cache/fundamental_features_akshare.parquet
"""

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

CACHE_FILE = PROJECT_ROOT / "cache" / "fundamental_features_akshare.parquet"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# 利润表关键字段 → 统一命名
PROFIT_COL_MAP = {
    'OPERATE_INCOME': 'revenue',
    'OPERATE_INCOME_YOY': 'revenue_yoy',
    'PARENT_NETPROFIT': 'net_profit',
    'REPORT_DATE': 'end_date',
    'NOTICE_DATE': 'notice_date',
    'REPORT_DATE_NAME': 'period_name',
}

# 资产负债表关键字段（akshare 列名 → 统一命名）
BALANCE_COL_MAP = {
    'TOTAL_PARENT_EQUITY': 'parent_equity',
    'TOTAL_EQUITY': 'total_equity',
    'REPORT_DATE': 'end_date',
    'NOTICE_DATE': 'notice_date',
}


def _fallback_effective_date(end_date):
    """Conservative report availability date when announcement date is missing."""
    end_date = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(end_date):
        return pd.NaT
    month_day = (end_date.month, end_date.day)
    if month_day == (3, 31):
        return end_date + pd.Timedelta(days=45)
    if month_day == (6, 30):
        return end_date + pd.Timedelta(days=60)
    if month_day == (9, 30):
        return end_date + pd.Timedelta(days=45)
    if month_day == (12, 31):
        return end_date + pd.Timedelta(days=120)
    return end_date + pd.Timedelta(days=90)


def _expected_latest_report_period(as_of=None):
    """Latest statutory report period expected to be public by the given date."""
    as_of = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of)
    year = as_of.year
    month_day = (as_of.month, as_of.day)
    if month_day >= (11, 1):
        return pd.Timestamp(year=year, month=9, day=30)
    if month_day >= (9, 1):
        return pd.Timestamp(year=year, month=6, day=30)
    if month_day >= (5, 1):
        return pd.Timestamp(year=year, month=3, day=31)
    if month_day >= (4, 30):
        return pd.Timestamp(year=year - 1, month=12, day=31)
    return pd.Timestamp(year=year - 1, month=9, day=30)


def get_stock_list():
    """从 data/raw/ 获取股票列表"""
    csv_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith('.csv') and f[0].isdigit()
    )
    return [f.replace('.csv', '') for f in csv_files]


def ts_code_to_ak(code):
    """Tushare code → akshare symbol: 000001.SZ → sz000001"""
    if code.endswith('.SZ'):
        return f'sz{code[:6]}'
    elif code.endswith('.SH'):
        return f'sh{code[:6]}'
    return code


def _quiet_ak_call(func, *args, **kwargs):
    """调用 akshare 函数，吞掉内部的 tqdm 输出（stdout + stderr）"""
    import contextlib
    f_out = io.StringIO()
    f_err = io.StringIO()
    with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
        result = func(*args, **kwargs)
    return result


def fetch_one_stock(code, max_retries=2):
    """
    拉取单只股票的利润表 + 资产负债表。
    返回 DataFrame 或 None。
    """
    import akshare as ak

    ak_code = ts_code_to_ak(code)

    # ---- 利润表 ----
    for attempt in range(max_retries):
        try:
            df = _quiet_ak_call(ak.stock_profit_sheet_by_report_em, symbol=ak_code)
            break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None

    if df is None or df.empty:
        return None

    # 重命名并保留需要的列
    df = df.rename(columns=PROFIT_COL_MAP)
    keep_cols = [v for k, v in PROFIT_COL_MAP.items() if v in df.columns]
    if 'revenue' not in df.columns:
        return None
    profit = df[keep_cols].copy()
    profit['ts_code'] = code

    # ---- 资产负债表 ----
    for attempt in range(max_retries):
        try:
            df_bs = _quiet_ak_call(ak.stock_balance_sheet_by_report_em, symbol=ak_code)
            break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None

    equity_col = None
    if df_bs is not None and not df_bs.empty:
        df_bs = df_bs.rename(columns=BALANCE_COL_MAP)
        for c in ['parent_equity', 'total_equity']:
            if c in df_bs.columns:
                equity_col = c
                bs_cols = ['end_date', c]
                if 'notice_date' in df_bs.columns:
                    bs_cols.append('notice_date')
                bs = df_bs[bs_cols].copy()
                break

    # ---- 合并计算 ROE ----
    profit['end_date'] = pd.to_datetime(profit['end_date'], errors='coerce')
    profit['notice_date'] = pd.to_datetime(profit['notice_date'], errors='coerce')
    profit['_key'] = profit['end_date'].astype(str)

    if equity_col:
        bs['end_date'] = pd.to_datetime(bs['end_date'], errors='coerce')
        if 'notice_date' in bs.columns:
            bs['balance_notice_date'] = pd.to_datetime(bs['notice_date'], errors='coerce')
        else:
            bs['balance_notice_date'] = pd.NaT
        bs['_key'] = bs['end_date'].astype(str)
        merged = profit.merge(
            bs[['_key', equity_col, 'balance_notice_date']],
            on='_key',
            how='left',
        )
        merged[equity_col] = pd.to_numeric(merged[equity_col], errors='coerce')
        merged.drop(columns=['_key'], inplace=True)
    else:
        merged = profit
        merged['parent_equity'] = np.nan
        merged['balance_notice_date'] = pd.NaT
        merged.drop(columns=['_key'], inplace=True)

    merged['revenue'] = pd.to_numeric(merged['revenue'], errors='coerce')
    merged['net_profit'] = pd.to_numeric(merged.get('net_profit', np.nan), errors='coerce')
    merged['revenue_yoy'] = pd.to_numeric(merged.get('revenue_yoy', np.nan), errors='coerce')

    equity_vals = merged.get('parent_equity', np.nan)
    merged['roe'] = np.where(
        pd.to_numeric(equity_vals, errors='coerce') > 0,
        merged['net_profit'] / pd.to_numeric(equity_vals, errors='coerce'),
        np.nan
    )

    fallback_dates = merged['end_date'].map(_fallback_effective_date)
    notice_dates = merged[['notice_date', 'balance_notice_date']].max(axis=1)
    merged['effective_date'] = notice_dates.fillna(fallback_dates)

    result = merged[['ts_code', 'effective_date', 'end_date', 'roe', 'revenue_yoy']].copy()
    result = result.dropna(subset=['effective_date'])

    # 数据清洗：过滤早于1990年的无效日期
    result = result[result['effective_date'] >= '1990-01-01']

    # Winsorize: ROE clip ±500%, revenue_yoy clip ±200%
    result['roe'] = np.clip(pd.to_numeric(result['roe'], errors='coerce'), -5.0, 5.0)
    result['revenue_yoy'] = np.clip(pd.to_numeric(result['revenue_yoy'], errors='coerce'), -200.0, 200.0)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--update', action='store_true',
                        help='仅更新最近财报（90天内数据跳过）')
    args = parser.parse_args()

    os.makedirs(CACHE_FILE.parent, exist_ok=True)

    all_codes = get_stock_list()
    if args.test:
        all_codes = all_codes[:args.test]
        print(f"[TEST] 仅处理前 {len(all_codes)} 只股票")

    print(f"待处理: {len(all_codes)} 只股票")
    print(f"并行线程: {args.workers}")

    existing_codes = set()
    if args.update and CACHE_FILE.exists():
        try:
            existing = pd.read_parquet(CACHE_FILE)
            existing_codes = set(existing['ts_code'].unique())
            existing['end_date'] = pd.to_datetime(existing['end_date'], errors='coerce')
            expected_period = _expected_latest_report_period()
            latest_periods = existing.groupby('ts_code')['end_date'].max()
            stale_codes = set(latest_periods[latest_periods < expected_period].index)
            remaining = [c for c in all_codes if c in stale_codes]
            new_codes = [c for c in all_codes if c not in existing_codes]
            remaining = remaining + new_codes
            print(
                f"已有缓存: {len(existing_codes)} 只, 应有报告期: "
                f"{expected_period.date()}, 需更新: {len(remaining)} 只"
            )
        except Exception:
            remaining = all_codes
    else:
        if CACHE_FILE.exists():
            try:
                existing = pd.read_parquet(CACHE_FILE)
                existing_codes = set(existing['ts_code'].unique())
                print(f"已有缓存: {len(existing_codes)} 只股票")
            except Exception:
                pass
        remaining = [c for c in all_codes if c not in existing_codes]

    print(f"待下载: {len(remaining)} 只")

    if not remaining:
        print("全部完成！")
        return

    all_results = []
    n_done = len(existing_codes)
    n_fail = 0
    SAVE_EVERY = 500  # 每500只保存一次，防止崩溃丢数据

    def _save_checkpoint(new_frames):
        """增量保存：合并现有缓存 + 新数据，去重写入"""
        if not new_frames:
            return
        new_data = pd.concat(new_frames, ignore_index=True)
        if CACHE_FILE.exists():
            old_data = pd.read_parquet(CACHE_FILE)
            new_data = pd.concat([old_data, new_data], ignore_index=True)
        new_data = new_data.sort_values(['ts_code', 'end_date', 'effective_date'])
        new_data = new_data.drop_duplicates(subset=['ts_code', 'end_date'], keep='last')
        new_data.to_parquet(CACHE_FILE, index=False)
        print(f"\n  [checkpoint] 已保存 {len(new_frames)} 条新数据 (总{new_data['ts_code'].nunique()}只)")

    start_time = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_one_stock, code): code for code in remaining}
            with tqdm(total=len(remaining), desc="下载基本面") as pbar:
                for fut in as_completed(futures):
                    code = futures[fut]
                    try:
                        df = fut.result(timeout=120)
                        if df is not None and not df.empty:
                            all_results.append(df)
                        else:
                            n_fail += 1
                    except Exception:
                        n_fail += 1
                    n_done += 1
                    pbar.set_postfix_str(f"完成={n_done}/{len(all_codes)} 失败={n_fail}")
                    pbar.update(1)

                    if len(all_results) >= SAVE_EVERY:
                        _save_checkpoint(all_results)
                        all_results.clear()

    except KeyboardInterrupt:
        print("\n中断，保存已有数据...")
    except Exception as e:
        print(f"\n错误: {e}，保存已有数据...")

    # 最终保存
    _save_checkpoint(all_results)
    all_results.clear()


if __name__ == "__main__":
    main()
