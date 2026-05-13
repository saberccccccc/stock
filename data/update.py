# update_data.py - 股票日线数据更新脚本
"""
用途：
  1. 将数据起始日期从 2017 提前到 2010
  2. 增量更新到最新交易日
  3. 重建截面数据集缓存

用法：
  python update_data.py                      # 增量更新（仅获取缺失日期）
  python update_data.py --extend 20100101    # 扩展历史 + 更新最新
  python update_data.py --rebuild-cache      # 更新后重做缓存
"""
import os
import sys
import time
import random
import argparse
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta


def resolve_tushare_token(token=None):
    resolved = token or os.getenv('TUSHARE_TOKEN')
    if not resolved:
        raise ValueError('缺少 TUSHARE_TOKEN，请设置环境变量或通过 --token 传入')
    return resolved

# ==================== TushareProLite（精简自用版本）===================
class TushareProLite:
    def __init__(self, token, max_workers=3, min_interval=2.0):
        import tushare as ts
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.max_workers = max_workers
        self.min_interval = min_interval
        self.last_request_time = 0
        self.request_lock = threading.Lock()

    def _safe_call(self, func, *args, **kwargs):
        for attempt in range(3):
            try:
                with self.request_lock:
                    elapsed = time.time() - self.last_request_time
                    if elapsed < self.min_interval:
                        time.sleep(self.min_interval - elapsed + random.uniform(0.2, 0.5))
                    self.last_request_time = time.time()
                return func(*args, **kwargs)
            except Exception as e:
                print(f"  重试 {attempt+1}/3: {e}")
                time.sleep(5 * (attempt + 1))
        return None

    def get_stable_stocks(self, data_dir, min_years=1):
        path = os.path.join(data_dir, 'stable_stocks.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, dtype=str)
            return df['ts_code'].tolist()

        frames = []
        for status in ['L', 'D', 'P']:
            df_status = self._safe_call(
                self.pro.stock_basic, exchange='', list_status=status,
                fields='ts_code,symbol,name,list_date,area,industry')
            if df_status is not None and not df_status.empty:
                frames.append(df_status)
        if not frames:
            return []

        df = pd.concat(frames, ignore_index=True).drop_duplicates('ts_code')
        threshold = (datetime.today() - timedelta(days=min_years * 365)).strftime('%Y%m%d')
        stable = df[df['list_date'] <= threshold]
        stable.to_csv(path, index=False)
        print(f"got {len(stable)} stable stocks")
        return stable['ts_code'].tolist()

    def update_single_stock(self, ts_code, data_dir, start_date='20100101'):
        """智能增量更新：补全缺失的历史 + 追加最新数据"""
        end_date = datetime.today().strftime('%Y%m%d')
        csv_path = os.path.join(data_dir, f"{ts_code}.csv")

        # 读取本地数据
        if os.path.exists(csv_path):
            try:
                local = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            except Exception:
                local = pd.DataFrame()
        else:
            local = pd.DataFrame()

        if not local.empty:
            local_last = local.index.max().strftime('%Y%m%d')
            local_first = local.index.min().strftime('%Y%m%d')

            # 情况1：本地已覆盖全部范围 -> 只补最新
            if local_first <= start_date and local_last >= end_date:
                return local

            need_history = (local_first > start_date)
            need_latest = (local_last < end_date)

            # 情况2：只补历史
            if need_history and not need_latest:
                new = self._fetch_range(ts_code, start_date, local_first, subtract=1)
                if new is not None and not new.empty:
                    combined = pd.concat([new, local]).sort_index()
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined.to_csv(csv_path)
                    return combined

            # 情况3：只补最新
            if need_latest and not need_history:
                next_day = (local.index.max() + timedelta(days=1)).strftime('%Y%m%d')
                new = self._fetch_range(ts_code, next_day, end_date)
                if new is not None and not new.empty:
                    combined = pd.concat([local, new]).sort_index()
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined.to_csv(csv_path)
                    return combined

            # 情况4：两头都缺 -> 需要全量
            if need_history and need_latest:
                new_full = self._fetch_range(ts_code, start_date, end_date)
                if new_full is not None and not new_full.empty:
                    combined = pd.concat([local, new_full]).sort_index()
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined.to_csv(csv_path)
                    return combined
        else:
            # 全新下载
            new = self._fetch_range(ts_code, start_date, end_date)
            if new is not None and not new.empty:
                new.to_csv(csv_path)
                return new

        return local

    def _fetch_range(self, ts_code, start_date, end_date, subtract=0):
        """Get daily data for specified date range"""
        if subtract > 0:
            end_dt = pd.to_datetime(end_date) - timedelta(days=subtract)
            end_date = end_dt.strftime('%Y%m%d')

        df = self._safe_call(
            self.pro.daily, ts_code=ts_code,
            start_date=start_date, end_date=end_date,
            fields='ts_code,trade_date,open,high,low,close,vol,amount')

        if df is None or df.empty:
            return pd.DataFrame()

        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        df.rename(columns={'vol': 'volume', 'amount': 'money'}, inplace=True)
        df['factor'] = 1.0
        df['code'] = ts_code
        df = df[['code', 'open', 'high', 'low', 'close', 'volume', 'money', 'factor']]
        return df


# ==================== 批量更新 ====================
def batch_update(data_dir, token, start_date='20100101', max_workers=3,
                 batch_size=200, batch_sleep=30, limit=None):
    """
    批量增量更新所有股票

    Args:
        data_dir: 数据目录
        token: tushare token
        start_date: 起始日期（YYYYMMDD）
        max_workers: 并发数
        batch_sleep: 批次间休眠秒数（避免限流）
        limit: 限制股票数（None=全部）
    """
    td = TushareProLite(token, max_workers=max_workers)
    stocks = td.get_stable_stocks(data_dir)
    print(f"稳定股票总数: {len(stocks)}")

    if limit:
        stocks = stocks[:limit]
        print(f"限制为前 {limit} 只")

    total = len(stocks)
    updated, skipped, failed = 0, 0, 0

    for start in range(0, total, batch_size):
        batch = stocks[start:start + batch_size]
        batch_num = start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"\n{'='*60}")
        print(f"批次 {batch_num}/{total_batches}: {len(batch)} 只股票")
        print(f"{'='*60}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for code in batch:
                futures[executor.submit(
                    td.update_single_stock, code, data_dir, start_date
                )] = code

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"批次 {batch_num}"):
                code = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        updated += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"  {code} 失败: {e}")
                    failed += 1

        print(f"进度: {start + len(batch)}/{total}  |  "
              f"更新:{updated} 跳过:{skipped} 失败:{failed}")

        if start + batch_size < total:
            print(f"批次间休眠 {batch_sleep}s...")
            time.sleep(batch_sleep)

    print(f"\n{'='*60}")
    print(f"更新完成! 更新:{updated} 跳过:{skipped} 失败:{failed}")
    print(f"{'='*60}")


# ==================== 更新宏观数据缓存 ====================
def update_macro_cache():
    """删除旧缓存，重新下载宏观数据"""
    cache_dir = "cache"
    for f in ['north_flow.csv', 'margin_balance.csv', 'pmi.csv', 'pmi_pit_v2.csv']:
        path = os.path.join(cache_dir, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"已删除旧缓存: {f}")

    from data.macro_factors import fetch_north_flow, fetch_margin_balance, fetch_pmi
    fetch_north_flow()
    fetch_margin_balance()
    fetch_pmi()
    print("宏观数据缓存已更新")


# ==================== 重建截面缓存 ====================
def rebuild_cross_cache():
    """删除旧截面缓存，触发重建"""
    cache_dir = "cache"
    for f in os.listdir(cache_dir):
        if f.startswith('cross_section_'):
            path = os.path.join(cache_dir, f)
            os.remove(path)
            print(f"已删除缓存 {f}")

    print("\n下次运行 data_pipeline 时将自动重建截面数据集")


# ==================== 主程序====================
def main():
    parser = argparse.ArgumentParser(description='股票日线数据更新工具')
    parser.add_argument('--token', type=str, default=None,
                        help='Tushare token；未传入时读取 TUSHARE_TOKEN 环境变量')
    parser.add_argument('--data-dir', type=str,
                        default='data/raw',
                        help='数据目录')
    parser.add_argument('--start-date', type=str, default='20100101',
                        help='起始日期 YYYYMMDD (默认20100101)')
    parser.add_argument('--extend', type=str, nargs='?', const='20100101',
                        help='extend history data to specified date (default 20100101)')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='重建截面数据缓存')
    parser.add_argument('--update-macro', action='store_true',
                        help='更新宏观因子缓存')
    parser.add_argument('--limit', type=int, default=None,
                        help='update stock data to latest date')
    parser.add_argument('--workers', type=int, default=3,
                        help='并发数（默认3，避免限流）')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='每批股票数（默认200）')
    parser.add_argument('--batch-sleep', type=int, default=45,
                        help='批次间休眠秒数（默认45）')

    args = parser.parse_args()
    start_date = args.extend if args.extend else args.start_date
    token = resolve_tushare_token(args.token)

    print(f"{'='*60}")
    print(f"数据更新工具")
    print(f"  数据目录: {args.data_dir}")
    print(f"  起始日期: {start_date}")
    print(f"  并发数   {args.workers}")
    print(f"  每批股票: {args.batch_size} (批次间休眠 {args.batch_sleep}s)")
    print(f"{'='*60}")

    batch_update(
        data_dir=args.data_dir,
        token=token,
        start_date=start_date,
        max_workers=args.workers,
        batch_size=args.batch_size,
        batch_sleep=args.batch_sleep,
        limit=args.limit,
    )

    if args.update_macro:
        update_macro_cache()

    if args.rebuild_cache:
        rebuild_cross_cache()

    print("\n全部完成!")


# ==================== 指数与行业数据下载（原fetch_data.py）====================
def fetch_index_data(symbol, save_path, start_date="2010-01-01", end_date="2026-12-31"):
    """下载指数日线数据"""
    import akshare as ak
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = ak.stock_zh_index_daily(symbol=symbol)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df.to_csv(save_path)
    print(f"指数数据已保存至 {save_path}，共 {len(df)} 条")


def fetch_hs300_data(save_path="data/raw/hs300_index.csv", start_date="2010-01-01", end_date="2026-12-31"):
    """下载沪深300指数日线数据"""
    fetch_index_data("sh000300", save_path, start_date, end_date)


def fetch_broad_index_data(data_dir="data/raw", start_date="2010-01-01", end_date="2026-12-31"):
    indices = {
        "hs300_index.csv": "sh000300",
        "sz50_index.csv": "sh000016",
        "zz500_index.csv": "sh000905",
        "cyb_index.csv": "sz399006",
    }
    for filename, symbol in indices.items():
        fetch_index_data(symbol, os.path.join(data_dir, filename), start_date, end_date)


def fetch_industry_data(save_path="data/stock_industry.csv"):
    """下载A股行业分类数据"""
    import baostock as bs
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return
    rs = bs.query_stock_industry()
    if rs.error_code != '0':
        print(f"查询失败: {rs.error_msg}")
        bs.logout()
        return
    industry_list = []
    while rs.next():
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    result.to_csv(save_path, index=False)
    bs.logout()
    print(f"行业数据已保存至 {save_path}，共 {len(result)} 条")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        fetch_broad_index_data()
        fetch_industry_data()
    else:
        main()
