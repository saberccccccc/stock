# update_daily_data.py - 个股日线智能增量更新（断点续传版）
# 用法: python update_daily_data.py [--workers 2]
# 可随时 Ctrl+C 中断，重新运行会自动跳过已更新的股票

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.api_utils import SafeAPICaller, resolve_tushare_token

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_DIR = "data/tracking_raw"  # 日更写入 tracking_raw，避免污染训练/回测全量数据
START_DATE = "20100101"
MIN_INTERVAL = 1.5       # API最小间隔（秒）
BATCH_SLEEP = 60         # 批次间休息（秒）
MIN_SAFE_ROWS = 200      # 如果本地已有文件超过此行数，绝不允许覆盖为更少的行

PROGRESS_FILE = os.path.join(DATA_DIR, "_update_progress.json")
os.makedirs(DATA_DIR, exist_ok=True)

_api_call = SafeAPICaller(
    min_interval=MIN_INTERVAL,
    max_retries=3,
    retry_base_delay=4.0,
    jitter=(0.2, 0.4),
    data_source="tushare",
)


def safe_call(func, *args, **kwargs):
    return _api_call(func, *args, **kwargs)


def fetch_one_stock(pro, ts_code, start_date, end_date):
    """全量获取单只股票日线"""
    df = safe_call(
        pro.daily, ts_code=ts_code,
        start_date=start_date, end_date=end_date,
        fields='ts_code,trade_date,open,high,low,close,vol,amount'
    )
    if df is None or df.empty:
        return None
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)
    df.rename(columns={'vol': 'volume', 'amount': 'money'}, inplace=True)
    df['factor'] = 1.0
    df['code'] = ts_code
    return df[['code', 'open', 'high', 'low', 'close', 'volume', 'money', 'factor']]


def fetch_batch_stocks(pro, ts_codes, start_date, end_date):
    """批量获取多只股票日线（单次API调用）"""
    code_str = ','.join(ts_codes)
    df = safe_call(
        pro.daily, ts_code=code_str,
        start_date=start_date, end_date=end_date,
        fields='ts_code,trade_date,open,high,low,close,vol,amount'
    )
    if df is None or df.empty:
        return {}
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)
    df.rename(columns={'vol': 'volume', 'amount': 'money'}, inplace=True)
    df['factor'] = 1.0
    result = {}
    for code, group in df.groupby('ts_code'):
        group = group.copy()
        group['code'] = code
        result[code] = group[['code', 'open', 'high', 'low', 'close', 'volume', 'money', 'factor']]
    return result


def safe_to_csv(df, csv_path, min_rows=MIN_SAFE_ROWS):
    """安全写入：如果本地已有更多行数据，则合并而非覆盖"""
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if len(existing) >= len(df):  # 已有≥新数据时合并，避免丢行
                combined = pd.concat([existing, df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
                combined.to_csv(csv_path)
                return len(combined)
        except Exception:
            pass
    df.to_csv(csv_path)
    return len(df)


def needs_update(ts_code):
    """检查股票是否需要更新"""
    csv_path = os.path.join(DATA_DIR, f"{ts_code}.csv")
    if not os.path.exists(csv_path):
        return True, 'full (no_file)'

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if df.empty:
            return True, 'full (empty_file)'

        last_date = df.index.max()
        days_behind = (datetime.today() - last_date).days

        if days_behind == 0:
            return False, f'skip (behind {days_behind}d)'
        return True, f'incremental (behind {days_behind}d)'
    except Exception:
        return True, 'read_error'


def update_one(pro, ts_code):
    """Update single stock, return (code, status, n_rows)"""
    needs, info = needs_update(ts_code)
    if not needs:
        return ts_code, 'skip', 0

    csv_path = os.path.join(DATA_DIR, f"{ts_code}.csv")
    end_date = datetime.today().strftime('%Y%m%d')

    if 'incremental' in info:
        # 只拉增量
        try:
            local_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except Exception:
            local_df = pd.DataFrame()
        if not local_df.empty:
            next_day = (local_df.index.max() + timedelta(days=1)).strftime('%Y%m%d')
            new_df = fetch_one_stock(pro, ts_code, next_day, end_date)
            if new_df is not None and not new_df.empty:
                combined = pd.concat([local_df, new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
                safe_to_csv(combined, csv_path)
                return ts_code, 'incremental', len(new_df)
            return ts_code, 'skip', 0
        # 本地文件无法读取或为空，回退到全量重拉
    else:
        # 全量重拉
        df = fetch_one_stock(pro, ts_code, START_DATE, end_date)
        if df is not None and not df.empty:
            safe_to_csv(df, csv_path)
            return ts_code, 'full', len(df)
        return ts_code, 'fail', 0


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'updated': [], 'batch': 0, 'stats': {'full': 0, 'incremental': 0, 'skip': 0, 'fail': 0}}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, ensure_ascii=False)


def get_stock_list(pro):
    cache_path = os.path.join(DATA_DIR, 'stable_stocks.csv')
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, dtype=str)
        print(f"股票列表(缓存): {len(df)} 只")
        return df['ts_code'].tolist()

    print("首次获取股票列表...")
    df = safe_call(pro.stock_basic, exchange='', list_status='L',
                   fields='ts_code,symbol,name,list_date')
    if df is None or df.empty:
        return []
    cutoff = (datetime.today() - timedelta(days=365)).strftime('%Y%m%d')
    df = df[df['list_date'] <= cutoff]
    df.to_csv(cache_path, index=False)
    print(f"稳定股票: {len(df)} 只")
    return df['ts_code'].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch', type=int, default=30,
                        help='每批处理的股票数（用于进度保存粒度，非API批次大小）')
    parser.add_argument('--api-batch', type=int, default=50,
                        help='每次Tushare API调用包含的股票数（默认50）')
    parser.add_argument('--api-sleep', type=int, default=5,
                        help='API批次间休息秒数（默认5）')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--token', type=str, default=None,
                        help='Tushare token；未传入时读取 TUSHARE_TOKEN 环境变量')
    parser.add_argument('--data-dir', type=str, default='data/tracking_raw',
                        help='日更数据目录（默认 data/tracking_raw，避免污染 data/raw 训练数据）')
    args = parser.parse_args()

    global DATA_DIR, PROGRESS_FILE
    DATA_DIR = args.data_dir
    PROGRESS_FILE = os.path.join(DATA_DIR, "_update_progress.json")
    os.makedirs(DATA_DIR, exist_ok=True)

    token = resolve_tushare_token(args.token)

    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()

    stocks = get_stock_list(pro)
    progress = load_progress()
    updated_set = set(progress['updated'])

    # 过滤已更新的股票
    remaining = [s for s in stocks if s not in updated_set]
    if args.test:
        remaining = remaining[:args.test]

    total = len(remaining)
    stats = progress['stats'].copy()

    if total == 0:
        print("全部股票已更新完成！")
        print(f"统计: 全量{stats['full']} | 增量{stats['incremental']} | 跳过{stats['skip']} | 失败{stats['fail']}")
        return

    print(f"待更新 {total} 只(共{len(stocks)} 只)")
    print(f"API批次大小: {args.api_batch} 只/调用, 批次间休息: {args.api_sleep}s")

    end_date = datetime.today().strftime('%Y%m%d')

    # 预处理：分类股票并决定每只需要拉取的日期范围
    tasks_full = []
    tasks_incremental = []
    skipped = 0
    for code in tqdm(remaining, desc="分类股票"):
        needs, info = needs_update(code)
        if not needs:
            skipped += 1
            stats['skip'] = stats.get('skip', 0) + 1
            continue
        if 'full' in info:
            tasks_full.append((code, START_DATE))
        else:
            csv_path = os.path.join(DATA_DIR, f"{code}.csv")
            local_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            next_day = (local_df.index.max() + timedelta(days=1)).strftime('%Y%m%d')
            tasks_incremental.append((code, next_day))
    stats['skip'] = stats.get('skip', 0) + skipped

    print(f"全量重拉: {len(tasks_full)} | 增量更新: {len(tasks_incremental)} | 跳过: {skipped}")

    api_batch_size = args.api_batch
    api_sleep = args.api_sleep

    def run_batches(task_list, batch_label):
        nonlocal updated_set
        task_map = {}  # date_range -> list of codes
        for code, start in task_list:
            task_map.setdefault(start, []).append(code)

        for start_date, codes in task_map.items():
            n_batches = (len(codes) + api_batch_size - 1) // api_batch_size
            for bi in range(n_batches):
                batch_codes = codes[bi * api_batch_size: (bi + 1) * api_batch_size]
                i = bi + 1
                print(f"\n{batch_label} {start_date}~{end_date}: "
                      f"批次 {i}/{n_batches} ({len(batch_codes)} 只)")

                result_map = fetch_batch_stocks(pro, batch_codes, start_date, end_date)
                for code in batch_codes:
                    if code in result_map and not result_map[code].empty:
                        csv_path = os.path.join(DATA_DIR, f"{code}.csv")
                        if start_date == START_DATE:
                            safe_to_csv(result_map[code], csv_path)
                            stats['full'] = stats.get('full', 0) + 1
                        else:
                            try:
                                local_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                            except Exception:
                                local_df = pd.DataFrame()
                            if not local_df.empty:
                                combined = pd.concat([local_df, result_map[code]])
                                combined = combined[~combined.index.duplicated(keep='last')]
                                combined.sort_index(inplace=True)
                                safe_to_csv(combined, csv_path)
                            else:
                                safe_to_csv(result_map[code], csv_path)
                            stats['incremental'] = stats.get('incremental', 0) + 1
                    else:
                        stats['fail'] = stats.get('fail', 0) + 1

                # 保存进度
                updated_set.update(batch_codes)
                progress['updated'] = list(updated_set)
                progress['stats'] = stats
                save_progress(progress)

                print(f"  全量:{stats.get('full', 0)} | 增量:{stats.get('incremental', 0)} | "
                      f"跳过:{stats.get('skip', 0)} | 失败:{stats.get('fail', 0)}")
                if i < n_batches:
                    time.sleep(api_sleep)

    # 先处理全量（如有），再处理增量
    if tasks_full:
        run_batches(tasks_full, "全量")
    if tasks_incremental:
        run_batches(tasks_incremental, "增量")

    print(f"\n{'=' * 55}")
    print("更新完成!")
    print(f"  全量下载: {stats.get('full', 0)} | 增量更新: {stats.get('incremental', 0)}")
    print(f"  跳过: {stats.get('skip', 0)} | 失败: {stats.get('fail', 0)}")

    # 清理进度文件
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断保存进度。重新运行将继续从未处理的股票开始。")
        sys.exit(0)
