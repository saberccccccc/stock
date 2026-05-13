# update_daily_data.py - 个股日线智能增量更新（断点续传版）
# 用法: python update_daily_data.py [--workers 2]
# 可随时 Ctrl+C 中断，重新运行会自动跳过已更新的股票

import os, sys, time, random, argparse, json
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_DIR = "data/raw"
START_DATE = "20100101"
MIN_INTERVAL = 1.5       # API最小间隔（秒）
BATCH_SLEEP = 60         # 批次间休息（秒）

PROGRESS_FILE = os.path.join(DATA_DIR, "_update_progress.json")
os.makedirs(DATA_DIR, exist_ok=True)


def resolve_tushare_token(token=None):
    resolved = token or os.getenv('TUSHARE_TOKEN')
    if not resolved:
        raise ValueError('缺少 TUSHARE_TOKEN，请设置环境变量或通过 --token 传入')
    return resolved

# ==================== 安全限流调用 ====================
_last_request_time = 0

def safe_call(func, *args, **kwargs):
    global _last_request_time
    for attempt in range(3):
        try:
            elapsed = time.time() - _last_request_time
            if elapsed < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - elapsed)
            _last_request_time = time.time()
            time.sleep(random.uniform(0.2, 0.4))
            return func(*args, **kwargs)
        except Exception as e:
            wait = 4 * (attempt + 1)
            print(f"  重试{attempt+1}/3: {e}, 等待{wait}s")
            time.sleep(wait)
    return None


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


def needs_update(ts_code):
    """检查股票是否需要更新（2010~2025的数据不满足要求）"""
    csv_path = os.path.join(DATA_DIR, f"{ts_code}.csv")
    if not os.path.exists(csv_path):
        return True, 'no_file'

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if df.empty:
            return True, 'empty_file'

        first_date = df.index.min()
        last_date = df.index.max()
        days_behind = (datetime.today() - last_date).days

        # 已有2010数据 + 最新 -> 跳过
        if first_date.year <= 2010 and days_behind <= 2:
            return False, f'skip ({first_date.date()}~{last_date.date()})'
        # 已有2010数据但落后 -> 增量
        if first_date.year <= 2010:
            return True, f'incremental (behind {days_behind}d)'
        # 只有2017+数据 -> 全量
        return True, f'full ({first_date.date()}~{last_date.date()})'
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
        local_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        next_day = (local_df.index.max() + timedelta(days=1)).strftime('%Y%m%d')
        new_df = fetch_one_stock(pro, ts_code, next_day, end_date)
        if new_df is not None and not new_df.empty:
            combined = pd.concat([local_df, new_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            combined.to_csv(csv_path)
            return ts_code, 'incremental', len(new_df)
        return ts_code, 'skip', 0
    else:
        # 全量重拉
        df = fetch_one_stock(pro, ts_code, START_DATE, end_date)
        if df is not None and not df.empty:
            df.to_csv(csv_path)
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
    parser.add_argument('--batch', type=int, default=30)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--token', type=str, default=None,
                        help='Tushare token；未传入时读取 TUSHARE_TOKEN 环境变量')
    args = parser.parse_args()
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
    start_batch = progress.get('batch', 0)

    if total == 0:
        print("全部股票已更新完成！")
        print(f"统计: 全量{stats['full']} | 增量{stats['incremental']} | 跳过{stats['skip']} | 失败{stats['fail']}")
        return

    print(f"待更新 {total} 只(共{len(stocks)} 只")
    batch_count = (total + args.batch - 1) // args.batch

    for bi in range(batch_count):
        batch = remaining[bi * args.batch: (bi + 1) * args.batch]
        print(f"\n{'=' * 55}")
        print(f"批次 {start_batch + bi + 1}/{start_batch + batch_count} | "
              f"{bi * args.batch + 1}-{min((bi + 1) * args.batch, total)} / {total}")

        batch_updated = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(update_one, pro, code): code for code in batch}
            for future in tqdm(as_completed(futures), total=len(batch), desc=f"进度"):
                try:
                    code, status, n_rows = future.result()
                    stats[status] = stats.get(status, 0) + 1
                    if status != 'fail':
                        batch_updated.append(code)
                except Exception as e:
                    print(f"  异常: {e}")
                    stats['fail'] += 1

        # 保存进度
        updated_set.update(batch_updated)
        progress['updated'] = list(updated_set)
        progress['batch'] = start_batch + bi + 1
        progress['stats'] = stats
        save_progress(progress)

        print(f"  全量:{stats['full']} | 增量:{stats['incremental']} | "
              f"跳过:{stats['skip']} | 失败:{stats['fail']}")

        if bi < batch_count - 1:
            print(f"  休息 {BATCH_SLEEP}s ...")
            time.sleep(BATCH_SLEEP)

    print(f"\n{'=' * 55}")
    print("更新完成!")
    print(f"  全量下载: {stats['full']} | 增量更新: {stats['incremental']}")
    print(f"  跳过: {stats['skip']} | 失败: {stats['fail']}")

    # 清理进度文件
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断保存进度。重新运行将继续从未处理的股票开始。")
        sys.exit(0)
