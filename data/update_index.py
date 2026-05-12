"""Update HS300 index data to latest date"""
import os
import pandas as pd
import tushare as ts

def update_hs300_index(token=None, output_path="data/raw/hs300_index.csv"):
    """
    从Tushare鑾峰彇娌?繁300指数日线数据并更新到CSV

    Args:
        token: Tushare token锛屽?果为None鍒欎粠鐜??变量读取
        output_path: 输出CSV璺?緞
    """
    if token is None:
        token = os.environ.get('TUSHARE_TOKEN')

    if not token:
        print("閿欒?: 鏈?壘鍒癟USHARE_TOKEN锛岃?璁剧疆鐜??鍙橀噺鎴栦紶鍏?oken参数")
        return False

    print("连接Tushare API...")
    pro = ts.pro_api(token)

    # 读取现有数据
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        existing['date'] = pd.to_datetime(existing['date'])
        last_date = existing['date'].max()
        print(f"鐜版湁鏁版嵁鏈EUR鍚庢棩鏈? {last_date.strftime('%Y-%m-%d')}")
        start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
        print("no existing data, fetching full history")
        print("no existing data, fetching full history")
        start_date = '20100101'
        existing = None

    # 鑾峰彇鏂版暟鎹?    print(f"获取 {start_date} 鑷充粖鐨勬暟鎹?..")
    try:
        new_data = pro.index_daily(
            ts_code='000300.SH',
            start_date=start_date,
            fields='trade_date,open,high,low,close,vol'
        )
    except Exception as e:
        print(f"API调用失败: {e}")
        return False

    if new_data.empty:
        print("娌℃湁鏂版暟鎹?渶瑕佹洿鏂?")
        return True

    # 鏍煎紡杞?崲
    new_data.rename(columns={
        'trade_date': 'date',
        'vol': 'volume'
    }, inplace=True)
    new_data['date'] = pd.to_datetime(new_data['date'])
    new_data = new_data.sort_values('date')

    # 合并数据
    if existing is not None:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date')
    else:
        combined = new_data

    # 保存
    combined.to_csv(output_path, index=False)
    print(f"鉁?宸叉洿鏂?{len(new_data)} 条新数据")
    print(f"鉁?数据范围: {combined['date'].min().strftime('%Y-%m-%d')} 鑷?{combined['date'].max().strftime('%Y-%m-%d')}")
    print(f"鉁?鎬昏? {len(combined)} 鏉¤?褰?")

    return True


if __name__ == "__main__":
    success = update_hs300_index()
    if not success:
        exit(1)
