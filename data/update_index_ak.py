"""Use akshare to update HS300 index data (free, no token needed)"""
import os
import pandas as pd

def update_hs300_index_ak(output_path="data/raw/hs300_index.csv"):
    """
    从akshare鑾峰彇娌?繁300指数日线数据并更新到CSV

    Args:
        output_path: 输出CSV璺?緞
    """
    try:
        import akshare as ak
    except ImportError:
        print("閿欒?: 鏈?畨瑁卆kshare锛岃?杩愯?: pip install akshare")
        return False

    # 读取现有数据
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        existing['date'] = pd.to_datetime(existing['date'])
        last_date = existing['date'].max()
        print(f"鐜版湁鏁版嵁鏈EUR鍚庢棩鏈? {last_date.strftime('%Y-%m-%d')}")
        print("no existing data, fetching full history")
        print("no existing data, fetching full history")
        existing = None
        last_date = None

    # 鑾峰彇娌?繁300指数数据
    print("从akshare鑾峰彇娌?繁300指数数据...")
    try:
        # akshare 返回全部历史数据
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df.rename(columns={'date': 'date', 'open': 'open', 'high': 'high',
                          'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    except Exception as e:
        print(f"获取数据失败: {e}")
        return False

    # 濡傛灉鏈夌幇鏈夋暟鎹?紝鍙?繚鐣欐柊鏁版嵁
    if last_date is not None:
        new_data = df[df['date'] > last_date]
        if new_data.empty:
            print("娌℃湁鏂版暟鎹?渶瑕佹洿鏂?")
            return True
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date')
        print(f"[OK] 宸叉洿鏂?{len(new_data)} 条新数据")
    else:
        combined = df
        print(f"[OK] 获取全部历史数据")

    # 保存
    combined.to_csv(output_path, index=False)
    print(f"[OK] 数据范围: {combined['date'].min().strftime('%Y-%m-%d')} 鑷?{combined['date'].max().strftime('%Y-%m-%d')}")
    print(f"[OK] 鎬昏? {len(combined)} 鏉¤?褰?")

    return True


if __name__ == "__main__":
    success = update_hs300_index_ak()
    if not success:
        exit(1)
