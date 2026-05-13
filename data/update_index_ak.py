"""Use akshare to update HS300 index data (free, no token needed)"""
import os
import pandas as pd

def update_hs300_index_ak(output_path="data/raw/hs300_index.csv"):
    """
    从akshare获取沪深300指数日线数据并更新到CSV

    Args:
        output_path: 输出CSV路径
    """
    try:
        import akshare as ak
    except ImportError:
        print("错误: 未安装akshare，请运行: pip install akshare")
        return False

    # 读取现有数据
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        existing['date'] = pd.to_datetime(existing['date'])
        last_date = existing['date'].max()
        print(f"现有数据最后日期: {last_date.strftime('%Y-%m-%d')}")
    else:
        print("无现有数据，获取全量历史")
        existing = None
        last_date = None

    # 获取沪深300指数数据
    print("从akshare获取沪深300指数数据...")
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

    # 如果有现有数据，只保留新数据
    if last_date is not None:
        new_data = df[df['date'] > last_date]
        if new_data.empty:
            print("没有新数据需要更新")
            return True
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date')
        print(f"[OK] 已更新 {len(new_data)} 条新数据")
    else:
        combined = df
        print(f"[OK] 获取全部历史数据")

    # 保存
    combined.to_csv(output_path, index=False)
    print(f"[OK] 数据范围: {combined['date'].min().strftime('%Y-%m-%d')} 至 {combined['date'].max().strftime('%Y-%m-%d')}")
    print(f"[OK] 总计 {len(combined)} 条记录")

    return True


if __name__ == "__main__":
    success = update_hs300_index_ak()
    if not success:
        exit(1)
