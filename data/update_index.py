"""Update HS300 index data to latest date"""
import os
import pandas as pd
import tushare as ts

def update_hs300_index(token=None, output_path="data/raw/hs300_index.csv"):
    """
    从Tushare获取沪深300指数日线数据并更新到CSV

    Args:
        token: Tushare token，如果为None则从环境变量读取
        output_path: 输出CSV路径
    """
    if token is None:
        token = os.environ.get('TUSHARE_TOKEN')

    if not token:
        print("错误: 未找到TUSHARE_TOKEN，请设置环境变量或传入token参数")
        return False

    print("连接Tushare API...")
    pro = ts.pro_api(token)

    # 读取现有数据
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        existing['date'] = pd.to_datetime(existing['date'])
        last_date = existing['date'].max()
        print(f"现有数据最后日期: {last_date.strftime('%Y-%m-%d')}")
        start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
    else:
        print("无现有数据，获取全量历史")
        start_date = '20100101'
        existing = None

    # 获取新数据
    print(f"获取 {start_date} 至今的数据...")
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
        print("没有新数据需要更新")
        return True

    # 格式转换
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
    print(f"[OK] 已更新 {len(new_data)} 条新数据")
    print(f"[OK] 数据范围: {combined['date'].min().strftime('%Y-%m-%d')} 至 {combined['date'].max().strftime('%Y-%m-%d')}")
    print(f"[OK] 总计 {len(combined)} 条记录")

    return True


if __name__ == "__main__":
    success = update_hs300_index()
    if not success:
        exit(1)
