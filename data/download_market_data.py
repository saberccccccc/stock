"""鎵归噺涓嬭浇甯傚満鐗瑰緛鏁版嵁锛堝?鍩烘寚鏁般€佽?业指数）"""
import os
import pandas as pd

def download_index_data(symbol, name, output_dir="data/raw"):
    """
    下载单个指数数据

    Args:
        symbol: akshare 指数代码
        name: 输出文件名（不含.csv锛?        output_dir: 杈撳嚭鐩?綍
    """
    try:
        import akshare as ak
    except ImportError:
        print("閿欒?: 鏈?畨瑁卆kshare")
        return False

    output_path = os.path.join(output_dir, f"{name}.csv")

    try:
        print(f"下载 {name}...")
        df = ak.stock_zh_index_daily(symbol=symbol)
        df.rename(columns={'date': 'date', 'open': 'open', 'high': 'high',
                          'low': 'low', 'close': 'close', 'volume': 'volume'},
                 inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        df.to_csv(output_path, index=False)
        print(f"  [OK] {len(df)} 鏉¤?录，{df['date'].min().strftime('%Y-%m-%d')} 鑷?{df['date'].max().strftime('%Y-%m-%d')}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def download_all_market_data():
    """Download all market characteristic data"""

    # 1. 宽基指数
    print("\n=== 下载宽基指数 ===")
    indices = [
        ("sh000300", "hs300_index"),      # 娌?繁300锛堝凡鏈夛紝鍙?烦杩囷級
        ("sh000016", "sz50_index"),       # 上证50
        ("sh000905", "zz500_index"),      # 涓?瘉500
        ("sz399006", "cyb_index"),        # 创业板指
    ]

    for symbol, name in indices:
        download_index_data(symbol, name)

    # 2. 琛屼笟鎸囨暟锛堢敵涓囦竴绾э級
    print("\n=== 涓嬭浇鐢充竾涓€绾ц?涓氭寚鏁?===")
    print("提示: 琛屼笟鎸囨暟鏁版嵁閲忚緝澶э紝鍙?兘闇€瑕佸嚑鍒嗛挓...")

    try:
        import akshare as ak
        # 获取申万行业分类
        sw_index = ak.sw_index_first_info()
        print(f"  total {len(sw_index)} first-level industries")

        output_dir = "data/raw/sw_industry"
        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        skip_count = 0
        for idx, row in sw_index.iterrows():
            code = row['行业代码'].replace('.SI', '')  # 去掉后缀
            name = row['琛屼笟鍚嶇О']
            output_path = os.path.join(output_dir, f"{code}_{name}.csv")

            # 跳过已下载的
            if os.path.exists(output_path):
                skip_count += 1
                continue

            try:
                print(f"  下载 {name}...")
                df = ak.index_hist_sw(symbol=code)
                if not df.empty:
                    df.rename(columns={'date': 'date', 'open': 'open', 'high': 'high',
                                      'low': 'low', 'close': 'close', 'volume': 'volume'},
                             inplace=True)
                    df['date'] = pd.to_datetime(df['date'])
                    df.to_csv(output_path, index=False)
                    success_count += 1
                    print(f"    [OK] {len(df)} records")
            except Exception as e:
                print(f"    [FAIL] {e}")
                continue
        print(f"  [OK] downloaded {success_count} industry indices")
        print(f"  [OK] downloaded {success_count} industry indices")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n=== 下载完成 ===")


if __name__ == "__main__":
    download_all_market_data()
