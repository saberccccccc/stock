"""鎵归噺涓嬭浇甯傚満鐗瑰緛鏁版嵁锛堝?鍩烘寚鏁般€佽?涓氭寚鏁帮級"""
import os
import pandas as pd

def download_index_data(symbol, name, output_dir="data/raw"):
    """
    涓嬭浇鍗曚釜鎸囨暟鏁版嵁

    Args:
        symbol: akshare 鎸囨暟浠ｇ爜
        name: 杈撳嚭鏂囦欢鍚嶏紙涓嶅惈.csv锛?        output_dir: 杈撳嚭鐩?綍
    """
    try:
        import akshare as ak
    except ImportError:
        print("閿欒?: 鏈?畨瑁卆kshare")
        return False

    output_path = os.path.join(output_dir, f"{name}.csv")

    try:
        print(f"涓嬭浇 {name}...")
        df = ak.stock_zh_index_daily(symbol=symbol)
        df.rename(columns={'date': 'date', 'open': 'open', 'high': 'high',
                          'low': 'low', 'close': 'close', 'volume': 'volume'},
                 inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        df.to_csv(output_path, index=False)
        print(f"  [OK] {len(df)} 鏉¤?褰曪紝{df['date'].min().strftime('%Y-%m-%d')} 鑷?{df['date'].max().strftime('%Y-%m-%d')}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def download_all_market_data():
    """Download all market characteristic data"""

    # 1. 瀹藉熀鎸囨暟
    print("\n=== 涓嬭浇瀹藉熀鎸囨暟 ===")
    indices = [
        ("sh000300", "hs300_index"),      # 娌?繁300锛堝凡鏈夛紝鍙?烦杩囷級
        ("sh000016", "sz50_index"),       # 涓婅瘉50
        ("sh000905", "zz500_index"),      # 涓?瘉500
        ("sz399006", "cyb_index"),        # 鍒涗笟鏉挎寚
    ]

    for symbol, name in indices:
        download_index_data(symbol, name)

    # 2. 琛屼笟鎸囨暟锛堢敵涓囦竴绾э級
    print("\n=== 涓嬭浇鐢充竾涓€绾ц?涓氭寚鏁?===")
    print("鎻愮ず: 琛屼笟鎸囨暟鏁版嵁閲忚緝澶э紝鍙?兘闇€瑕佸嚑鍒嗛挓...")

    try:
        import akshare as ak
        # 鑾峰彇鐢充竾琛屼笟鍒嗙被
        sw_index = ak.sw_index_first_info()
        print(f"  total {len(sw_index)} first-level industries")

        output_dir = "data/raw/sw_industry"
        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        skip_count = 0
        for idx, row in sw_index.iterrows():
            code = row['琛屼笟浠ｇ爜'].replace('.SI', '')  # 鍘绘帀鍚庣紑
            name = row['琛屼笟鍚嶇О']
            output_path = os.path.join(output_dir, f"{code}_{name}.csv")

            # 璺宠繃宸蹭笅杞界殑
            if os.path.exists(output_path):
                skip_count += 1
                continue

            try:
                print(f"  涓嬭浇 {name}...")
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

    print("\n=== 涓嬭浇瀹屾垚 ===")


if __name__ == "__main__":
    download_all_market_data()
