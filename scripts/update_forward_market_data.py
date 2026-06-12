"""Update broad and Shenwan industry indices for forward inference."""

import argparse
import os
import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from data.market_features import SW_INDUSTRIES


BROAD_INDICES = {
    "hs300_index.csv": "sh000300",
    "sz50_index.csv": "sh000016",
    "zz500_index.csv": "sh000905",
    "cyb_index.csv": "sz399006",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Update forward market index files")
    parser.add_argument("--data-dir", default="data/forward_raw")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)
    return parser.parse_args()


def retry_fetch(func, retries, label):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            result = func()
            if result is None or result.empty:
                raise ValueError("empty response")
            return result
        except Exception as exc:
            last_error = exc
            print(f"{label}: attempt {attempt}/{retries} failed: {exc}")
            time.sleep(attempt * 2)
    raise RuntimeError(f"{label}: update failed after {retries} attempts: {last_error}")


def update_broad_indices(data_dir, retries):
    for filename, symbol in BROAD_INDICES.items():
        df = retry_fetch(
            lambda symbol=symbol: ak.stock_zh_index_daily(symbol=symbol),
            retries,
            symbol,
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        path = data_dir / filename
        df.to_csv(path)
        print(f"{filename}: {df.index.min().date()} ~ {df.index.max().date()} ({len(df)})")


def update_industry_indices(data_dir, retries, sleep_seconds):
    output_dir = data_dir / "sw_industry"
    output_dir.mkdir(parents=True, exist_ok=True)
    rename = {
        "代码": "code",
        "日期": "date",
        "收盘": "close",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "money",
    }
    for code, name in SW_INDUSTRIES:
        df = retry_fetch(
            lambda code=code: ak.index_hist_sw(symbol=code, period="day"),
            retries,
            code,
        ).rename(columns=rename)
        required = ["code", "date", "close", "open", "high", "low", "volume", "money"]
        df = df[required].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        path = output_dir / f"{code}_{name}.csv"
        df.to_csv(path, index=False)
        print(f"{code} {name}: {df['date'].min().date()} ~ {df['date'].max().date()} ({len(df)})")
        time.sleep(max(sleep_seconds, 0.0))


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    update_broad_indices(data_dir, args.retries)
    update_industry_indices(data_dir, args.retries, args.sleep)


if __name__ == "__main__":
    main()
