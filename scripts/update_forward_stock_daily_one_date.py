"""Append one trading day's A-share OHLCV rows to forward_raw CSV files.

This is intentionally narrower than data/update_daily.py: for a known missing
trading date it uses one Tushare `daily(trade_date=...)` call instead of
looping through thousands of stocks.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from data.api_utils import SafeAPICaller, resolve_tushare_token


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/forward_raw")
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--token", default=None)
    parser.add_argument("--min-existing-rows", type=int, default=200)
    return parser.parse_args()


def normalize_trade_date(value):
    return pd.Timestamp(value).strftime("%Y%m%d")


def fetch_daily_frame(token, trade_date):
    import tushare as ts

    ts.set_token(token)
    pro = ts.pro_api()
    caller = SafeAPICaller(
        min_interval=1.0,
        max_retries=3,
        retry_base_delay=4.0,
        jitter=(0.2, 0.5),
        data_source="tushare",
    )
    frame = caller(
        pro.daily,
        trade_date=trade_date,
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.rename(columns={"ts_code": "code", "vol": "volume", "amount": "money"})
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["factor"] = 1.0
    return frame[["trade_date", "code", "open", "high", "low", "close", "volume", "money", "factor"]]


def append_one_code(path, row, min_existing_rows):
    if path.exists():
        existing = pd.read_csv(path, parse_dates=["trade_date"])
        if len(existing) < min_existing_rows:
            return False
        combined = pd.concat([existing, row.to_frame().T], ignore_index=True)
    else:
        combined = row.to_frame().T
    combined = combined.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    combined.to_csv(path, index=False)
    return True


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    trade_date = normalize_trade_date(args.trade_date)
    token = resolve_tushare_token(args.token)

    frame = fetch_daily_frame(token, trade_date)
    if frame.empty:
        raise ValueError(f"Tushare returned no daily rows for {trade_date}")

    updated = 0
    skipped_missing_local = 0
    skipped_short_local = 0
    for _, row in frame.iterrows():
        code = str(row["code"])
        path = data_dir / f"{code}.csv"
        if not path.exists():
            skipped_missing_local += 1
            continue
        if append_one_code(path, row, args.min_existing_rows):
            updated += 1
        else:
            skipped_short_local += 1

    print(
        f"updated {updated} files for {trade_date}; "
        f"skipped_missing_local={skipped_missing_local}; "
        f"skipped_short_local={skipped_short_local}",
        flush=True,
    )


if __name__ == "__main__":
    main()
