"""Download validated daily bars directly into MarketDailyStore."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.api_utils import resolve_tushare_token
from data.market_daily_update import (
    AkshareBroadIndexClient,
    CompositeMarketDailyClient,
    TushareMarketDailyClient,
    run_incremental_update,
)


def _project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _date_key(value: str) -> str:
    import pandas as pd

    return pd.Timestamp(value).strftime("%Y%m%d")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    dates = parser.add_mutually_exclusive_group(required=True)
    dates.add_argument("--trade-date")
    dates.add_argument("--start-date")
    dates.add_argument("--open-dates")
    parser.add_argument("--end-date")
    parser.add_argument("--token")
    parser.add_argument("--progress")
    parser.add_argument("--min-equity-rows", type=int, default=3000)
    parser.add_argument("--equity-only", action="store_true")
    parser.add_argument(
        "--index-source",
        choices=("akshare", "tushare"),
        default="akshare",
        help="AkShare avoids low-tier Tushare index_daily per-code rate limits.",
    )
    parser.add_argument("--allow-revision", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.start_date) != bool(args.end_date):
        parser.error("--start-date and --end-date must be provided together")
    return args


def main(argv=None):
    args = parse_args(argv)
    equity_client = TushareMarketDailyClient(resolve_tushare_token(args.token))
    if args.equity_only or args.index_source == "tushare":
        index_client = equity_client
    else:
        index_client = AkshareBroadIndexClient()
    client = CompositeMarketDailyClient(equity_client, index_client)
    if args.trade_date:
        dates = [_date_key(args.trade_date)]
    elif args.open_dates:
        dates = sorted({_date_key(value) for value in args.open_dates.split(",") if value.strip()})
    else:
        start = _date_key(args.start_date)
        end = _date_key(args.end_date)
        if end < start:
            raise ValueError("end date must be on or after start date")
        dates = client.trade_dates(start, end)
    store_root = _project_path(args.store_root)
    progress = (
        _project_path(args.progress)
        if args.progress
        else store_root / "update_manifests" / f"market_daily_{dates[0]}_{dates[-1]}.json"
    )
    result = run_incremental_update(
        client=client,
        store_root=store_root,
        dates=dates,
        progress_path=progress,
        min_equity_rows=args.min_equity_rows,
        include_indices=not args.equity_only,
        allow_revision=args.allow_revision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
