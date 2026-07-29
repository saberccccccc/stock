"""Efficiently append one date or a date range to forward A-share daily CSVs."""

import argparse
import msvcrt
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.api_utils import SafeAPICaller, resolve_tushare_token
from data.forward_daily_update import DailyFileWriter, load_progress, write_progress


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/forward_raw")
    dates = parser.add_mutually_exclusive_group(required=True)
    dates.add_argument("--trade-date", help="One date in YYYYMMDD or YYYY-MM-DD form")
    dates.add_argument("--start-date", help="First date of a resumable range")
    dates.add_argument(
        "--open-dates",
        help="Comma-separated known exchange-open dates; bypasses trade_cal",
    )
    parser.add_argument("--end-date", help="Last date; required with --start-date")
    parser.add_argument("--token", default=None)
    parser.add_argument("--min-existing-rows", type=int, default=1)
    parser.add_argument("--min-market-rows", type=int, default=3000)
    parser.add_argument("--progress", default=None)
    parser.add_argument("--codes-file", default=None, help="Optional newline-delimited code subset")
    parser.add_argument(
        "--max-existing-rows",
        type=int,
        default=None,
        help="Update only local files with at most this many existing data rows",
    )
    args = parser.parse_args(argv)
    if bool(args.start_date) != bool(args.end_date):
        parser.error("--start-date and --end-date must be provided together")
    return args


def normalize_trade_date(value):
    return pd.Timestamp(value).strftime("%Y%m%d")


class TushareDailyClient:
    def __init__(self, token):
        import tushare as ts

        ts.set_token(token)
        self.pro = ts.pro_api()
        self.caller = SafeAPICaller(
            min_interval=1.0,
            max_retries=3,
            retry_base_delay=4.0,
            jitter=(0.2, 0.5),
            data_source="tushare",
            non_retryable_markers=("频率超限", "权限"),
        )

    def trade_dates(self, start_date, end_date):
        frame = self.caller(
            self.pro.trade_cal,
            exchange="SSE",
            start_date=start_date,
            end_date=end_date,
            is_open="1",
            fields="cal_date,is_open",
        )
        if frame is None or frame.empty:
            raise ValueError(f"Tushare returned no open dates for {start_date}..{end_date}")
        return sorted(frame["cal_date"].astype(str).tolist())

    def daily(self, trade_date):
        frame = self.caller(
            self.pro.daily,
            trade_date=trade_date,
            fields="ts_code,trade_date,open,high,low,close,vol,amount",
        )
        if frame is None or frame.empty:
            return pd.DataFrame()
        frame = frame.rename(columns={"ts_code": "code", "vol": "volume", "amount": "money"})
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["factor"] = 1.0
        return frame[
            ["trade_date", "code", "open", "high", "low", "close", "volume", "money", "factor"]
        ]


def resolve_requested_dates(args, client):
    if args.trade_date:
        date = normalize_trade_date(args.trade_date)
        return date, date, [date]
    if args.open_dates:
        dates = sorted(
            {normalize_trade_date(value) for value in args.open_dates.split(",") if value.strip()}
        )
        if not dates:
            raise ValueError("--open-dates is empty")
        return dates[0], dates[-1], dates
    start = normalize_trade_date(args.start_date)
    end = normalize_trade_date(args.end_date)
    if end < start:
        raise ValueError("end date must be on or after start date")
    return start, end, client.trade_dates(start, end)


def read_code_filter(path):
    if not path:
        return None
    values = {
        line.strip().replace(".csv", "")
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not values:
        raise ValueError("codes file is empty")
    return values


def resolve_project_path(path):
    if not path:
        return None
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def codes_with_at_most_rows(data_dir, maximum):
    if maximum is None:
        return None
    selected = set()
    for path in data_dir.glob("*.csv"):
        code = path.stem
        if "." not in code or not code.split(".", 1)[0].isdigit():
            continue
        with path.open("rb") as handle:
            lines = 0
            for line in handle:
                if line.strip():
                    lines += 1
                if lines > maximum + 1:
                    break
        if max(lines - 1, 0) <= maximum:
            selected.add(code)
    return selected


def update_one_date(client, writer, trade_date, args, code_filter=None):
    frame = client.daily(trade_date)
    if len(frame) < args.min_market_rows:
        raise ValueError(
            f"Tushare returned only {len(frame)} daily rows for {trade_date}; "
            f"minimum is {args.min_market_rows}"
        )
    if code_filter is not None:
        frame = frame.loc[frame["code"].astype(str).isin(code_filter)]
    counts = Counter()
    for _, row in frame.iterrows():
        result = writer.write(row)
        counts[result.status] += 1
    return {"selected_rows": int(len(frame)), **dict(sorted(counts.items()))}


@contextmanager
def progress_lock(progress_path):
    lock_path = Path(str(progress_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"another updater is using {progress_path}") from exc
    try:
        yield
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def run_update(args, client, writer, code_filter, progress_path, start, end):
    with progress_lock(progress_path):
        progress = load_progress(progress_path, start_date=start, end_date=end)
        dates = progress.get("requested_dates")
        if not dates:
            _, _, dates = resolve_requested_dates(args, client)
            progress["requested_dates"] = dates
            write_progress(progress_path, progress)
        completed = set(progress["completed_dates"])
        for trade_date in dates:
            if trade_date in completed:
                print(f"{trade_date}: already completed", flush=True)
                continue
            metrics = update_one_date(client, writer, trade_date, args, code_filter)
            progress["dates"][trade_date] = metrics
            progress["completed_dates"].append(trade_date)
            progress["completed_dates"].sort()
            write_progress(progress_path, progress)
            print(f"{trade_date}: {metrics}", flush=True)
        progress["status"] = "completed"
        write_progress(progress_path, progress)
        print(f"completed {len(dates)} trading dates; progress={progress_path}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    data_dir = (ROOT / args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    token = resolve_tushare_token(args.token)
    client = TushareDailyClient(token)
    writer = DailyFileWriter(data_dir, min_existing_rows=args.min_existing_rows)
    code_filter = read_code_filter(resolve_project_path(args.codes_file))
    short_filter = codes_with_at_most_rows(data_dir, args.max_existing_rows)
    if code_filter is None:
        code_filter = short_filter
    elif short_filter is not None:
        code_filter &= short_filter
    if args.start_date:
        start = normalize_trade_date(args.start_date)
        end = normalize_trade_date(args.end_date)
    elif args.trade_date:
        start = end = normalize_trade_date(args.trade_date)
    else:
        explicit = sorted(
            {normalize_trade_date(value) for value in args.open_dates.split(",") if value.strip()}
        )
        start, end = explicit[0], explicit[-1]
    progress_path = (
        resolve_project_path(args.progress)
        if args.progress
        else data_dir / "update_manifests" / f"forward_daily_{start}_{end}.json"
    )
    run_update(args, client, writer, code_filter, progress_path, start, end)


if __name__ == "__main__":
    main()
