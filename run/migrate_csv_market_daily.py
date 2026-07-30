"""Migrate one calendar month or year of per-stock CSVs into MarketDailyStore."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.market_daily_migration import migrate_csv_month


def _project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="data/forward_raw")
    parser.add_argument("--store-root", default="data/market_daily")
    period = parser.add_mutually_exclusive_group(required=True)
    period.add_argument("--month", help="Calendar month in YYYY-MM form")
    period.add_argument("--year", help="Calendar year in YYYY form")
    parser.add_argument("--source", default="legacy_forward_csv")
    parser.add_argument("--progress", default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--report", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    period = args.month or args.year
    payload = migrate_csv_month(
        source_root=_project_path(args.source_root),
        store_root=_project_path(args.store_root),
        month=period,
        source=args.source,
        progress_path=_project_path(args.progress) if args.progress else None,
        progress_every=args.progress_every,
    )
    if args.report:
        path = _project_path(args.report)
        _write_json_atomic(path, payload)
        print(f"wrote {path}", flush=True)
    print(json.dumps(payload["audit"], ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
