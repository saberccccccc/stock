"""Diagnose what a negative filter removes.

Low-memory design:
- stream base/full-rerank JSONL rows;
- only inspect a base rank window;
- load per-stock raw CSVs on demand with a small LRU cache;
- write aggregate CSVs instead of large verbose logs.
"""

import argparse
import json
import sys
from collections import OrderedDict
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research_protocol import RESEARCH_END_DATE, assert_research_end_date


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose negative-filter removals")
    parser.add_argument("--base-alpha", required=True)
    parser.add_argument("--full-rerank-alpha", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--start-rank", type=int, default=30)
    parser.add_argument("--end-rank", type=int, default=100)
    parser.add_argument("--drop-n", type=int, default=3)
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument("--max-data-date", default="2026-05-18")
    return parser.parse_args()


def iter_rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row["date"] = pd.Timestamp(row["date"])
                yield row


class PriceCache:
    def __init__(self, data_dir, maxsize, max_data_date=RESEARCH_END_DATE):
        self.data_dir = Path(data_dir)
        self.maxsize = int(maxsize)
        self.max_data_date = assert_research_end_date(
            max_data_date,
            context="negative-filter diagnostics",
        )
        self.cache = OrderedDict()

    def get(self, code):
        code = str(code)
        if code in self.cache:
            self.cache.move_to_end(code)
            return self.cache[code]
        path = self.data_dir / f"{code}.csv"
        if not path.exists():
            self.cache[code] = None
            return None
        df = pd.read_csv(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").drop_duplicates("trade_date")
        df = df.loc[df["trade_date"] <= self.max_data_date].copy()
        for col in ["open", "high", "low", "close", "money", "volume"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.reset_index(drop=True)
        self.cache[code] = df
        self.cache.move_to_end(code)
        while len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
        return df


def safe_div(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return np.nan
    return float(a / b - 1.0)


def features_for(cache, code, date):
    df = cache.get(code)
    if df is None or df.empty:
        return None
    dates = df["trade_date"].to_numpy(dtype="datetime64[ns]")
    pos = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(date)), side="left"))
    if pos >= len(df) or pd.Timestamp(df.loc[pos, "trade_date"]) != pd.Timestamp(date):
        return None
    row = df.loc[pos]
    close = float(row["close"])
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    money = float(row["money"]) if "money" in row else np.nan
    prev_close = float(df.loc[pos - 1, "close"]) if pos >= 1 else np.nan

    def close_at(offset):
        i = pos + offset
        if 0 <= i < len(df):
            return float(df.loc[i, "close"])
        return np.nan

    def open_at(offset):
        i = pos + offset
        if 0 <= i < len(df):
            return float(df.loc[i, "open"])
        return np.nan

    win20 = df.loc[max(0, pos - 19) : pos, "close"].to_numpy(dtype=float)
    win_money = df.loc[max(0, pos - 19) : pos, "money"].to_numpy(dtype=float) if "money" in df else np.asarray([])
    min20 = np.nanmin(win20) if len(win20) else np.nan
    max20 = np.nanmax(win20) if len(win20) else np.nan
    pos20 = np.nan
    if np.isfinite(min20) and np.isfinite(max20) and max20 > min20:
        pos20 = float((close - min20) / (max20 - min20))

    open_t1 = open_at(1)
    return {
        "ret_1d": safe_div(close, prev_close),
        "ret_5d": safe_div(close, close_at(-5)),
        "ret_20d": safe_div(close, close_at(-20)),
        "intraday_ret": safe_div(close, open_),
        "amplitude": safe_div(high, low),
        "close_pos_20d": pos20,
        "money": money,
        "adv20_money": float(np.nanmean(win_money)) if len(win_money) else np.nan,
        "next_gap": safe_div(open_t1, close),
        "future_oo_1d": safe_div(open_at(2), open_t1),
        "future_oo_5d": safe_div(open_at(6), open_t1),
        "future_oo_10d": safe_div(open_at(11), open_t1),
    }


def summarize(frame):
    metrics = [
        "base_rank",
        "full_rank",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "intraday_ret",
        "amplitude",
        "close_pos_20d",
        "money",
        "adv20_money",
        "next_gap",
        "future_oo_1d",
        "future_oo_5d",
        "future_oo_10d",
    ]
    rows = []
    for group, sub in frame.groupby("group"):
        out = {"group": group, "n": len(sub)}
        for metric in metrics:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            out[f"{metric}_mean"] = float(vals.mean())
            out[f"{metric}_median"] = float(vals.median())
            out[f"{metric}_p25"] = float(vals.quantile(0.25))
            out[f"{metric}_p75"] = float(vals.quantile(0.75))
        rows.append(out)
    return pd.DataFrame(rows)


def iter_aligned_rows(base_path, rerank_path):
    """Yield matching rows and reject silent truncation or date misalignment."""

    missing = object()
    for base, rerank in zip_longest(iter_rows(base_path), iter_rows(rerank_path), fillvalue=missing):
        if base is missing or rerank is missing:
            raise ValueError("base and rerank alpha files have different row counts")
        if pd.Timestamp(base["date"]) != pd.Timestamp(rerank["date"]):
            raise ValueError(f"date mismatch: {base['date']} vs {rerank['date']}")
        if pd.Timestamp(base["date"]) > RESEARCH_END_DATE:
            raise ValueError(
                f"negative-filter diagnostics contain signal date {base['date'].date()} "
                f"after frozen research boundary {RESEARCH_END_DATE.date()}"
            )
        yield base, rerank


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = PriceCache(args.data_dir, args.cache_size, args.max_data_date)
    records = []
    missing = 0
    for base, rerank in iter_aligned_rows(args.base_alpha, args.full_rerank_alpha):
        date = pd.Timestamp(base["date"])
        codes = [str(code) for code in base["codes"]]
        rerank_codes = [str(code) for code in rerank["codes"]]
        rerank_pos = {code: i for i, code in enumerate(rerank_codes)}
        lo = min(max(0, int(args.start_rank)), len(codes))
        hi = min(int(args.end_rank), len(codes))
        window = codes[lo:hi]
        worst = sorted(
            window,
            key=lambda code: rerank_pos.get(code, len(codes) + codes.index(code)),
            reverse=True,
        )[: min(int(args.drop_n), len(window))]
        drop_set = set(worst)
        for code in window:
            feats = features_for(cache, code, date)
            if feats is None:
                missing += 1
                continue
            feats.update(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "code": code,
                    "group": "dropped" if code in drop_set else "kept_window",
                    "base_rank": int(codes.index(code)),
                    "full_rank": int(rerank_pos.get(code, len(codes))),
                }
            )
            records.append(feats)

    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / "negative_filter_diagnostics_rows.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(out_dir / "negative_filter_diagnostics_summary.csv", index=False)

    diff = {}
    if set(summary["group"]) >= {"dropped", "kept_window"}:
        s = summary.set_index("group")
        for col in summary.columns:
            if col not in ("group", "n"):
                diff[col] = float(s.loc["dropped", col] - s.loc["kept_window", col])
    pd.DataFrame([diff]).to_csv(out_dir / "negative_filter_diagnostics_diff.csv", index=False)
    print(
        f"rows={len(frame)} missing={missing} out={out_dir} "
        f"groups={frame['group'].value_counts().to_dict()}",
        flush=True,
    )
    print(summary[["group", "n", "ret_20d_mean", "close_pos_20d_mean", "future_oo_5d_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
