# -*- coding: utf-8 -*-
"""Universe filter diagnostics.

The functions here only report tradability/universe quality. They do not change
training samples or backtest semantics until explicitly wired into those paths.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def _stock_csvs(data_dir):
    data_dir = Path(data_dir)
    return sorted(p for p in data_dir.glob("*.csv") if p.name[0].isdigit())


def _load_names(data_dir):
    path = Path(data_dir) / "stable_stocks.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return {}
    if "ts_code" not in df.columns or "name" not in df.columns:
        return {}
    return dict(zip(df["ts_code"].astype(str), df["name"].fillna("").astype(str)))


def _read_ohlcv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    df.columns = df.columns.str.strip().str.lower()
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date"]).set_index("trade_date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
    return df.sort_index()


def stock_universe_diagnostics(data_dir="data/raw", min_listing_days=180, limit_pct=0.098):
    """Return per-stock universe diagnostics as a DataFrame."""
    names = _load_names(data_dir)
    rows = []
    for path in _stock_csvs(data_dir):
        code = path.stem
        df = _read_ohlcv(path)
        name = names.get(code, "")
        row = {
            "code": code,
            "name": name,
            "rows": len(df),
            "start": pd.NaT,
            "end": pd.NaT,
            "is_st": ("ST" in name.upper()) if name else False,
            "short_listing": True,
            "zero_volume_days": 0,
            "zero_volume_ratio": 0.0,
            "limit_up_days": 0,
            "limit_down_days": 0,
            "limit_day_ratio": 0.0,
        }
        if df.empty:
            rows.append(row)
            continue
        row["start"] = df.index.min()
        row["end"] = df.index.max()
        row["short_listing"] = len(df) < min_listing_days

        volume = pd.to_numeric(df.get("volume", pd.Series(index=df.index, dtype=float)), errors="coerce")
        row["zero_volume_days"] = int((volume <= 0).sum())
        row["zero_volume_ratio"] = float((volume <= 0).mean()) if len(volume) else 0.0

        close = pd.to_numeric(df.get("close", pd.Series(index=df.index, dtype=float)), errors="coerce")
        ret = close.pct_change()
        limit_up = ret >= limit_pct
        limit_down = ret <= -limit_pct
        row["limit_up_days"] = int(limit_up.sum())
        row["limit_down_days"] = int(limit_down.sum())
        row["limit_day_ratio"] = float((limit_up | limit_down).mean()) if len(ret) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_universe_diagnostics(df):
    if df.empty:
        return {
            "stocks": 0,
            "st": 0,
            "short_listing": 0,
            "zero_volume_any": 0,
            "high_limit_day_ratio": 0,
        }
    return {
        "stocks": int(len(df)),
        "st": int(df["is_st"].sum()),
        "short_listing": int(df["short_listing"].sum()),
        "zero_volume_any": int((df["zero_volume_days"] > 0).sum()),
        "high_limit_day_ratio": int((df["limit_day_ratio"] > 0.05).sum()),
    }


def print_universe_report(df):
    summary = summarize_universe_diagnostics(df)
    print("\n== Universe filter diagnostics ==")
    for key, val in summary.items():
        pct = val / max(summary["stocks"], 1) if key != "stocks" else 1.0
        if key == "stocks":
            print(f"{key}: {val}")
        else:
            print(f"{key}: {val} ({pct * 100:.2f}%)")
    if not df.empty:
        latest = df["end"].max()
        stale = df[df["end"] < latest - pd.Timedelta(days=10)]
        print(f"stale >10 calendar days: {len(stale)} ({len(stale) / max(len(df), 1) * 100:.2f}%)")
        print("\nTop zero-volume ratios:")
        cols = ["code", "name", "rows", "zero_volume_ratio", "limit_day_ratio", "end"]
        print(df.sort_values("zero_volume_ratio", ascending=False)[cols].head(10).to_string(index=False))
