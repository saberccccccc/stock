#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data quality and PIT coverage report for the experiment branch."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def _fmt_pct(x):
    return f"{100.0 * x:.2f}%"


def _safe_read_parquet(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"[WARN] failed to read {path}: {exc}")
        return None


def report_raw_data(data_dir):
    data_dir = Path(data_dir)
    files = sorted(f for f in data_dir.glob("*.csv") if f.name[0].isdigit())
    rows = []
    for path in files:
        try:
            df = pd.read_csv(path, usecols=["trade_date", "close", "volume"])
            dates = pd.to_datetime(df["trade_date"], errors="coerce")
            rows.append({
                "code": path.stem,
                "rows": len(df),
                "start": dates.min(),
                "end": dates.max(),
                "bad_close": int((pd.to_numeric(df["close"], errors="coerce") <= 0).sum()),
                "bad_volume": int((pd.to_numeric(df["volume"], errors="coerce") < 0).sum()),
            })
        except Exception:
            rows.append({"code": path.stem, "rows": 0, "start": pd.NaT, "end": pd.NaT,
                         "bad_close": -1, "bad_volume": -1})

    out = pd.DataFrame(rows)
    print("\n== Raw daily data ==")
    print(f"stocks: {len(out)}")
    if out.empty:
        return out
    print(f"date span: {out['start'].min().date()} ~ {out['end'].max().date()}")
    print(f"median rows: {out['rows'].median():.0f}, min rows: {out['rows'].min():.0f}")
    stale_cutoff = out["end"].max() - pd.Timedelta(days=10)
    stale = out[out["end"] < stale_cutoff]
    print(f"stale >10 calendar days vs latest: {len(stale)} ({_fmt_pct(len(stale) / max(len(out), 1))})")
    print(f"bad close rows: {int(out['bad_close'].clip(lower=0).sum())}")
    print(f"bad volume rows: {int(out['bad_volume'].clip(lower=0).sum())}")
    return out


def report_fundamentals(codes):
    df = _safe_read_parquet("cache/fundamental_features_akshare.parquet")
    print("\n== Fundamentals PIT cache ==")
    if df is None or df.empty:
        print("missing or empty")
        return pd.DataFrame()
    df = df.copy()
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    covered = set(df["ts_code"].astype(str).unique())
    print(f"rows: {len(df)}, stocks: {len(covered)}, coverage: {_fmt_pct(len(covered & codes) / max(len(codes), 1))}")
    print(f"effective span: {df['effective_date'].min().date()} ~ {df['effective_date'].max().date()}")
    bad_pit = (df["effective_date"] < df["end_date"]).sum()
    print(f"effective_date before report end_date: {bad_pit}")
    for col in ["roe", "revenue_yoy"]:
        if col in df.columns:
            nonzero = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().mean()
            print(f"{col} non-null: {_fmt_pct(nonzero)}")
    return df


def report_shareholder(codes):
    df = _safe_read_parquet("cache/shareholder_features.parquet")
    print("\n== Shareholder cache ==")
    if df is None or df.empty:
        print("missing or empty")
        return pd.DataFrame()
    df = df.copy()
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    covered = set(df["code"].astype(str).str.zfill(6))
    pure_codes = {c[:6] for c in codes}
    print(f"rows: {len(df)}, stocks: {len(covered)}, coverage: {_fmt_pct(len(covered & pure_codes) / max(len(pure_codes), 1))}")
    print(f"announce span: {df['announce_date'].min().date()} ~ {df['announce_date'].max().date()}")
    print(f"missing announce_date: {_fmt_pct(df['announce_date'].isna().mean())}")
    return df


def report_restricted(codes):
    df = _safe_read_parquet("cache/restricted_features.parquet")
    print("\n== Restricted release cache ==")
    if df is None or df.empty:
        print("missing or empty")
        return pd.DataFrame()
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    if "effective_date" in df.columns:
        df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    covered = set(df["code"].astype(str).str.zfill(6))
    pure_codes = {c[:6] for c in codes}
    print(f"rows: {len(df)}, stocks: {len(covered)}, coverage: {_fmt_pct(len(covered & pure_codes) / max(len(pure_codes), 1))}")
    print(f"release span: {df['release_date'].min().date()} ~ {df['release_date'].max().date()}")
    if "effective_date" in df.columns:
        lag = (df["release_date"] - df["effective_date"]).dt.days
        print(f"effective lag median days: {lag.median():.0f}, min: {lag.min():.0f}")
    else:
        print("effective_date missing: rerun restricted feature download/update to materialize conservative PIT dates")
    return df


def main():
    parser = argparse.ArgumentParser(description="Report raw/PIT data coverage and obvious data issues.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default=None, help="Optional CSV path for per-stock raw data summary.")
    parser.add_argument("--universe", action="store_true", help="Also report ST/short-listing/tradability diagnostics.")
    parser.add_argument("--universe-output", default=None, help="Optional CSV path for universe diagnostics.")
    args = parser.parse_args()

    raw = report_raw_data(args.data_dir)
    codes = set(raw["code"].astype(str)) if not raw.empty else set()
    report_fundamentals(codes)
    report_shareholder(codes)
    report_restricted(codes)

    if args.universe:
        from data.universe_filters import print_universe_report, stock_universe_diagnostics
        uni = stock_universe_diagnostics(args.data_dir)
        print_universe_report(uni)
        if args.universe_output and not uni.empty:
            out = Path(args.universe_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            uni.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"\nsaved universe diagnostics: {out}")

    if args.output and not raw.empty:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nsaved raw stock summary: {out}")


if __name__ == "__main__":
    main()
