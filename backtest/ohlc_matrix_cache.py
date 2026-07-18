"""Global OHLC matrix cache for open-ledger backtests.

The cache is built once from per-stock CSV files and reused across arbitrary
date-window backtests. Runtime loads are matrix slices instead of repeated CSV
scans.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CACHE_VERSION = 2
MATRIX_DTYPE = np.float64
STOCK_SUFFIXES = {"SH", "SZ", "BJ"}
RAW_FIELDS = ("open", "high", "low", "close", "volume", "money")
DERIVED_FIELDS = ("pre_close", "pct_chg")


def is_stock_csv(path: Path) -> bool:
    parts = path.stem.split(".")
    return (
        len(parts) == 2
        and len(parts[0]) == 6
        and parts[0].isdigit()
        and parts[1].upper() in STOCK_SUFFIXES
    )


def discover_stock_csvs(data_dir):
    data_path = Path(data_dir)
    return sorted(path for path in data_path.glob("*.csv") if is_stock_csv(path))


def source_signature(csv_paths):
    digest = hashlib.sha256()
    for path in csv_paths:
        stat = path.stat()
        digest.update(
            f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode(
                "utf-8", errors="ignore"
            )
        )
    return {
        "source_count": len(csv_paths),
        "source_hash": digest.hexdigest(),
    }


def cache_paths(cache_dir):
    root = Path(cache_dir)
    paths = {
        "root": root,
        "meta": root / "ohlc_matrix_meta.json",
    }
    for field in RAW_FIELDS:
        paths[field] = root / f"{field}.dat"
    return paths


def _read_trade_dates(path):
    try:
        frame = pd.read_csv(path, usecols=["trade_date"], parse_dates=["trade_date"])
    except Exception:
        return []
    return pd.to_datetime(frame["trade_date"]).dropna().dt.normalize().tolist()


def _read_ohlc_frame(path):
    frame = pd.read_csv(
        path,
        usecols=lambda col: str(col).strip().lower()
        in {"trade_date", *RAW_FIELDS},
        parse_dates=["trade_date"],
    )
    frame.columns = frame.columns.str.strip().str.lower()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame.dropna(subset=["trade_date"])
    return frame.sort_values("trade_date")


def build_ohlc_matrix_cache(data_dir, cache_dir, progress_every=1000):
    paths = cache_paths(cache_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    csv_paths = discover_stock_csvs(data_dir)
    if not csv_paths:
        raise ValueError(f"No stock CSV files found under {data_dir}")

    dates = set()
    for i, path in enumerate(csv_paths, start=1):
        dates.update(_read_trade_dates(path))
        if progress_every > 0 and i % progress_every == 0:
            print(f"scanned OHLC dates {i}/{len(csv_paths)}", flush=True)
    all_dates = pd.DatetimeIndex(sorted(dates))
    all_codes = [path.stem for path in csv_paths]
    shape = (len(all_dates), len(all_codes))
    if not all_dates.empty:
        date2idx = {date: i for i, date in enumerate(all_dates)}
    else:
        date2idx = {}

    matrices = {
        name: np.memmap(path, dtype=MATRIX_DTYPE, mode="w+", shape=shape)
        for name, path in ((field, paths[field]) for field in RAW_FIELDS)
    }
    for matrix in matrices.values():
        matrix[:] = np.nan
        matrix.flush()

    for col, path in enumerate(csv_paths):
        try:
            frame = _read_ohlc_frame(path)
        except Exception:
            continue
        if frame.empty:
            continue
        row_idx = frame["trade_date"].map(date2idx).to_numpy()
        valid = pd.notna(row_idx)
        if not np.any(valid):
            continue
        row_idx = row_idx[valid].astype(np.int64)
        for field in RAW_FIELDS:
            if field not in frame.columns:
                continue
            matrices[field][row_idx, col] = pd.to_numeric(
                frame.loc[valid, field], errors="coerce"
            ).to_numpy(dtype=MATRIX_DTYPE)
        if progress_every > 0 and (col + 1) % progress_every == 0:
            print(f"built OHLC matrix {col + 1}/{len(csv_paths)}", flush=True)

    for matrix in matrices.values():
        matrix.flush()

    signature = source_signature(csv_paths)
    meta = {
        "version": CACHE_VERSION,
        "data_dir": str(Path(data_dir).resolve()),
        "dtype": np.dtype(MATRIX_DTYPE).name,
        "shape": list(shape),
        "fields": list(RAW_FIELDS),
        "derived_fields": list(DERIVED_FIELDS),
        "dates": [str(date.date()) for date in all_dates],
        "codes": all_codes,
        **signature,
    }
    tmp_meta = paths["meta"].with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    tmp_meta.replace(paths["meta"])
    return meta


def load_ohlc_matrix_meta(cache_dir):
    path = cache_paths(cache_dir)["meta"]
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def matrix_cache_is_current(data_dir, cache_dir):
    meta = load_ohlc_matrix_meta(cache_dir)
    if not meta or meta.get("version") != CACHE_VERSION:
        return False
    if meta.get("data_dir") != str(Path(data_dir).resolve()):
        return False
    paths = cache_paths(cache_dir)
    fields = meta.get("fields") or ("open", "close", "money")
    if not all(paths[name].exists() for name in fields):
        return False
    csv_paths = discover_stock_csvs(data_dir)
    return source_signature(csv_paths) == {
        "source_count": meta.get("source_count"),
        "source_hash": meta.get("source_hash"),
    }


def ensure_ohlc_matrix_cache(data_dir, cache_dir, progress_every=1000, rebuild=False):
    if rebuild or not matrix_cache_is_current(data_dir, cache_dir):
        print(f"building OHLC matrix cache: {cache_dir}", flush=True)
        return build_ohlc_matrix_cache(data_dir, cache_dir, progress_every=progress_every)
    meta = load_ohlc_matrix_meta(cache_dir)
    print(
        f"loaded OHLC matrix metadata: {cache_dir} "
        f"codes={len(meta['codes'])} dates={len(meta['dates'])}",
        flush=True,
    )
    return meta


def _slice_positions(dates, start_date=None, end_date=None):
    all_dates = pd.DatetimeIndex(pd.to_datetime(dates))
    start = pd.Timestamp(start_date).normalize() if start_date is not None else None
    end = pd.Timestamp(end_date).normalize() if end_date is not None else None
    mask = np.ones(len(all_dates), dtype=bool)
    if start is not None:
        mask &= all_dates >= start
    if end is not None:
        mask &= all_dates <= end
    return all_dates[mask], np.flatnonzero(mask)


def load_ohlc_money_from_matrix_cache(
    data_dir,
    cache_dir,
    codes,
    money_scale,
    start_date=None,
    end_date=None,
    progress_every=1000,
    rebuild=False,
):
    meta = ensure_ohlc_matrix_cache(
        data_dir,
        cache_dir,
        progress_every=progress_every,
        rebuild=rebuild,
    )
    code2idx = {code: i for i, code in enumerate(meta["codes"])}
    selected_codes = [str(code) for code in codes if str(code) in code2idx]
    if not selected_codes:
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name="trade_date"))
        return empty.copy(), empty.copy(), empty.copy()

    all_dates, row_idx = _slice_positions(meta["dates"], start_date, end_date)
    col_idx = np.asarray([code2idx[code] for code in selected_codes], dtype=np.int64)
    if len(row_idx) == 0:
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name="trade_date"), columns=selected_codes)
        return empty.copy(), empty.copy(), empty.copy()

    frames = load_ohlcv_fields_from_matrix_cache(
        data_dir,
        cache_dir,
        selected_codes,
        money_scale=money_scale,
        start_date=start_date,
        end_date=end_date,
        fields=("open", "close", "money"),
        progress_every=progress_every,
        rebuild=False,
    )
    return frames["open"], frames["close"], frames["money"]


def load_ohlcv_fields_from_matrix_cache(
    data_dir,
    cache_dir,
    codes,
    money_scale=1.0,
    start_date=None,
    end_date=None,
    fields=RAW_FIELDS,
    progress_every=1000,
    rebuild=False,
):
    meta = ensure_ohlc_matrix_cache(
        data_dir,
        cache_dir,
        progress_every=progress_every,
        rebuild=rebuild,
    )
    requested = tuple(fields)
    unknown = sorted(set(requested) - set(RAW_FIELDS) - set(DERIVED_FIELDS))
    if unknown:
        raise ValueError(f"Unknown OHLC matrix fields: {unknown}")

    code2idx = {code: i for i, code in enumerate(meta["codes"])}
    selected_codes = [str(code) for code in codes if str(code) in code2idx]
    if not selected_codes:
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name="trade_date"))
        return {field: empty.copy() for field in requested}

    all_dates, row_idx = _slice_positions(meta["dates"], start_date, end_date)
    index = pd.DatetimeIndex(all_dates, name="trade_date")
    if len(row_idx) == 0:
        empty = pd.DataFrame(index=index, columns=selected_codes)
        return {field: empty.copy() for field in requested}

    paths = cache_paths(cache_dir)
    shape = tuple(meta["shape"])
    dtype = meta["dtype"]
    col_idx = np.asarray([code2idx[code] for code in selected_codes], dtype=np.int64)
    ix = np.ix_(row_idx, col_idx)
    frames = {}
    raw_needed = set(requested) & set(RAW_FIELDS)
    if "pre_close" in requested or "pct_chg" in requested:
        raw_needed.add("close")
    for field in sorted(raw_needed):
        matrix = np.memmap(paths[field], dtype=dtype, mode="r", shape=shape)
        values = np.asarray(matrix[ix], dtype=np.float64)
        if field == "money":
            values *= float(money_scale)
        frames[field] = pd.DataFrame(values, index=index, columns=selected_codes)
    if "pre_close" in requested or "pct_chg" in requested:
        pre_close = frames["close"].shift(1)
        frames["pre_close"] = pre_close
    if "pct_chg" in requested:
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = frames["close"] / frames["pre_close"] - 1.0
        frames["pct_chg"] = pct.replace([np.inf, -np.inf], np.nan)
    return {field: frames[field] for field in requested}
