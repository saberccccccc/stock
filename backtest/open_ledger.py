"""Reusable helpers for open-price share-ledger backtests."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from alpha.io import load_alpha_rows as load_shared_alpha_rows
from backtest.execution import apply_open_ledger_constraints, open_limit_trade_mask
from backtest.ohlc_matrix_cache import (
    load_ohlc_money_from_matrix_cache,
    load_ohlcv_fields_from_matrix_cache,
)
from backtest.monthly_ohlcv_cache import MonthlyOhlcvCache
from backtest.reports import (
    calc_active_management_metrics,
    calc_extended_metrics,
    calc_metrics,
)
from backtest.strategy import retention_target, select_target_policy
from data.providers import CsvMarketDailyBackend
from data.st_status import (
    find_st_status_events_path,
    load_st_status_events as load_historical_st_status_events,
)


def parse_float_list(raw):
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def load_alpha_rows(path):
    return load_shared_alpha_rows(path, timestamp_dates=True)


def normalize_date_bound(value):
    if value is None:
        return None
    return pd.Timestamp(value).normalize()


def infer_ohlc_load_window(alpha_rows, max_data_date=None, execution_lag=0, lookback_days=160):
    if not alpha_rows:
        return None, normalize_date_bound(max_data_date)
    dates = [pd.Timestamp(row["date"]).normalize() for row in alpha_rows]
    first_signal = min(dates)
    last_signal = max(dates)
    load_start = first_signal - pd.Timedelta(days=int(lookback_days))
    load_end = normalize_date_bound(max_data_date)
    if load_end is None:
        load_end = last_signal + pd.Timedelta(days=int(execution_lag) + 10)
    return load_start, load_end


def _ohlc_cache_path(data_dir, codes, money_scale, start_date, end_date, cache_dir):
    data_path = Path(data_dir)
    digest = hashlib.sha256()
    digest.update(b"open_ledger_ohlc_v2")
    digest.update(str(data_path.resolve()).encode("utf-8", errors="ignore"))
    digest.update(f"|money_scale={float(money_scale):.12g}".encode("ascii"))
    digest.update(f"|start={start_date if start_date is not None else ''}".encode("ascii"))
    digest.update(f"|end={end_date if end_date is not None else ''}".encode("ascii"))
    for code in sorted(str(code) for code in codes):
        path = data_path / f"{code}.csv"
        if not path.exists():
            digest.update(f"|{code}:missing".encode("utf-8", errors="ignore"))
            continue
        stat = path.stat()
        digest.update(
            f"|{code}:{stat.st_size}:{stat.st_mtime_ns}".encode(
                "utf-8", errors="ignore"
            )
        )
    return Path(cache_dir) / f"ohlc_money_{digest.hexdigest()[:24]}.npz"


def _load_ohlc_cache(cache_path):
    try:
        payload = np.load(cache_path, allow_pickle=False)
        index = pd.DatetimeIndex(pd.to_datetime(payload["index"].astype("datetime64[ns]")))
        columns = [str(code) for code in payload["columns"].astype(str).tolist()]
        open_df = pd.DataFrame(payload["open"], index=index, columns=columns)
        close_df = pd.DataFrame(payload["close"], index=index, columns=columns)
        money_df = pd.DataFrame(payload["money"], index=index, columns=columns)
        return open_df, close_df, money_df
    except Exception:
        return None


def _save_ohlc_cache(cache_path, open_df, close_df, money_df):
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            np.savez(
                handle,
                open=open_df.to_numpy(dtype=np.float64, copy=False),
                close=close_df.to_numpy(dtype=np.float64, copy=False),
                money=money_df.to_numpy(dtype=np.float64, copy=False),
                index=open_df.index.to_numpy(dtype="datetime64[ns]"),
                columns=np.asarray(open_df.columns, dtype=str),
            )
        tmp_path.replace(cache_path)
    except Exception:
        return


def load_ohlc_money(
    data_dir,
    codes,
    money_scale,
    progress_every,
    start_date=None,
    end_date=None,
    cache_dir=None,
    use_cache=True,
    matrix_cache_dir=None,
    use_matrix_cache=True,
    rebuild_matrix_cache=False,
):
    start_ts = normalize_date_bound(start_date)
    end_ts = normalize_date_bound(end_date)
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date")

    if matrix_cache_dir and use_matrix_cache:
        try:
            return load_ohlc_money_from_matrix_cache(
                data_dir,
                matrix_cache_dir,
                codes,
                money_scale,
                start_date=start_ts,
                end_date=end_ts,
                progress_every=progress_every,
                rebuild=rebuild_matrix_cache,
            )
        except Exception as exc:
            print(f"OHLC matrix cache unavailable, falling back to CSV: {exc}", flush=True)

    cache_path = None
    if cache_dir and use_cache:
        cache_path = _ohlc_cache_path(
            data_dir, codes, money_scale, start_ts, end_ts, cache_dir
        )
        if cache_path.exists():
            cached = _load_ohlc_cache(cache_path)
            if cached is not None:
                print(f"loaded OHLC cache: {cache_path}", flush=True)
                return cached

    open_series = {}
    close_series = {}
    money_series = {}
    for i, code in enumerate(codes, start=1):
       path = Path(data_dir) / f"{code}.csv"
       if not path.exists():
           continue
       try:
           frame = pd.read_csv(
               path,
               usecols=["trade_date", "open", "close", "money"],
               parse_dates=["trade_date"],
           )
           frame.columns = frame.columns.str.strip().str.lower()
           if start_ts is not None:
               frame = frame.loc[frame["trade_date"] >= start_ts]
           if end_ts is not None:
               frame = frame.loc[frame["trade_date"] <= end_ts]
           if frame.empty:
               continue
           frame = frame.set_index("trade_date").sort_index()
           open_series[code] = frame["open"].astype(float).replace([np.inf, -np.inf], np.nan)
           close_series[code] = frame["close"].astype(float).replace([np.inf, -np.inf], np.nan)
           money_series[code] = (frame["money"].astype(float) * float(money_scale)).replace([np.inf, -np.inf], np.nan)
       except Exception:
           continue
       if progress_every > 0 and i % progress_every == 0:
           print(f"loaded OHLC data {i}/{len(codes)}", flush=True)
    if not open_series:
       empty = pd.DataFrame(index=pd.DatetimeIndex([], name="trade_date"))
       return empty.copy(), empty.copy(), empty.copy()
    all_dates = pd.DatetimeIndex(sorted(set().union(*[series.index for series in open_series.values()])))
    open_df = pd.DataFrame({code: series.reindex(all_dates) for code, series in open_series.items()}, index=all_dates)
    close_df = pd.DataFrame({code: series.reindex(all_dates) for code, series in close_series.items()}, index=all_dates)
    money_df = pd.DataFrame({code: series.reindex(all_dates) for code, series in money_series.items()}, index=all_dates)
    if cache_path is not None:
       _save_ohlc_cache(cache_path, open_df, close_df, money_df)
       print(f"saved OHLC cache: {cache_path}", flush=True)
    return open_df, close_df, money_df


def load_execution_market_frames(
    args,
    codes,
    *,
    start_date,
    end_date,
):
    """Load execution inputs through the explicitly selected storage backend."""
    backend = str(getattr(args, "ohlc_backend", "legacy")).strip().lower()
    if backend == "legacy":
        open_df, close_df, money_df = load_ohlc_money(
            args.data_dir,
            codes,
            args.money_scale,
            args.progress_every,
            start_date=start_date,
            end_date=end_date,
            cache_dir=args.ohlc_cache_dir,
            use_cache=not args.no_ohlc_cache,
            matrix_cache_dir=args.ohlc_matrix_cache_dir,
            use_matrix_cache=not args.no_ohlc_matrix_cache,
            rebuild_matrix_cache=args.rebuild_ohlc_matrix_cache,
        )
        return {
            "open": open_df,
            "close": close_df,
            "money": money_df,
        }
    if backend == "csv":
        requested_fields = ("open", "high", "low", "close", "volume", "money")
        normalized_codes = list(
            dict.fromkeys(
                str(code).strip().upper()
                for code in codes
                if str(code).strip()
            )
        )
        long = CsvMarketDailyBackend(args.data_dir).load_long(
            codes=normalized_codes,
            fields=requested_fields,
            start_date=start_date,
            end_date=end_date,
        )
        if long.empty:
            empty = pd.DataFrame(
                index=pd.DatetimeIndex([], name="trade_date"),
                dtype=float,
            )
            return {field: empty.copy() for field in requested_fields}
        present = set(long["code"].astype(str))
        ordered_codes = [code for code in normalized_codes if code in present]
        wide = long.pivot(
            index="trade_date",
            columns="code",
            values=list(requested_fields),
        ).sort_index()
        frames = {
            field: wide[field].reindex(columns=ordered_codes).astype(float)
            for field in requested_fields
        }
        frames["money"] = frames["money"] * float(args.money_scale)
        return frames
    if backend != "monthly":
        raise ValueError(f"unknown OHLC backend: {backend}")
    cache = MonthlyOhlcvCache(
        store_root=args.market_daily_store_root,
        cache_root=args.ohlc_monthly_cache_dir,
    )
    return cache.load(
        codes=codes,
        fields=("open", "high", "low", "close", "volume", "money"),
        start_date=start_date,
        end_date=end_date,
        money_scale=args.money_scale,
    )


def recompute_adv(money, adv_window):
    min_periods = max(3, int(adv_window) // 4)
    return money.rolling(int(adv_window), min_periods=min_periods).mean().shift(1)


def load_stock_name_map(data_dir):
    metadata = load_stock_metadata(data_dir)
    return metadata["names"]


def _read_csv_with_fallback(path):
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    return None


def _parse_yyyymmdd(value):
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NaT
    if text.endswith(".0"):
        text = text[:-2]
    try:
        return pd.Timestamp(pd.to_datetime(text, format="%Y%m%d")).normalize()
    except Exception:
        return pd.Timestamp(pd.to_datetime(text, errors="coerce")).normalize()


def load_stock_metadata(data_dir):
    stable_path = Path(data_dir) / "stable_stocks.csv"
    if not stable_path.exists():
        return {"names": {}, "list_dates": {}}
    frame = _read_csv_with_fallback(stable_path)
    if frame is None or "ts_code" not in frame:
        return {"names": {}, "list_dates": {}}
    codes = frame["ts_code"].astype(str).str.strip()
    names = (
        frame["name"].astype(str).str.strip()
        if "name" in frame
        else pd.Series([""] * len(frame))
    )
    list_dates = (
        frame["list_date"].map(_parse_yyyymmdd)
        if "list_date" in frame
        else pd.Series([pd.NaT] * len(frame))
    )
    return {
        "names": dict(zip(codes, names)),
        "list_dates": {
            code: date
            for code, date in zip(codes, list_dates)
            if pd.notna(date)
        },
    }


def _load_st_status_events(data_dir):
    historical_path = find_st_status_events_path(data_dir)
    if historical_path is not None:
        return load_historical_st_status_events(data_dir)

    candidates = [
        Path(data_dir) / "stock_industry.csv",
        Path(data_dir).parent / "stock_industry.csv",
    ]
    events = []
    for path in candidates:
        if not path.exists():
            continue
        frame = _read_csv_with_fallback(path)
        if frame is None:
            continue
        date_col = "updateDate" if "updateDate" in frame else None
        code_col = "code" if "code" in frame else None
        name_col = "code_name" if "code_name" in frame else None
        if date_col is None or code_col is None or name_col is None:
            continue
        for _, row in frame[[date_col, code_col, name_col]].dropna(subset=[code_col]).iterrows():
            code = normalize_ts_code(row[code_col])
            if not code:
                continue
            date = pd.Timestamp(pd.to_datetime(row[date_col], errors="coerce")).normalize()
            if pd.isna(date):
                continue
            events.append((date, code, "ST" in str(row[name_col]).upper()))
    return events


def _st_status_frame(index, columns, data_dir):
    historical_path = find_st_status_events_path(data_dir)
    base = pd.DataFrame(False, index=index, columns=columns)
    if historical_path is None:
        metadata = load_stock_metadata(data_dir)
        name_map = metadata["names"]
        for code in columns:
            if "ST" in str(name_map.get(str(code), "")).upper():
                base[str(code)] = True

    events = _load_st_status_events(data_dir)
    if not events:
        return base
    for code in columns:
        code_events = sorted((date, is_st) for date, event_code, is_st in events if event_code == str(code))
        if not code_events:
            continue
        event_series = pd.Series(
            [is_st for _, is_st in code_events],
            index=pd.DatetimeIndex([date for date, _ in code_events]),
        )
        event_series = event_series[~event_series.index.duplicated(keep="last")]
        asof_status = event_series.reindex(index.union(event_series.index)).sort_index().ffill().reindex(index)
        base.loc[asof_status.notna(), str(code)] = asof_status.dropna().astype(bool)
    return base.astype(bool)


def _estimated_prior_trading_days(list_date, first_date):
    if pd.isna(list_date) or pd.isna(first_date) or list_date >= first_date:
        return 0
    try:
        return int(np.busday_count(list_date.date(), first_date.date()))
    except Exception:
        return max(0, int((first_date - list_date).days * 5 / 7))


def _listing_trade_age_frame(index, columns, data_dir, valid_trade):
    metadata = load_stock_metadata(data_dir)
    list_dates = metadata["list_dates"]
    age = pd.DataFrame(0, index=index, columns=columns, dtype=np.int32)
    if len(index) == 0:
        return age
    first_date = pd.Timestamp(index[0]).normalize()
    for code in columns:
        code_str = str(code)
        valid = valid_trade[code_str].fillna(False).astype(bool)
        counted_age = valid.cumsum().astype(np.int32)
        list_date = list_dates.get(code_str)
        if list_date is not None and pd.notna(list_date):
            listed = pd.Series(index >= list_date, index=index)
            prior_days = _estimated_prior_trading_days(list_date, first_date)
            calendar_age = listed.cumsum().astype(np.int32) + prior_days
            calendar_age.loc[~listed] = 0
            age[code_str] = np.maximum(counted_age.to_numpy(), calendar_age.to_numpy())
        else:
            age[code_str] = counted_age
    return age


def _limit_pct_frame(index, columns, data_dir, listing_trade_age=None, no_limit_first_trading_days=0):
    limit_pct = pd.DataFrame(0.10, index=index, columns=columns, dtype=np.float64)
    st_status = _st_status_frame(index, columns, data_dir)
    reform_date = pd.Timestamp("2020-08-24")
    for code in columns:
        code_str = str(code)
        symbol = code_str.split(".", 1)[0]
        suffix = code_str.split(".", 1)[1].upper() if "." in code_str else ""
        if suffix == "BJ":
            limit_pct[code_str] = 0.30
        elif symbol.startswith("688"):
            limit_pct[code_str] = 0.20
        elif symbol.startswith("300"):
            limit_pct.loc[limit_pct.index >= reform_date, code_str] = 0.20
    limit_pct = limit_pct.mask(st_status, 0.05)
    if listing_trade_age is not None and int(no_limit_first_trading_days) > 0:
        no_limit = (listing_trade_age > 0) & (
            listing_trade_age <= int(no_limit_first_trading_days)
        )
        limit_pct = limit_pct.mask(no_limit, np.nan)
    return limit_pct


def _round_price_to_cent(price_df):
    values = price_df.to_numpy(dtype=np.float64, copy=True)
    with np.errstate(invalid="ignore"):
        rounded = np.floor(values * 100.0 + 0.5) / 100.0
    return pd.DataFrame(rounded, index=price_df.index, columns=price_df.columns)


def build_execution_constraint_masks(
    open_df,
    close_df,
    high_df,
    low_df,
    volume_df,
    money_df,
    data_dir,
    block_intraday_limit_touch=False,
    tolerance=1e-4,
    no_limit_first_trading_days=5,
    min_buy_listing_days=60,
):
    high_df = high_df.reindex(index=open_df.index, columns=open_df.columns)
    low_df = low_df.reindex(index=open_df.index, columns=open_df.columns)
    volume_df = volume_df.reindex(index=open_df.index, columns=open_df.columns)
    money_df = money_df.reindex(index=open_df.index, columns=open_df.columns)

    valid_open = np.isfinite(open_df) & (open_df > 0)
    valid_close = np.isfinite(close_df) & (close_df > 0)
    valid_trade = valid_open & valid_close
    listing_trade_age = _listing_trade_age_frame(
        open_df.index,
        open_df.columns,
        data_dir,
        valid_trade,
    )
    pre_close = close_df.shift(1)
    limit_pct = _limit_pct_frame(
        open_df.index,
        open_df.columns,
        data_dir,
        listing_trade_age=listing_trade_age,
        no_limit_first_trading_days=no_limit_first_trading_days,
    )
    limit_up_price = _round_price_to_cent(pre_close * (1.0 + limit_pct))
    limit_down_price = _round_price_to_cent(pre_close * (1.0 - limit_pct))

    valid_pre_close = np.isfinite(pre_close) & (pre_close > 0)
    no_trade = (
        (~valid_open)
        | (~np.isfinite(volume_df))
        | (volume_df <= 0)
        | (~np.isfinite(money_df))
        | (money_df <= 0)
    )
    limit_up_open = valid_pre_close & valid_open & (
        open_df >= limit_up_price * (1.0 - float(tolerance))
    )
    limit_down_open = valid_pre_close & valid_open & (
        open_df <= limit_down_price * (1.0 + float(tolerance))
    )
    limit_up_touch = valid_pre_close & np.isfinite(high_df) & (
        high_df >= limit_up_price * (1.0 - float(tolerance))
    )
    limit_down_touch = valid_pre_close & np.isfinite(low_df) & (
        low_df <= limit_down_price * (1.0 + float(tolerance))
    )

    if block_intraday_limit_touch:
        buy_block = limit_up_open | limit_up_touch
        sell_block = limit_down_open | limit_down_touch
    else:
        buy_block = limit_up_open
        sell_block = limit_down_open
    new_stock_buy_block = pd.DataFrame(
        False,
        index=open_df.index,
        columns=open_df.columns,
    )
    if int(min_buy_listing_days) > 0:
        new_stock_buy_block = (listing_trade_age > 0) & (
            listing_trade_age < int(min_buy_listing_days)
        )
        buy_block = buy_block | new_stock_buy_block

    return {
        "buy_block": buy_block.fillna(False).astype(bool),
        "sell_block": sell_block.fillna(False).astype(bool),
        "no_trade": no_trade.fillna(True).astype(bool),
        "limit_up_open": limit_up_open.fillna(False).astype(bool),
        "limit_down_open": limit_down_open.fillna(False).astype(bool),
        "limit_up_touch": limit_up_touch.fillna(False).astype(bool),
        "limit_down_touch": limit_down_touch.fillna(False).astype(bool),
        "new_stock_buy_block": new_stock_buy_block.fillna(False).astype(bool),
    }


def prepare_execution_constraint_masks(
    args,
    all_codes,
    open_df,
    close_df,
    money_df,
    load_start=None,
    load_end=None,
    ohlcv_frames=None,
):
    mode = getattr(args, "execution_constraint_mode", "proxy")
    if mode == "proxy":
        return None
    if mode != "realistic":
        raise ValueError(f"Unknown execution_constraint_mode: {mode}")
    backend = str(getattr(args, "ohlc_backend", "legacy")).strip().lower()
    if backend == "legacy" and getattr(args, "no_ohlc_matrix_cache", False):
        raise RuntimeError("realistic execution constraints require OHLC matrix cache")
    cache_path = _execution_mask_cache_path(args, all_codes, open_df.index)
    if not getattr(args, "no_execution_mask_cache", False):
        cached = _load_execution_mask_cache(cache_path, open_df.index, open_df.columns)
        if cached is not None:
            print(f"loaded execution mask cache: {cache_path}", flush=True)
            return cached
    if ohlcv_frames is not None:
        frames = ohlcv_frames
    else:
        frames = load_ohlcv_fields_from_matrix_cache(
            args.data_dir,
            args.ohlc_matrix_cache_dir,
            all_codes,
            money_scale=args.money_scale,
            start_date=load_start,
            end_date=load_end,
            fields=("high", "low", "volume", "money"),
            progress_every=args.progress_every,
            rebuild=getattr(args, "rebuild_ohlc_matrix_cache", False),
        )
    matrix_money_df = frames["money"].reindex(index=open_df.index, columns=open_df.columns)
    if matrix_money_df.isna().all().all():
        matrix_money_df = money_df
    masks = build_execution_constraint_masks(
        open_df=open_df,
        close_df=close_df,
        high_df=frames["high"],
        low_df=frames["low"],
        volume_df=frames["volume"],
        money_df=matrix_money_df,
        data_dir=args.data_dir,
        block_intraday_limit_touch=getattr(args, "block_intraday_limit_touch", False),
        tolerance=getattr(args, "limit_price_tolerance", 1e-4),
        no_limit_first_trading_days=getattr(args, "no_limit_first_trading_days", 5),
        min_buy_listing_days=getattr(args, "min_buy_listing_days", 60),
    )
    if not getattr(args, "no_execution_mask_cache", False):
        _save_execution_mask_cache(cache_path, masks)
        print(f"saved execution mask cache: {cache_path}", flush=True)
    return masks


def _execution_mask_cache_path(args, codes, index):
    """Build a cache key from immutable execution inputs and mask settings."""
    digest = hashlib.sha256()
    digest.update(b"open_ledger_execution_masks_v2")
    digest.update(str(Path(args.data_dir).resolve()).encode("utf-8", errors="ignore"))
    backend = str(getattr(args, "ohlc_backend", "legacy")).strip().lower()
    digest.update(f"|backend={backend}".encode("ascii"))
    if backend == "monthly" and len(index):
        identity = MonthlyOhlcvCache(
            store_root=args.market_daily_store_root,
            cache_root=args.ohlc_monthly_cache_dir,
        ).active_identity(
            start_date=index[0],
            end_date=index[-1],
            ensure_current=False,
        )
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    elif backend == "csv":
        for code in codes:
            source = Path(args.data_dir) / f"{code}.csv"
            if source.is_file():
                stat = source.stat()
                digest.update(
                    f"|csv={code}:{stat.st_size}:{stat.st_mtime_ns}".encode(
                        "utf-8"
                    )
                )
    else:
        matrix_meta = Path(args.ohlc_matrix_cache_dir) / "ohlc_matrix_meta.json"
        if matrix_meta.exists():
            stat = matrix_meta.stat()
            digest.update(f"|matrix={stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    for relative in (
        "stable_stocks.csv",
        "../stock_industry.csv",
        "st_status_events.csv",
        "../st_status_events.csv",
        "st_status_events_manifest.json",
        "../st_status_events_manifest.json",
    ):
        source = Path(args.data_dir) / relative
        if source.exists():
            stat = source.stat()
            digest.update(f"|{relative}={stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    for value in (
        getattr(args, "block_intraday_limit_touch", False),
        getattr(args, "limit_price_tolerance", 1e-4),
        getattr(args, "no_limit_first_trading_days", 5),
        getattr(args, "min_buy_listing_days", 60),
        len(index),
        str(index[0]) if len(index) else "",
        str(index[-1]) if len(index) else "",
    ):
        digest.update(f"|{value}".encode("utf-8"))
    for code in codes:
        digest.update(f"|{code}".encode("utf-8"))
    cache_dir = Path(getattr(args, "execution_mask_cache_dir", "cache/open_ledger_execution_masks"))
    return cache_dir / f"execution_masks_{digest.hexdigest()[:24]}.npz"


def _load_execution_mask_cache(path, index, columns):
    if not path.exists():
        return None
    names = (
        "buy_block",
        "sell_block",
        "no_trade",
        "limit_up_open",
        "limit_down_open",
        "limit_up_touch",
        "limit_down_touch",
        "new_stock_buy_block",
    )
    try:
        with np.load(path, allow_pickle=False) as payload:
            masks = {}
            for name in names:
                values = payload[name]
                if values.shape != (len(index), len(columns)):
                    return None
                masks[name] = pd.DataFrame(values.astype(bool, copy=False), index=index, columns=columns)
            return masks
    except (OSError, KeyError, ValueError):
        return None


def _save_execution_mask_cache(path, masks):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                **{name: frame.to_numpy(dtype=bool, copy=False) for name, frame in masks.items()},
            )
        tmp_path.replace(path)
    except OSError:
        return


def save_stage_breakdown(out_dir, returns_by_tag):
    yearly_rows = []
    monthly_rows = []
    for tag, returns_df in returns_by_tag.items():
       if returns_df.empty:
           continue
       frame = returns_df.copy()
       frame["date"] = pd.to_datetime(frame["date"])
       frame["year"] = frame["date"].dt.year
       frame["month"] = frame["date"].dt.to_period("M").astype(str)
       for year, group in frame.groupby("year"):
           ann, sharpe, mdd = calc_metrics(group["return"].to_numpy(float))
           active_metrics = calc_active_management_metrics(
               group["return"].to_numpy(float),
               group["benchmark_return"].to_numpy(float)
               if "benchmark_return" in group
               else None,
           )
           yearly_rows.append({
               "tag": tag,
               "period": str(year),
               "days": len(group),
               "ann": ann,
               "sharpe": sharpe,
               "mdd": mdd,
               "sum_return": float(group["return"].sum()),
               "benchmark_ann": active_metrics["benchmark_ann"],
               "active_ann": active_metrics["active_ann"],
               "tracking_error": active_metrics["tracking_error"],
               "information_ratio": active_metrics["information_ratio"],
               "beta_to_benchmark": active_metrics["beta_to_benchmark"],
           })
       ann, sharpe, mdd = calc_metrics(frame["return"].to_numpy(float))
       active_metrics = calc_active_management_metrics(
           frame["return"].to_numpy(float),
           frame["benchmark_return"].to_numpy(float)
           if "benchmark_return" in frame
           else None,
       )
       yearly_rows.append({
           "tag": tag,
           "period": "all",
           "days": len(frame),
           "ann": ann,
           "sharpe": sharpe,
           "mdd": mdd,
           "sum_return": float(frame["return"].sum()),
           "benchmark_ann": active_metrics["benchmark_ann"],
           "active_ann": active_metrics["active_ann"],
           "tracking_error": active_metrics["tracking_error"],
           "information_ratio": active_metrics["information_ratio"],
           "beta_to_benchmark": active_metrics["beta_to_benchmark"],
       })
       for month, group in frame.groupby("month"):
           active_metrics = calc_active_management_metrics(
               group["return"].to_numpy(float),
               group["benchmark_return"].to_numpy(float)
               if "benchmark_return" in group
               else None,
           )
           monthly_rows.append({
               "tag": tag,
               "month": month,
               "days": len(group),
               "sum_return": float(group["return"].sum()),
               "mean_return": float(group["return"].mean()),
               "benchmark_sum_return": (
                   float(group["benchmark_return"].sum())
                   if "benchmark_return" in group
                   else 0.0
               ),
               "active_sum_return": (
                   float(group["active_return"].sum())
                   if "active_return" in group
                   else float(group["return"].sum())
               ),
               "information_ratio": active_metrics["information_ratio"],
           })
    if yearly_rows:
       pd.DataFrame(yearly_rows).to_csv(Path(out_dir) / "yearly_summary.csv", index=False)
    if monthly_rows:
       pd.DataFrame(monthly_rows).to_csv(Path(out_dir) / "monthly_summary.csv", index=False)


def build_desired_target(row, current_codes, target_frac, hold_frac):
    codes = list(row["codes"])
    proposal = retention_target(
        codes,
        current_codes,
        target_frac=target_frac,
        hold_frac=hold_frac,
    )
    hold_n = max(proposal.target_n, int(len(codes) * float(hold_frac)))
    rank_map = {code: i for i, code in enumerate(codes)}
    return list(proposal.selected), list(proposal.retained), proposal.target_n, hold_n, rank_map


def apply_defensive_tilt(desired, selected, code2idx, date, close_df, idx_close, defensive_tilt):
    """Blend target weights toward defensive stocks (low beta, low vol) in weak markets."""
    if defensive_tilt <= 0 or not selected:
       return desired
    # Market state: HS300 close < MA60
    # idx_close passed directly
    if date not in idx_close.index:
       return desired
    ma60 = idx_close.rolling(60, min_periods=30).mean()
    if date not in ma60.index or pd.isna(ma60.loc[date]):
       return desired
    if idx_close.loc[date] >= ma60.loc[date]:
       return desired  # not weak
    # Compute defense scores for selected stocks
    scores = np.ones(len(selected)) * 0.5
    mkt = idx_close.pct_change()
    for i, code in enumerate(selected):
       if code not in close_df.columns:
           continue
       cs = close_df[code].dropna()
       cb = cs[cs.index <= date]
       if len(cb) < 30:
           continue
       ret = cb.pct_change().dropna()
       vol = ret.tail(60).std() if len(ret) >= 60 else ret.std()
       idx_r = mkt.reindex(ret.tail(60).index).dropna()
       al = ret.tail(60).reindex(idx_r.index).dropna()
       if len(al) >= 20:
           ix_v = np.nanvar(idx_r.reindex(al.index).values)
           beta = np.nanmean((al.values - al.values.mean()) * (idx_r.reindex(al.index).values - idx_r.reindex(al.index).values.mean())) / max(ix_v, 1e-12)
       else:
           beta = 1.0
       vol_s = max(0.0, min(1.0, 1.0 - vol / 0.05))
       beta_s = max(0.0, min(1.0, 1.0 - (beta - 1.0) / 1.5))
       scores[i] = 0.5 * vol_s + 0.5 * beta_s
    avg = np.mean(scores)
    if avg > 1e-12:
       for i, code in enumerate(selected):
           idx = code2idx.get(code, -1)
           if idx >= 0:
               desired[idx] = desired[idx] * (1.0 - defensive_tilt + defensive_tilt * scores[i] / avg)
    return desired


def should_apply_defensive_tilt(defensive_tilt, market_mult, market_mult_below=1.0):
    """Gate defensive tilt by the current portfolio market multiplier."""
    tilt = float(defensive_tilt or 0.0)
    if tilt <= 0:
       return False
    threshold = float(market_mult_below)
    return threshold >= 1.0 or float(market_mult) < threshold


def active_drawdown_throttle_state(
    active_returns,
    lookback_days,
    trigger_return,
    scale,
    cooldown_days,
    remaining_days,
    trigger_allowed=True,
):
    """Return the current active-risk throttle using only realized active returns."""
    lookback_days = int(lookback_days or 0)
    cooldown_days = int(cooldown_days or 0)
    remaining_days = max(int(remaining_days or 0), 0)
    scale = float(scale)
    if (
       lookback_days <= 0
       or cooldown_days <= 0
       or not np.isfinite(scale)
       or scale >= 1.0
       or scale < 0.0
    ):
       return 1.0, np.nan, 0
    trailing_active = np.nan
    if len(active_returns) >= lookback_days:
       window = np.asarray(active_returns[-lookback_days:], dtype=np.float64)
       window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
       trailing_active = float(np.prod(1.0 + window) - 1.0)
       if bool(trigger_allowed) and trailing_active <= float(trigger_return):
           remaining_days = max(remaining_days, cooldown_days)
    current_scale = scale if remaining_days > 0 else 1.0
    return float(current_scale), trailing_active, int(remaining_days)


def active_drawdown_throttle_continuous_state(
    active_returns,
    lookback_days,
    trigger_return,
    min_scale,
    width,
    cooldown_days,
    remaining_days,
    remaining_scale=1.0,
    trigger_allowed=True,
):
    """Return a smooth throttle scale as active return worsens past trigger."""
    lookback_days = int(lookback_days or 0)
    cooldown_days = int(cooldown_days or 0)
    remaining_days = max(int(remaining_days or 0), 0)
    min_scale = float(min_scale)
    width = float(width)
    remaining_scale = float(remaining_scale or 1.0)
    if (
       lookback_days <= 0
       or not np.isfinite(min_scale)
       or not np.isfinite(width)
       or min_scale < 0.0
       or min_scale >= 1.0
       or width <= 0.0
    ):
       return 1.0, np.nan, 0, 1.0

    trailing_active = np.nan
    current_scale = 1.0
    if len(active_returns) >= lookback_days:
       window = np.asarray(active_returns[-lookback_days:], dtype=np.float64)
       window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
       trailing_active = float(np.prod(1.0 + window) - 1.0)
       if bool(trigger_allowed) and trailing_active <= float(trigger_return):
           severity = (float(trigger_return) - trailing_active) / width
           severity = float(np.clip(severity, 0.0, 1.0))
           current_scale = 1.0 - severity * (1.0 - min_scale)
           current_scale = float(np.clip(current_scale, min_scale, 1.0))
           if cooldown_days > 0:
              remaining_days = max(remaining_days, cooldown_days)
              remaining_scale = min(remaining_scale, current_scale)

    if cooldown_days > 0:
       if remaining_days > 0:
           current_scale = min(current_scale, remaining_scale)
       else:
           remaining_scale = 1.0

    return float(current_scale), trailing_active, int(remaining_days), float(remaining_scale)


def parse_active_drawdown_throttle_steps(raw):
    """Parse stair-step throttle specs like '-0.03:0.85,-0.06:0.65'."""
    if raw is None or str(raw).strip() == "":
       return []
    steps = []
    for token in str(raw).split(","):
       token = token.strip()
       if not token:
           continue
       trigger_text, separator, scale_text = token.partition(":")
       if not separator:
           raise ValueError(f"invalid active drawdown throttle step: {token!r}")
       trigger = float(trigger_text)
       scale = float(scale_text)
       if not np.isfinite(trigger) or not np.isfinite(scale):
           raise ValueError(f"non-finite active drawdown throttle step: {token!r}")
       if scale < 0.0 or scale >= 1.0:
           raise ValueError(f"step scale must be in [0, 1): {token!r}")
       steps.append((trigger, scale))
    return sorted(steps, key=lambda item: item[0], reverse=True)


def active_drawdown_throttle_step_state(
    active_returns,
    lookback_days,
    steps,
    cooldown_days,
    remaining_days,
    remaining_scale=1.0,
    trigger_allowed=True,
):
    """Return throttle scale for multi-level active drawdown rules."""
    lookback_days = int(lookback_days or 0)
    cooldown_days = int(cooldown_days or 0)
    remaining_days = max(int(remaining_days or 0), 0)
    remaining_scale = float(remaining_scale or 1.0)
    steps = list(steps or [])
    if lookback_days <= 0 or cooldown_days <= 0 or not steps:
       return 1.0, np.nan, 0, 1.0, ""

    trailing_active = np.nan
    triggered_scale = None
    triggered_label = ""
    if len(active_returns) >= lookback_days:
       window = np.asarray(active_returns[-lookback_days:], dtype=np.float64)
       window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
       trailing_active = float(np.prod(1.0 + window) - 1.0)
       if bool(trigger_allowed):
           for trigger, scale in steps:
              if trailing_active <= trigger:
                  triggered_scale = scale if triggered_scale is None else min(triggered_scale, scale)
                  triggered_label = f"{trigger:g}:{triggered_scale:g}"

    if triggered_scale is not None:
       remaining_days = max(remaining_days, cooldown_days)
       remaining_scale = min(remaining_scale, float(triggered_scale))
    elif remaining_days <= 0:
       remaining_scale = 1.0

    current_scale = remaining_scale if remaining_days > 0 else 1.0
    return float(current_scale), trailing_active, int(remaining_days), float(remaining_scale), triggered_label


def active_drawdown_observation(active_drawdown_history, lookback):
    """Compute trailing active return for diagnostics without changing risk budget."""
    lookback = int(lookback or 0)
    if lookback <= 0 or len(active_drawdown_history) < lookback:
       return np.nan
    window = np.asarray(active_drawdown_history[-lookback:], dtype=np.float64)
    if len(window) < lookback or not np.all(np.isfinite(window)):
       return np.nan
    return float(np.prod(1.0 + window) - 1.0)


def active_drawdown_condition_needs_industry(mode):
    return str(mode or "active_only").strip().lower().startswith("crowding")


def active_drawdown_condition_state(
    args,
    codes,
    weights,
    industry_map,
    close_df,
    close_ret_daily,
    day,
):
    """Evaluate optional state gates for active drawdown throttle triggers."""
    mode = str(
        getattr(args, "active_drawdown_throttle_condition", "active_only")
        or "active_only"
    ).strip().lower()
    weights = np.asarray(weights, dtype=np.float64)
    gross = float(np.sum(np.abs(weights)))
    diag = {
       "active_drawdown_condition": mode,
       "active_drawdown_condition_allowed": 1,
       "active_drawdown_condition_top_industry": "",
       "active_drawdown_condition_top_industry_weight": 0.0,
       "active_drawdown_condition_industry_hhi": 0.0,
       "active_drawdown_condition_momentum20": np.nan,
       "active_drawdown_condition_volatility60": np.nan,
    }
    if mode in ("", "active_only", "none"):
       return True, diag
    if gross <= 1e-12:
       diag["active_drawdown_condition_allowed"] = 0
       return False, diag

    industry_weights = {}
    if industry_map:
       for code, weight in zip(codes, weights):
           if abs(float(weight)) <= 1e-12:
               continue
           industry = industry_map.get(normalize_ts_code(code), "UNKNOWN")
           industry_weights[industry] = industry_weights.get(industry, 0.0) + float(weight)
    if industry_weights:
       top_industry, top_weight = max(
           industry_weights.items(),
           key=lambda item: abs(float(item[1])),
       )
       diag["active_drawdown_condition_top_industry"] = str(top_industry)
       diag["active_drawdown_condition_top_industry_weight"] = float(top_weight)
       diag["active_drawdown_condition_industry_hhi"] = float(
           sum((abs(float(w)) / gross) ** 2 for w in industry_weights.values())
       )

    momentum20 = np.nan
    if day >= 21 and len(close_df.index) > day:
       latest = close_df.iloc[day - 1].to_numpy(dtype=np.float64)
       past = close_df.iloc[day - 21].to_numpy(dtype=np.float64)
       with np.errstate(divide="ignore", invalid="ignore"):
           stock_mom = latest / past - 1.0
       aw = np.abs(weights)
       valid = np.isfinite(stock_mom) & np.isfinite(aw) & (aw > 1e-12)
       if np.any(valid):
           momentum20 = float(np.sum(aw[valid] * stock_mom[valid]) / np.sum(aw[valid]))
    diag["active_drawdown_condition_momentum20"] = momentum20

    volatility60 = np.nan
    if day >= 20:
       start = max(1, int(day) - 60)
       if close_ret_daily.shape[0] == len(weights):
           hist = close_ret_daily[:, start:int(day)]
           vol_axis = 1
       else:
           hist = close_ret_daily[start:int(day), :]
           vol_axis = 0
       if hist.size:
           stock_vol = np.nanstd(hist, axis=vol_axis) * np.sqrt(252)
           aw = np.abs(weights)
           valid = np.isfinite(stock_vol) & np.isfinite(aw) & (aw > 1e-12)
           if np.any(valid):
               volatility60 = float(np.sum(aw[valid] * stock_vol[valid]) / np.sum(aw[valid]))
    diag["active_drawdown_condition_volatility60"] = volatility60

    crowding_ok = True
    if "crowding" in mode:
       min_top = float(
           getattr(args, "active_drawdown_throttle_min_top_industry_weight", 0.0)
           or 0.0
       )
       min_hhi = float(
           getattr(args, "active_drawdown_throttle_min_industry_hhi", 0.0)
           or 0.0
       )
       crowding_ok = (
           abs(float(diag["active_drawdown_condition_top_industry_weight"])) >= min_top
           and float(diag["active_drawdown_condition_industry_hhi"]) >= min_hhi
       )

    momentum_ok = True
    if "momentum" in mode:
       max_momentum = float(
           getattr(args, "active_drawdown_throttle_max_momentum20", 0.0)
           or 0.0
       )
       momentum_ok = np.isfinite(momentum20) and float(momentum20) <= max_momentum

    vol_ok = True
    if "volatility" in mode:
       min_vol = float(
           getattr(args, "active_drawdown_throttle_min_volatility60", 0.0)
           or 0.0
       )
       vol_ok = np.isfinite(volatility60) and float(volatility60) >= min_vol

    allowed = bool(crowding_ok and momentum_ok and vol_ok)
    diag["active_drawdown_condition_allowed"] = int(allowed)
    return allowed, diag


def weights_from_selected(selected, code2idx, n_codes, gross_weight, max_weight):
    weights = np.zeros(n_codes, dtype=np.float64)
    valid_indices = list(dict.fromkeys(
       code2idx[code] for code in selected if code in code2idx
    ))
    if not valid_indices:
       return weights
    equal_weight = min(
       max(float(max_weight), 0.0),
       max(float(gross_weight), 0.0) / len(valid_indices),
    )
    weights[valid_indices] = equal_weight
    return weights


def load_index_returns(data_dir, index_file, all_dates):
    path = Path(data_dir) / index_file
    if not path.exists():
       return pd.Series(np.nan, index=all_dates), pd.Series(0.0, index=all_dates)
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    date_col = "trade_date" if "trade_date" in frame.columns else frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.set_index(date_col).sort_index()
    close = frame["close"].astype(float).reindex(all_dates)
    daily = close.pct_change().fillna(0.0)
    return close, daily


def normalize_ts_code(code):
    code = str(code).strip()
    if not code or code.lower() == "nan":
       return None
    lower = code.lower()
    if lower.startswith("sh.") or lower.startswith("sz.") or lower.startswith("bj."):
       return f"{code[3:9]}.{lower[:2].upper()}"
    if "." in code:
       left, right = code.split(".", 1)
       if left.isdigit():
           return f"{left.zfill(6)}.{right.upper()}"
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) >= 6:
       suffix = "BJ" if digits[:2] in {"43", "83", "87", "88", "92"} else ("SH" if digits[:1] in {"5", "6", "9"} else "SZ")
       return f"{digits[-6:]}.{suffix}"
    return code


def load_industry_map(path):
    path = Path(path)
    if not path.exists():
       return {}
    frame = pd.read_csv(path)
    if "code" not in frame.columns or "industry" not in frame.columns:
       return {}
    frame["code_norm"] = frame["code"].map(normalize_ts_code)
    frame = frame.dropna(subset=["code_norm", "industry"])
    return dict(zip(frame["code_norm"], frame["industry"].astype(str)))


def compute_market_multiplier(
    idx_close,
    idx_daily,
    ret_daily,
    col_cur,
    mode,
    min_mult,
    max_mult,
    legacy_bear_mult=0.7,
    legacy_crash_mult=0.3,
):
    if mode in (None, "", "none"):
       return 1.0
    if col_cur < 60 or not np.isfinite(idx_close.iloc[col_cur]):
       return float(max_mult)

    idx_cur = float(idx_close.iloc[col_cur])
    idx_ma60 = float(idx_close.iloc[col_cur - 60:col_cur].mean())
    if mode == "legacy":
       market_mult = float(legacy_bear_mult) if idx_cur < idx_ma60 else 1.0
       if col_cur >= 120 and np.isfinite(idx_close.iloc[col_cur - 120]) and idx_close.iloc[col_cur - 120] > 0:
           idx_ret_6m = idx_cur / float(idx_close.iloc[col_cur - 120]) - 1.0
           if idx_ret_6m < -0.10:
               market_mult = min(market_mult, float(legacy_crash_mult))
       return float(np.clip(market_mult, 0.0, max_mult))

    if mode != "dynamic":
       raise ValueError(f"Unknown market_timing_mode: {mode}")

    ma_score = 1.0 if idx_cur >= idx_ma60 else 0.0
    mom_score = 0.5
    if col_cur >= 20 and np.isfinite(idx_close.iloc[col_cur - 20]) and idx_close.iloc[col_cur - 20] > 0:
       mom20 = idx_cur / float(idx_close.iloc[col_cur - 20]) - 1.0
       mom_score = float(np.clip((mom20 + 0.08) / 0.16, 0.0, 1.0))

    breadth_score = 0.5
    if col_cur >= 20 and ret_daily.shape[1] >= col_cur:
       recent_rets = ret_daily[:, col_cur - 20:col_cur]
       finite = np.isfinite(recent_rets)
       if np.any(finite):
           breadth_score = float(np.nanmean(recent_rets[finite] > 0))

    vol_score = 0.5
    if col_cur >= 20:
       recent_idx_ret = np.asarray(
           idx_daily.iloc[col_cur - 19:col_cur + 1],
           dtype=float,
       )
       recent_idx_ret = recent_idx_ret[np.isfinite(recent_idx_ret)]
       if len(recent_idx_ret) > 5:
           ann_vol = float(np.std(recent_idx_ret) * np.sqrt(252))
           vol_score = float(1.0 - np.clip((ann_vol - 0.15) / 0.25, 0.0, 1.0))

    score = 0.4 * ma_score + 0.3 * mom_score + 0.2 * breadth_score + 0.1 * vol_score
    market_mult = min_mult + (max_mult - min_mult) * score
    if col_cur >= 120 and np.isfinite(idx_close.iloc[col_cur - 120]) and idx_close.iloc[col_cur - 120] > 0:
       idx_ret_6m = idx_cur / float(idx_close.iloc[col_cur - 120]) - 1.0
       if idx_ret_6m < -0.10:
           market_mult = min(market_mult, max(min_mult, 0.35))
    return float(np.clip(market_mult, min_mult, max_mult))


def load_global_risk_features(path, all_dates):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"global risk feature file not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError("global risk feature file must contain a date column")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.drop_duplicates("date", keep="last").set_index("date").sort_index()
    return frame.reindex(pd.DatetimeIndex(all_dates).normalize())


def global_risk_overlay_state(args, global_risk_frame, date):
    """Return a gross/target risk budget from global overnight features."""
    mode = getattr(args, "global_risk_overlay_mode", "none")
    if global_risk_frame is None:
        return {
            "global_risk_triggered": 0,
            "global_risk_pressure": np.nan,
            "global_risk_market_scale": 1.0,
            "global_risk_target_frac": np.nan,
        }
    if mode not in (None, "", "none", "defensive_pressure", "defensive_pressure_continuous"):
        raise ValueError(f"Unknown global_risk_overlay_mode: {mode}")
    date = pd.Timestamp(date).normalize()
    if date not in global_risk_frame.index:
        return {
            "global_risk_triggered": 0,
            "global_risk_pressure": np.nan,
            "global_risk_market_scale": 1.0,
            "global_risk_target_frac": np.nan,
        }
    column = getattr(args, "global_risk_pressure_col", "global_defensive_pressure")
    row = global_risk_frame.loc[date]
    pressure = row.get(column, np.nan)
    try:
        pressure = float(pressure)
    except Exception:
        pressure = np.nan
    if not np.isfinite(pressure):
        return {
            "global_risk_triggered": 0,
            "global_risk_pressure": np.nan,
            "global_risk_market_scale": 1.0,
            "global_risk_target_frac": np.nan,
        }
    if mode in (None, "", "none"):
        return {
            "global_risk_triggered": 0,
            "global_risk_pressure": pressure,
            "global_risk_market_scale": 1.0,
            "global_risk_target_frac": np.nan,
        }
    threshold = float(getattr(args, "global_risk_pressure_threshold", 0.0) or 0.0)
    triggered = pressure >= threshold
    if not triggered:
        return {
            "global_risk_triggered": 0,
            "global_risk_pressure": pressure,
            "global_risk_market_scale": 1.0,
            "global_risk_target_frac": np.nan,
        }
    if mode == "defensive_pressure_continuous":
        min_scale = float(getattr(args, "global_risk_market_scale", 1.0) or 1.0)
        width = float(getattr(args, "global_risk_pressure_width", 0.0) or 0.0)
        if width <= 0.0 or min_scale >= 1.0:
            market_scale = min_scale
        else:
            severity = float(np.clip((pressure - threshold) / width, 0.0, 1.0))
            market_scale = 1.0 - severity * (1.0 - min_scale)
            market_scale = float(np.clip(market_scale, min_scale, 1.0))
        return {
            "global_risk_triggered": int(market_scale < 1.0),
            "global_risk_pressure": pressure,
            "global_risk_market_scale": market_scale,
            "global_risk_target_frac": (
                float(getattr(args, "global_risk_target_frac"))
                if getattr(args, "global_risk_target_frac", None) is not None
                else np.nan
            ),
        }
    return {
        "global_risk_triggered": 1,
        "global_risk_pressure": pressure,
        "global_risk_market_scale": float(
            getattr(args, "global_risk_market_scale", 1.0) or 1.0
        ),
        "global_risk_target_frac": (
            float(getattr(args, "global_risk_target_frac"))
            if getattr(args, "global_risk_target_frac", None) is not None
            else np.nan
        ),
    }


def estimate_trailing_stock_risk(close_ret_daily, idx_daily, day_pos, window=60):
    """Estimate stock beta and residual volatility using data known before day_pos."""
    n_codes = close_ret_daily.shape[0]
    beta = np.zeros(n_codes, dtype=np.float64)
    residual_vol = np.zeros(n_codes, dtype=np.float64)
    if day_pos < 3 or close_ret_daily.shape[1] == 0:
       return beta, residual_vol, 0

    end = max(int(day_pos) - 1, 0)
    start = max(0, end - int(window))
    if end <= start:
       return beta, residual_vol, 0

    stock_hist = np.asarray(close_ret_daily[:, start:end], dtype=np.float64)
    idx_hist = np.asarray(idx_daily.iloc[start + 1:end + 1], dtype=np.float64)
    valid_idx = np.isfinite(idx_hist)
    if valid_idx.sum() < 3:
       return beta, residual_vol, int(valid_idx.sum())

    stock_hist = stock_hist[:, valid_idx]
    idx_hist = idx_hist[valid_idx]
    stock_hist = np.nan_to_num(stock_hist, nan=0.0, posinf=0.0, neginf=0.0)
    idx_hist = np.nan_to_num(idx_hist, nan=0.0, posinf=0.0, neginf=0.0)

    idx_centered = idx_hist - idx_hist.mean()
    stock_centered = stock_hist - stock_hist.mean(axis=1, keepdims=True)
    idx_var = float(np.mean(idx_centered ** 2))
    if idx_var > 1e-12:
       beta = np.mean(stock_centered * idx_centered, axis=1) / idx_var
       beta = np.clip(np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)
    residual = stock_centered - beta[:, None] * idx_centered[None, :]
    residual_vol = np.std(residual, axis=1) * np.sqrt(252)
    residual_vol = np.nan_to_num(residual_vol, nan=0.0, posinf=0.0, neginf=0.0)
    return beta, residual_vol, int(valid_idx.sum())


def prepare_open_ledger_context(open_df, close_df):
    """Precompute immutable matrices shared by a parameter-sweep run.

    A sweep changes portfolio rules, not OHLC inputs.  Keeping these derived
    arrays and the per-day risk observations in one context avoids rebuilding
    the same DataFrame-to-NumPy conversions for every grid cell.
    """
    codes = list(open_df.columns)
    open_mark = open_df.ffill().to_numpy(dtype=np.float64, copy=True)
    open_mark[~np.isfinite(open_mark)] = 0.0
    close_mat = close_df.to_numpy(dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        close_ret_daily = close_mat[:, 1:] / close_mat[:, :-1] - 1.0
    close_ret_daily[~np.isfinite(close_ret_daily)] = 0.0
    return {
        "codes": codes,
        "code2idx": {code: index for index, code in enumerate(codes)},
        "all_dates": open_df.index,
        "open_mark": open_mark,
        "close_ret_daily": close_ret_daily,
        "risk_cache": {},
    }


def summarize_portfolio_risk(weights, stock_beta, residual_vol):
    weights = np.asarray(weights, dtype=np.float64)
    stock_beta = np.asarray(stock_beta, dtype=np.float64)
    residual_vol = np.asarray(residual_vol, dtype=np.float64)
    gross = float(np.sum(np.abs(weights)))
    if gross <= 1e-12:
       return {
           "portfolio_beta_60d": 0.0,
           "portfolio_beta_per_gross_60d": 0.0,
           "portfolio_specific_vol_60d": 0.0,
       }
    beta_exposure = float(np.dot(weights, stock_beta))
    specific_var = float(np.sum((weights * residual_vol) ** 2))
    return {
       "portfolio_beta_60d": beta_exposure,
       "portfolio_beta_per_gross_60d": beta_exposure / gross,
       "portfolio_specific_vol_60d": float(np.sqrt(max(specific_var, 0.0))),
    }


def encode_holdings(codes, weights, min_abs_weight=1e-10):
    parts = []
    for code, weight in zip(codes, weights):
       weight = float(weight)
       if abs(weight) <= min_abs_weight:
           continue
       parts.append(f"{code}={weight:.10f}")
    return ";".join(parts)


def apply_industry_selection_cap(
    selected,
    candidate_codes,
    industry_map,
    max_industry_weight,
    gross_weight,
    max_weight,
    target_n,
):
    if not industry_map or max_industry_weight is None or float(max_industry_weight) <= 0:
       return selected, {
           "industry_cap_active": 0,
           "industry_cap_max_names": 0,
           "industry_cap_removed": 0,
       }
    target_n = max(int(target_n), 1)
    gross_weight = max(float(gross_weight), 0.0)
    if gross_weight <= 0:
       return selected, {
           "industry_cap_active": 0,
           "industry_cap_max_names": 0,
           "industry_cap_removed": 0,
       }
    equal_target = min(max(float(max_weight), 0.0), gross_weight / target_n)
    if equal_target <= 1e-12:
       return selected, {
           "industry_cap_active": 0,
           "industry_cap_max_names": 0,
           "industry_cap_removed": 0,
       }
    max_names = max(1, int(np.floor(float(max_industry_weight) / equal_target + 1e-12)))
    counts = {}
    capped = []
    capped_set = set()

    def industry_of(code):
       return industry_map.get(normalize_ts_code(code), "UNKNOWN")

    def try_add(code):
       if code in capped_set or len(capped) >= target_n:
           return False
       industry = industry_of(code)
       if counts.get(industry, 0) >= max_names:
           return False
       counts[industry] = counts.get(industry, 0) + 1
       capped.append(code)
       capped_set.add(code)
       return True

    for code in selected:
       try_add(code)
    for code in candidate_codes:
       if len(capped) >= target_n:
           break
       try_add(code)

    removed = max(0, len(selected) - len([code for code in selected if code in capped_set]))
    return capped, {
       "industry_cap_active": 1,
       "industry_cap_max_names": int(max_names),
       "industry_cap_removed": int(removed),
    }


def _rank01_from_values(values):
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, 0.5, dtype=np.float64)
    mask = np.isfinite(arr)
    if mask.sum() <= 1:
       return out
    order = np.argsort(arr[mask], kind="mergesort")
    ranks = np.empty(order.shape[0], dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, order.shape[0])
    out[mask] = ranks
    return out


def state_aware_selection_pressure(args, global_risk_frame, date):
    if global_risk_frame is None:
       return np.nan, 0.0
    date = pd.Timestamp(date).normalize()
    if date not in global_risk_frame.index:
       return np.nan, 0.0
    column = getattr(args, "state_aware_selection_pressure_col", None) or getattr(
       args,
       "global_risk_pressure_col",
       "global_defensive_pressure",
    )
    row = global_risk_frame.loc[date]
    pressure = row.get(column, np.nan)
    try:
       pressure = float(pressure)
    except Exception:
       pressure = np.nan
    if not np.isfinite(pressure):
       return np.nan, 0.0
    threshold = float(getattr(args, "state_aware_selection_pressure_threshold", 0.0) or 0.0)
    width = max(float(getattr(args, "state_aware_selection_pressure_width", 0.0) or 0.0), 1e-12)
    stress = float(np.clip((pressure - threshold) / width, 0.0, 1.0))
    return pressure, stress


def _state_selection_diag(
    *,
    active=0,
    pressure=np.nan,
    stress=0.0,
    changed=0,
    avg_new_risk=0.0,
    max_new_risk=0.0,
    top_industry_share=0.0,
    suppressed=0,
    mean_suppressed_risk_delta=0.0,
):
    return {
        "state_aware_selection_active": int(active),
        "state_aware_selection_pressure": pressure,
        "state_aware_selection_stress": float(stress),
        "state_aware_selection_changed": int(changed),
        "state_aware_selection_avg_new_risk": float(avg_new_risk),
        "state_aware_selection_max_new_risk": float(max_new_risk),
        "state_aware_selection_top_industry_share": float(top_industry_share),
        "state_aware_selection_suppressed": int(suppressed),
        "state_aware_selection_mean_suppressed_risk_delta": float(
            mean_suppressed_risk_delta
        ),
    }


def apply_state_aware_selection_rank(
    selected,
    row,
    current_selected,
    target_n,
    codes,
    code2idx,
    close_df,
    close_ret_daily,
    idx_daily,
    day,
    stock_beta,
    residual_vol,
    industry_map,
    global_risk_frame,
    args,
):
    """Rerank replacement candidates under fragile market states.

    This is a selection-layer risk budget: it preserves currently retained names
    first, then fills new slots from Alpha candidates after penalizing fragile
    exposures. It does not change gross exposure or target name count.

    ``risk_rank`` changes the order of new candidates under stress.  The separate
    ``risk_suppress`` family only suppresses a replacement when its risk score is
    materially above the risk score of the incumbent that would be displaced.
    Keeping these families separate makes the attribution and promotion gate
    interpretable.
    """
    mode = str(getattr(args, "state_aware_selection_mode", "none") or "none").strip().lower()
    if mode in ("", "none"):
       return selected, _state_selection_diag()
    if mode not in ("risk_rank", "risk_suppress"):
       raise ValueError(f"Unknown state_aware_selection_mode: {mode}")

    execution_date = close_df.index[day] if 0 <= int(day) < len(close_df.index) else row["date"]
    pressure, stress = state_aware_selection_pressure(args, global_risk_frame, execution_date)
    min_stress = float(getattr(args, "state_aware_selection_min_stress", 0.0) or 0.0)
    if stress <= min_stress or not selected:
       return selected, _state_selection_diag(pressure=pressure, stress=stress)

    candidate_codes = [normalize_ts_code(code) for code in row.get("codes", [])]
    n = len(candidate_codes)
    if n == 0:
       return selected, _state_selection_diag(pressure=pressure, stress=stress)

    target_n = max(1, int(target_n))
    current_set = {normalize_ts_code(code) for code in current_selected}
    selected_norm = [normalize_ts_code(code) for code in selected]
    retained = []
    retained_set = set()
    for code in selected_norm:
       if code in current_set and code not in retained_set:
           retained.append(code)
           retained_set.add(code)
       if len(retained) >= target_n:
           break

    new_slots = max(target_n - len(retained), 0)
    if new_slots <= 0:
       limited = retained[:target_n]
       return limited, _state_selection_diag(
           active=1,
           pressure=pressure,
           stress=stress,
           changed=int(limited != selected_norm[:len(limited)]),
       )

    top_frac = float(getattr(args, "state_aware_selection_top_frac", 0.006) or 0.006)
    top_n = min(n, max(1, int(np.ceil(n * top_frac))))
    industry_counts = {}
    if industry_map:
       for code in candidate_codes[:top_n]:
           industry = industry_map.get(code, "UNKNOWN")
           industry_counts[industry] = industry_counts.get(industry, 0) + 1
    denom = max(top_n, 1)
    industry_share = {key: value / denom for key, value in industry_counts.items()}
    top_industry_share = max(industry_share.values()) if industry_share else 0.0

    risk_rows = []
    for base_rank, code in enumerate(candidate_codes):
       if code in retained_set:
           continue
       idx = code2idx.get(code)
       if idx is None:
           continue
       if day >= 21 and idx < close_df.shape[1]:
           cur_close = close_df.iloc[day - 1, idx]
           old_close = close_df.iloc[max(day - 21, 0), idx]
           if np.isfinite(cur_close) and np.isfinite(old_close) and old_close > 0:
              momentum20 = float(cur_close / old_close - 1.0)
           else:
              momentum20 = np.nan
       else:
           momentum20 = np.nan
       beta = float(stock_beta[idx]) if idx < len(stock_beta) else np.nan
       vol = float(residual_vol[idx]) if idx < len(residual_vol) else np.nan
       industry = industry_map.get(code, "UNKNOWN") if industry_map else "UNKNOWN"
       crowd = float(industry_share.get(industry, 0.0))
       risk_rows.append((code, base_rank, momentum20, beta, vol, crowd))

    raw_rows = [item for item in risk_rows if item[0] not in current_set]
    scoring_rows = raw_rows if mode == "risk_rank" else risk_rows
    if not scoring_rows:
       return selected, _state_selection_diag(
           active=1,
           pressure=pressure,
           stress=stress,
           top_industry_share=top_industry_share,
       )

    momentum_rank = _rank01_from_values([item[2] for item in scoring_rows])
    beta_risk = np.clip(
       (np.nan_to_num([item[3] for item in scoring_rows], nan=1.0) - 0.8) / 1.0,
       0.0,
       1.0,
    )
    vol_rank = _rank01_from_values([item[4] for item in scoring_rows])
    crowd_risk = np.clip(
       np.asarray([item[5] for item in scoring_rows], dtype=np.float64)
       / max(float(getattr(args, "state_aware_selection_crowd_scale", 0.10) or 0.10), 1e-12),
       0.0,
       1.0,
    )
    weights = np.asarray([
       float(getattr(args, "state_aware_selection_momentum_weight", 0.40) or 0.0),
       float(getattr(args, "state_aware_selection_beta_weight", 0.20) or 0.0),
       float(getattr(args, "state_aware_selection_vol_weight", 0.20) or 0.0),
       float(getattr(args, "state_aware_selection_industry_weight", 0.20) or 0.0),
    ], dtype=np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
       risk = np.zeros(len(scoring_rows), dtype=np.float64)
    else:
       risk = (
          weights[0] * momentum_rank
          + weights[1] * beta_risk
          + weights[2] * vol_rank
          + weights[3] * crowd_risk
       ) / weight_sum

    risk_by_code = {
       item[0]: float(score)
       for item, score in zip(scoring_rows, risk)
    }
    suppressed = 0
    suppressed_deltas = []
    if mode == "risk_rank":
       rank_penalty = float(getattr(args, "state_aware_selection_rank_penalty", 0.0) or 0.0)
       adjusted = []
       for item, risk_value in zip(scoring_rows, risk):
          code, base_rank = item[0], item[1]
          if code in current_set:
             continue
          adjusted_rank = float(base_rank) + stress * rank_penalty * float(risk_value) * n
          adjusted.append((code, adjusted_rank, float(risk_value)))
       adjusted.sort(key=lambda item: (item[1], item[0]))
       picked_new = adjusted[:new_slots]
       final = retained + [code for code, _, _ in picked_new]
    else:
       selected_new = [code for code in selected_norm if code not in current_set]
       dropped_current = [
           normalize_ts_code(code)
           for code in current_selected
           if normalize_ts_code(code) in candidate_codes
           and normalize_ts_code(code) not in retained_set
       ]
       final = list(retained)
       risk_delta_threshold = float(
           getattr(args, "state_aware_selection_risk_delta_threshold", 0.15) or 0.0
       )
       for position, candidate in enumerate(selected_new):
          incumbent = dropped_current[position] if position < len(dropped_current) else None
          candidate_risk = risk_by_code.get(candidate, np.nan)
          incumbent_risk = risk_by_code.get(incumbent, np.nan) if incumbent else np.nan
          risk_delta = candidate_risk - incumbent_risk
          should_suppress = bool(
              incumbent
              and np.isfinite(candidate_risk)
              and np.isfinite(incumbent_risk)
              and risk_delta >= risk_delta_threshold
          )
          chosen = incumbent if should_suppress else candidate
          if should_suppress:
             suppressed += 1
             suppressed_deltas.append(float(risk_delta))
          if chosen not in final:
             final.append(chosen)

    final_set = set(final)
    for code in selected_norm:
       if len(final) >= target_n:
          break
       if code not in final_set:
          final.append(code)
          final_set.add(code)
    for code in candidate_codes:
       if len(final) >= target_n:
          break
       if code not in final_set:
          final.append(code)
          final_set.add(code)

    selected_new = [code for code in selected_norm if code not in current_set]
    final_new = [code for code in final if code not in current_set]
    risks = [risk_by_code.get(code, np.nan) for code in final_new]
    risks = [value for value in risks if np.isfinite(value)]
    return final[:target_n], _state_selection_diag(
       active=1,
       pressure=pressure,
       stress=stress,
       changed=int(final[:target_n] != selected_norm[:target_n]),
       avg_new_risk=float(np.mean(risks)) if risks else 0.0,
       max_new_risk=float(np.max(risks)) if risks else 0.0,
       top_industry_share=top_industry_share,
       suppressed=suppressed,
       mean_suppressed_risk_delta=(
           float(np.mean(suppressed_deltas)) if suppressed_deltas else 0.0
       ),
    )


def limit_new_names(
    selected,
    kept,
    row,
    max_new_names,
    current_selected,
    target_n,
    exit_hold_frac=None,
    switch_gap_frac=0.0,
    mode="legacy",
):
    if max_new_names <= 0 or not current_selected:
       return selected
    codes = list(row.get("codes", []))
    rank_map = {code: rank for rank, code in enumerate(codes)}
    if exit_hold_frac is not None and exit_hold_frac > 0:
       exit_n = max(int(len(codes) * float(exit_hold_frac)), target_n)
       eligible_current = [
           code for code in current_selected
           if rank_map.get(code, len(codes) + 1) < exit_n
       ]
    else:
       eligible_current = [code for code in current_selected if code in rank_map]
    current_ranked = sorted(eligible_current, key=lambda code: rank_map[code])
    target_n = max(int(target_n), 1)
    max_new_names = max(int(max_new_names), 0)
    switch_gap = max(int(len(codes) * float(switch_gap_frac)), 0)
    current_set = set(current_selected)
    if mode == "legacy":
       min_old = max(target_n - max_new_names, 0)
       limited = current_ranked[:min(len(current_ranked), min_old)]
       selected_set = set(limited)
       added = 0
       replacement_old = current_ranked[len(limited):]
       replacement_slot = 0
       for code in codes:
           if len(limited) >= target_n or added >= max_new_names:
               break
           if code in selected_set or code in current_set:
               continue
           if switch_gap > 0 and replacement_slot < len(replacement_old):
               old_code = replacement_old[replacement_slot]
               if rank_map[code] + switch_gap >= rank_map[old_code]:
                   continue
           limited.append(code)
           selected_set.add(code)
           added += 1
           replacement_slot += 1
       for code in current_ranked:
           if len(limited) >= target_n:
               break
           if code not in selected_set:
               limited.append(code)
               selected_set.add(code)
       return limited
    if mode != "at_most":
       raise ValueError(f"Unknown max_new_names mode: {mode}")

    desired_new = [code for code in selected if code not in current_set]
    new_limit = min(max_new_names, len(desired_new), target_n)
    old_target = max(target_n - new_limit, 0)
    limited = current_ranked[:old_target]
    selected_set = set(limited)
    replacement_old = current_ranked[old_target:]
    replacement_slot = 0

    for code in desired_new[:new_limit]:
       if len(limited) >= target_n:
           break
       if switch_gap > 0 and replacement_slot < len(replacement_old):
           old_code = replacement_old[replacement_slot]
           if rank_map[code] + switch_gap >= rank_map[old_code]:
               continue
       limited.append(code)
       selected_set.add(code)
       replacement_slot += 1

    for code in replacement_old:
       if len(limited) >= target_n:
           break
       if code not in selected_set:
           limited.append(code)
           selected_set.add(code)
    return limited


def summarize_open_ledger_result(
    returns_active,
    diag_df,
    closed_ages,
    target_frac,
    hold_frac,
    args,
    benchmark_returns=None,
):
    ann, sharpe, mdd = calc_metrics(returns_active)
    ext = calc_extended_metrics(returns_active)
    active_metrics = calc_active_management_metrics(returns_active, benchmark_returns)
    return {
       "target_frac": target_frac,
       "hold_frac": hold_frac,
       "n_return_days": int(len(returns_active)),
       "ann": float(ann),
       "sharpe": float(sharpe),
       "mdd": float(mdd),
       "calmar": float(ext.get("calmar", 0.0)),
       "sortino": float(ext.get("sortino", 0.0)),
       "win_rate": float(ext.get("win_rate", 0.0)),
       "avg_daily_return": float(np.mean(returns_active)) if len(returns_active) else 0.0,
       "vol": float(np.std(returns_active) * np.sqrt(252)) if len(returns_active) else 0.0,
       "benchmark_ann": active_metrics["benchmark_ann"],
       "benchmark_sharpe": active_metrics["benchmark_sharpe"],
       "benchmark_mdd": active_metrics["benchmark_mdd"],
       "active_ann": active_metrics["active_ann"],
       "active_sharpe": active_metrics["active_sharpe"],
       "active_mdd": active_metrics["active_mdd"],
       "tracking_error": active_metrics["tracking_error"],
       "information_ratio": active_metrics["information_ratio"],
       "beta_to_benchmark": active_metrics["beta_to_benchmark"],
       "benchmark_corr": active_metrics["benchmark_corr"],
       "avg_turnover": float(diag_df["turnover"].mean()) if "turnover" in diag_df else 0.0,
       "avg_executed_turnover": float(diag_df["executed_turnover"].mean()) if "executed_turnover" in diag_df else 0.0,
       "avg_unfilled_turnover": float(diag_df["unfilled_turnover"].mean()) if "unfilled_turnover" in diag_df else 0.0,
       "avg_holding_days": float(np.mean(closed_ages)) if closed_ages else 0.0,
       "avg_names": float(diag_df["selected_n"].mean()) if "selected_n" in diag_df else 0.0,
       "avg_gross_weight": float(diag_df["gross_weight"].mean()) if "gross_weight" in diag_df else 0.0,
       "avg_portfolio_beta_60d": (
           float(diag_df["portfolio_beta_60d"].mean())
           if "portfolio_beta_60d" in diag_df
           else 0.0
       ),
       "avg_portfolio_beta_per_gross_60d": (
           float(diag_df["portfolio_beta_per_gross_60d"].mean())
           if "portfolio_beta_per_gross_60d" in diag_df
           else 0.0
       ),
       "avg_portfolio_specific_vol_60d": (
           float(diag_df["portfolio_specific_vol_60d"].mean())
           if "portfolio_specific_vol_60d" in diag_df
           else 0.0
       ),
       "avg_risk_obs_60d": (
           float(diag_df["risk_obs_60d"].mean())
           if "risk_obs_60d" in diag_df
           else 0.0
       ),
       "max_industry_weight": float(getattr(args, "max_industry_weight", 0.0) or 0.0),
       "avg_industry_cap_removed": (
           float(diag_df["industry_cap_removed"].mean())
           if "industry_cap_removed" in diag_df
           else 0.0
       ),
       "industry_cap_days": (
           int((diag_df["industry_cap_removed"] > 0).sum())
           if "industry_cap_removed" in diag_df
           else 0
       ),
       "market_timing_mode": args.market_timing_mode,
       "avg_market_mult": float(diag_df["market_mult"].mean()) if "market_mult" in diag_df else 1.0,
       "active_drawdown_throttle_lookback": int(
           getattr(args, "active_drawdown_throttle_lookback", 0) or 0
       ),
       "active_drawdown_throttle_mode": str(
           getattr(args, "active_drawdown_throttle_mode", "fixed") or "fixed"
       ),
       "active_drawdown_throttle_trigger": float(
           getattr(args, "active_drawdown_throttle_trigger", 0.0) or 0.0
       ),
       "active_drawdown_throttle_scale": float(
           getattr(args, "active_drawdown_throttle_scale", 1.0) or 1.0
       ),
       "active_drawdown_throttle_continuous_width": float(
           getattr(args, "active_drawdown_throttle_continuous_width", 0.0) or 0.0
       ),
       "active_drawdown_throttle_steps": str(
           getattr(args, "active_drawdown_throttle_steps", "") or ""
       ),
       "active_drawdown_throttle_cooldown": int(
           getattr(args, "active_drawdown_throttle_cooldown", 0) or 0
       ),
       "active_drawdown_throttle_condition": str(
           getattr(args, "active_drawdown_throttle_condition", "active_only")
           or "active_only"
       ),
       "active_drawdown_throttle_min_top_industry_weight": float(
           getattr(args, "active_drawdown_throttle_min_top_industry_weight", 0.0)
           or 0.0
       ),
       "active_drawdown_throttle_min_industry_hhi": float(
           getattr(args, "active_drawdown_throttle_min_industry_hhi", 0.0)
           or 0.0
       ),
       "active_drawdown_throttle_max_momentum20": float(
           getattr(args, "active_drawdown_throttle_max_momentum20", 0.0)
           or 0.0
       ),
       "active_drawdown_throttle_min_volatility60": float(
           getattr(args, "active_drawdown_throttle_min_volatility60", 0.0)
           or 0.0
       ),
       "avg_active_drawdown_throttle_scale": (
           float(diag_df["active_drawdown_throttle_scale"].mean())
           if "active_drawdown_throttle_scale" in diag_df
           else 1.0
       ),
       "active_drawdown_throttle_days": (
           int((diag_df["active_drawdown_throttle_scale"] < 1.0).sum())
           if "active_drawdown_throttle_scale" in diag_df
           else 0
       ),
       "active_drawdown_condition_allowed_days": (
           int(diag_df["active_drawdown_condition_allowed"].sum())
           if "active_drawdown_condition_allowed" in diag_df
           else 0
       ),
       "avg_active_drawdown_condition_top_industry_abs_weight": (
           float(diag_df["active_drawdown_condition_top_industry_weight"].abs().mean())
           if "active_drawdown_condition_top_industry_weight" in diag_df
           else 0.0
       ),
       "avg_active_drawdown_condition_industry_hhi": (
           float(diag_df["active_drawdown_condition_industry_hhi"].mean())
           if "active_drawdown_condition_industry_hhi" in diag_df
           else 0.0
       ),
       "avg_active_drawdown_condition_momentum20": (
           float(diag_df["active_drawdown_condition_momentum20"].mean())
           if "active_drawdown_condition_momentum20" in diag_df
           else 0.0
       ),
       "avg_active_drawdown_condition_volatility60": (
           float(diag_df["active_drawdown_condition_volatility60"].mean())
           if "active_drawdown_condition_volatility60" in diag_df
           else 0.0
       ),
       "global_risk_overlay_mode": getattr(args, "global_risk_overlay_mode", "none"),
       "global_risk_pressure_col": getattr(
           args,
           "global_risk_pressure_col",
           "global_defensive_pressure",
       ),
       "global_risk_pressure_threshold": float(
           getattr(args, "global_risk_pressure_threshold", 0.0) or 0.0
       ),
       "global_risk_pressure_width": float(
           getattr(args, "global_risk_pressure_width", 0.0) or 0.0
       ),
       "global_risk_market_scale": float(
           getattr(args, "global_risk_market_scale", 1.0) or 1.0
       ),
       "global_risk_target_frac": (
           float(getattr(args, "global_risk_target_frac"))
           if getattr(args, "global_risk_target_frac", None) is not None
           else np.nan
       ),
       "avg_global_risk_pressure": (
           float(diag_df["global_risk_pressure"].mean())
           if "global_risk_pressure" in diag_df
           else np.nan
       ),
       "avg_global_risk_market_scale": (
           float(diag_df["global_risk_market_scale"].mean())
           if "global_risk_market_scale" in diag_df
           else 1.0
       ),
       "global_risk_overlay_days": (
           int(diag_df["global_risk_triggered"].sum())
           if "global_risk_triggered" in diag_df
           else 0
       ),
       "state_aware_selection_mode": getattr(args, "state_aware_selection_mode", "none"),
       "state_aware_selection_pressure_col": getattr(
           args,
           "state_aware_selection_pressure_col",
           getattr(args, "global_risk_pressure_col", "global_defensive_pressure"),
       ),
       "state_aware_selection_pressure_threshold": float(
           getattr(args, "state_aware_selection_pressure_threshold", 0.0) or 0.0
       ),
       "state_aware_selection_pressure_width": float(
           getattr(args, "state_aware_selection_pressure_width", 0.0) or 0.0
       ),
       "state_aware_selection_rank_penalty": float(
           getattr(args, "state_aware_selection_rank_penalty", 0.0) or 0.0
       ),
       "state_aware_selection_min_stress": float(
           getattr(args, "state_aware_selection_min_stress", 0.0) or 0.0
       ),
       "state_aware_selection_top_frac": float(
           getattr(args, "state_aware_selection_top_frac", 0.006) or 0.006
       ),
       "state_aware_selection_crowd_scale": float(
           getattr(args, "state_aware_selection_crowd_scale", 0.10) or 0.10
       ),
       "state_aware_selection_momentum_weight": float(
           getattr(args, "state_aware_selection_momentum_weight", 0.40) or 0.40
       ),
       "state_aware_selection_beta_weight": float(
           getattr(args, "state_aware_selection_beta_weight", 0.20) or 0.20
       ),
       "state_aware_selection_vol_weight": float(
           getattr(args, "state_aware_selection_vol_weight", 0.20) or 0.20
       ),
       "state_aware_selection_industry_weight": float(
           getattr(args, "state_aware_selection_industry_weight", 0.20) or 0.20
       ),
       "state_aware_selection_risk_delta_threshold": float(
           getattr(args, "state_aware_selection_risk_delta_threshold", 0.15) or 0.0
       ),
       "state_aware_selection_days": (
           int(diag_df["state_aware_selection_active"].sum())
           if "state_aware_selection_active" in diag_df
           else 0
       ),
       "state_aware_selection_changed_days": (
           int(diag_df["state_aware_selection_changed"].sum())
           if "state_aware_selection_changed" in diag_df
           else 0
       ),
       "avg_state_aware_selection_stress": (
           float(diag_df["state_aware_selection_stress"].mean())
           if "state_aware_selection_stress" in diag_df
           else 0.0
       ),
       "avg_state_aware_selection_new_risk": (
           float(diag_df["state_aware_selection_avg_new_risk"].mean())
           if "state_aware_selection_avg_new_risk" in diag_df
           else 0.0
       ),
       "avg_state_aware_selection_top_industry_share": (
           float(diag_df["state_aware_selection_top_industry_share"].mean())
           if "state_aware_selection_top_industry_share" in diag_df
           else 0.0
       ),
       "state_aware_selection_suppressed_days": (
           int((diag_df["state_aware_selection_suppressed"] > 0).sum())
           if "state_aware_selection_suppressed" in diag_df
           else 0
       ),
       "total_state_aware_selection_suppressed": (
           int(diag_df["state_aware_selection_suppressed"].sum())
           if "state_aware_selection_suppressed" in diag_df
           else 0
       ),
       "avg_state_aware_selection_suppressed_risk_delta": (
           float(diag_df["state_aware_selection_mean_suppressed_risk_delta"].mean())
           if "state_aware_selection_mean_suppressed_risk_delta" in diag_df
           else 0.0
       ),
       "total_cost": float(diag_df["cost"].sum()) if "cost" in diag_df else 0.0,
       "total_commission": float(diag_df["commission"].sum()) if "commission" in diag_df else 0.0,
       "total_stamp_tax": float(diag_df["stamp_tax"].sum()) if "stamp_tax" in diag_df else 0.0,
       "total_slippage": float(diag_df["slippage"].sum()) if "slippage" in diag_df else 0.0,
       "blocked_buy": int(diag_df["blocked_buy"].sum()) if "blocked_buy" in diag_df else 0,
       "blocked_sell": int(diag_df["blocked_sell"].sum()) if "blocked_sell" in diag_df else 0,
       "adv_blocked": int(diag_df["adv_blocked"].sum()) if "adv_blocked" in diag_df else 0,
       "missing_adv": int(diag_df["missing_adv"].sum()) if "missing_adv" in diag_df else 0,
       "no_open": int(diag_df["no_open"].sum()) if "no_open" in diag_df else 0,
       "no_trade_blocked": int(diag_df["no_trade_blocked"].sum()) if "no_trade_blocked" in diag_df else 0,
       "limit_up_open_blocked": int(diag_df["limit_up_open_blocked"].sum()) if "limit_up_open_blocked" in diag_df else 0,
       "limit_down_open_blocked": int(diag_df["limit_down_open_blocked"].sum()) if "limit_down_open_blocked" in diag_df else 0,
       "limit_up_touch_blocked": int(diag_df["limit_up_touch_blocked"].sum()) if "limit_up_touch_blocked" in diag_df else 0,
       "limit_down_touch_blocked": int(diag_df["limit_down_touch_blocked"].sum()) if "limit_down_touch_blocked" in diag_df else 0,
       "new_stock_buy_blocked": int(diag_df["new_stock_buy_blocked"].sum()) if "new_stock_buy_blocked" in diag_df else 0,
       "capped": int(diag_df["capped"].sum()) if "capped" in diag_df else 0,
       "lot_blocked": int(diag_df["lot_blocked"].sum()) if "lot_blocked" in diag_df else 0,
       "band_skipped": int(diag_df["band_skipped"].sum()) if "band_skipped" in diag_df else 0,
       "execution_lag": int(args.execution_lag),
       "lot_size": int(args.lot_size),
       "min_commission_cny": float(args.min_commission_cny),
       "rebalance_band": float(args.rebalance_band),
       "max_new_names": int(getattr(args, "max_new_names", 0)),
       "selection_policy": str(getattr(args, "selection_policy", "retention")),
       "top_k": int(getattr(args, "top_k", 0) or 0),
       "n_drop": int(getattr(args, "n_drop", 0) or 0),
       "risk_target_frac": (
           float(args.risk_target_frac)
           if getattr(args, "risk_target_frac", None) is not None
           else np.nan
       ),
       "risk_target_market_mult_below": float(
           getattr(args, "risk_target_market_mult_below", 1.0)
       ),
       "avg_effective_target_frac": (
           float(diag_df["effective_target_frac"].mean())
           if "effective_target_frac" in diag_df
           else float(target_frac)
       ),
       "exit_hold_frac": float(getattr(args, "exit_hold_frac", 0.0) or 0.0),
       "switch_gap_frac": float(getattr(args, "switch_gap_frac", 0.0) or 0.0),
       "min_buy_listing_days": int(getattr(args, "min_buy_listing_days", 0) or 0),
       "no_limit_first_trading_days": int(getattr(args, "no_limit_first_trading_days", 0) or 0),
    }


def run_open_ledger(
    alpha_rows,
    open_df,
    close_df,
    adv_df,
    target_frac,
    hold_frac,
    args,
    idx_close,
    idx_daily,
    execution_masks=None,
    prepared_context=None,
    execution_trace_sink=None,
    position_trace_sink=None,
):
    context = prepared_context or prepare_open_ledger_context(open_df, close_df)
    codes = context["codes"]
    code2idx = context["code2idx"]
    all_dates = context["all_dates"]
    n_codes, t_total = len(codes), len(all_dates)
    open_mark = context["open_mark"]
    close_ret_daily = context["close_ret_daily"]
    risk_cache = context["risk_cache"]

    row_by_day = {}
    for row in alpha_rows:
       pos = all_dates.searchsorted(row["date"], side="right") + max(int(args.execution_lag), 0)
       if 0 < pos < t_total:
           row_by_day[int(pos)] = row
    entry_days = sorted(row_by_day)
    if not entry_days:
       return {}, pd.DataFrame(), pd.DataFrame()

    current_shares = np.zeros(n_codes, dtype=np.float64)
    cash = float(args.portfolio_value)
    current_selected = []
    equity_curve = np.full(t_total, np.nan, dtype=np.float64)
    diag_rows = []
    closed_ages = []
    holding_ages = {}
    industry_map = {}
    max_industry_weight = float(getattr(args, "max_industry_weight", 0.0) or 0.0)
    if (
       max_industry_weight > 0
       or str(getattr(args, "state_aware_selection_mode", "none") or "none").strip().lower() != "none"
       or active_drawdown_condition_needs_industry(
           getattr(args, "active_drawdown_throttle_condition", "active_only")
       )
    ):
       industry_map = load_industry_map(getattr(args, "industry_csv", "data/stock_industry.csv"))
    market_args = SimpleNamespace(
       market_timing_mode=args.market_timing_mode,
       market_min_mult=args.market_min_mult,
       market_max_mult=args.market_max_mult,
       legacy_bear_mult=args.legacy_bear_mult,
       legacy_crash_mult=args.legacy_crash_mult,
    )
    active_drawdown_history = []
    active_drawdown_remaining = 0
    active_drawdown_remaining_scale = 1.0
    active_drawdown_steps = parse_active_drawdown_throttle_steps(
       getattr(args, "active_drawdown_throttle_steps", "")
    )
    global_risk_frame = getattr(args, "_global_risk_frame", None)
    if global_risk_frame is None and getattr(args, "global_risk_features", None):
       global_risk_frame = load_global_risk_features(
           getattr(args, "global_risk_features"),
           all_dates,
       )

    for day in range(t_total):
       marked_prices = open_mark[day]
       equity_before_trade = float(cash + np.dot(current_shares, marked_prices))
       if (
           day > 0
           and np.isfinite(equity_curve[day - 1])
           and equity_curve[day - 1] > 0
           and np.any(current_shares > 0)
       ):
           realized_return = equity_before_trade / float(equity_curve[day - 1]) - 1.0
           if idx_daily is not None:
               realized_benchmark = float(
                   idx_daily.reindex(pd.DatetimeIndex([all_dates[day]])).fillna(0.0).iloc[0]
               )
           else:
               realized_benchmark = 0.0
           active_drawdown_history.append(float(realized_return - realized_benchmark))
       current_weights_for_condition = (
           current_shares * marked_prices / max(equity_before_trade, 1.0)
       )
       active_condition_allowed, active_condition_diag = active_drawdown_condition_state(
           args,
           codes,
           current_weights_for_condition,
           industry_map,
           close_df,
           close_ret_daily,
           day,
       )
       active_drawdown_step = ""
       active_mode = str(
           getattr(args, "active_drawdown_throttle_mode", "fixed") or "fixed"
       ).strip().lower()
       if active_drawdown_steps:
           (
              active_throttle_scale,
              active_trailing_return,
              active_drawdown_remaining,
              active_drawdown_remaining_scale,
              active_drawdown_step,
           ) = active_drawdown_throttle_step_state(
              active_drawdown_history,
              getattr(args, "active_drawdown_throttle_lookback", 0),
              active_drawdown_steps,
              getattr(args, "active_drawdown_throttle_cooldown", 0),
              active_drawdown_remaining,
              active_drawdown_remaining_scale,
              trigger_allowed=active_condition_allowed,
           )
       elif active_mode == "continuous":
           (
              active_throttle_scale,
              active_trailing_return,
              active_drawdown_remaining,
              active_drawdown_remaining_scale,
           ) = active_drawdown_throttle_continuous_state(
              active_drawdown_history,
              getattr(args, "active_drawdown_throttle_lookback", 0),
              getattr(args, "active_drawdown_throttle_trigger", 0.0),
              getattr(args, "active_drawdown_throttle_scale", 1.0),
              getattr(args, "active_drawdown_throttle_continuous_width", 0.0),
              getattr(args, "active_drawdown_throttle_cooldown", 0),
              active_drawdown_remaining,
              active_drawdown_remaining_scale,
              trigger_allowed=active_condition_allowed,
           )
       else:
           active_throttle_scale, active_trailing_return, active_drawdown_remaining = (
              active_drawdown_throttle_state(
                  active_drawdown_history,
                  getattr(args, "active_drawdown_throttle_lookback", 0),
                  getattr(args, "active_drawdown_throttle_trigger", 0.0),
                  getattr(args, "active_drawdown_throttle_scale", 1.0),
                  getattr(args, "active_drawdown_throttle_cooldown", 0),
                  active_drawdown_remaining,
                  trigger_allowed=active_condition_allowed,
              )
           )
           if active_drawdown_remaining <= 0:
              active_drawdown_remaining_scale = 1.0
       if not np.isfinite(active_trailing_return):
           observation_lookback = max(
               int(getattr(args, "active_drawdown_throttle_lookback", 0) or 0),
               int(getattr(args, "active_drawdown_observation_lookback", 20) or 0),
           )
           active_trailing_return = active_drawdown_observation(
               active_drawdown_history,
               observation_lookback,
           )
       row = row_by_day.get(day)
       if row is not None:
           market_mult = 1.0
           if args.market_timing_mode != "none":
               market_mult = compute_market_multiplier(
                   idx_close,
                   idx_daily,
                   close_ret_daily,
                   max(day - 1, 0),
                   market_args.market_timing_mode,
                   market_args.market_min_mult,
                   market_args.market_max_mult,
                   market_args.legacy_bear_mult,
                   market_args.legacy_crash_mult,
               )
           if getattr(args, "use_row_market_mult", False):
               for transform_key in (
                   "breadth_market_transform",
                   "state_market_transform",
               ):
                   transform = row.get(transform_key)
                   if isinstance(transform, dict) and transform.get("triggered"):
                       row_mult = transform.get("effective_market_mult")
                       if row_mult is not None:
                           market_mult = min(float(market_mult), float(row_mult))
           base_market_mult = float(market_mult)
           market_mult = float(market_mult) * float(active_throttle_scale)
           effective_target_frac = float(target_frac)
           global_risk_state = global_risk_overlay_state(
               args,
               global_risk_frame,
               all_dates[day],
           )
           risk_key = (id(idx_daily), int(day), 60)
           if risk_key not in risk_cache:
               risk_cache[risk_key] = estimate_trailing_stock_risk(
                   close_ret_daily,
                   idx_daily,
                   day,
                   window=60,
               )
           stock_beta, residual_vol, risk_obs = risk_cache[risk_key]
           market_mult = float(market_mult) * float(
               global_risk_state["global_risk_market_scale"]
           )
           if np.isfinite(global_risk_state["global_risk_target_frac"]):
               effective_target_frac = min(
                   effective_target_frac,
                   float(global_risk_state["global_risk_target_frac"]),
               )
           risk_target_frac = getattr(args, "risk_target_frac", None)
           if (
               risk_target_frac is not None
               and market_mult < float(getattr(args, "risk_target_market_mult_below", 1.0))
           ):
               effective_target_frac = min(float(target_frac), float(risk_target_frac))
           if getattr(args, "use_row_target_frac", False):
               for transform_key in (
                   "breadth_target_transform",
                   "state_target_transform",
               ):
                   transform = row.get(transform_key)
                   if isinstance(transform, dict) and transform.get("triggered"):
                       row_target = transform.get("effective_target_frac")
                       if row_target is not None:
                           effective_target_frac = min(
                               float(effective_target_frac),
                               float(row_target),
                           )
           selection_policy = str(getattr(args, "selection_policy", "retention"))
           proposal = select_target_policy(
               row["codes"],
               current_selected,
               policy=selection_policy,
               target_frac=effective_target_frac,
               hold_frac=hold_frac,
               top_k=getattr(args, "top_k", 0),
               n_drop=getattr(args, "n_drop", 0),
           )
           selected = list(proposal.selected)
           kept = list(proposal.retained)
           target_n = proposal.target_n
           hold_n = max(target_n, int(len(row["codes"]) * float(hold_frac)))
           if selection_policy == "retention":
               selected = limit_new_names(
                   selected,
                   kept,
                   row,
                   max(int(getattr(args, "max_new_names", 0)), 0),
                   current_selected,
                   target_n,
                   getattr(args, "exit_hold_frac", None),
                   getattr(args, "switch_gap_frac", 0.0),
                   getattr(args, "max_new_names_mode", "legacy"),
               )
           selected_before_state = list(selected)
           selected, state_selection_diag = apply_state_aware_selection_rank(
               selected,
               row,
               current_selected,
               target_n,
               codes,
               code2idx,
               close_df,
               close_ret_daily,
               idx_daily,
               day,
               stock_beta,
               residual_vol,
               industry_map,
               global_risk_frame,
               args,
           )
           selected_before_industry_cap = list(selected)
           selected, industry_cap_diag = apply_industry_selection_cap(
               selected,
               row.get("codes", []),
               industry_map,
               max_industry_weight,
               market_mult,
               args.max_weight,
               target_n,
           )
           weighting_mode = getattr(args, "weighting_mode", "equal")
           if weighting_mode == "score":
               score_by_code = {}
               if "calibrated_bps" in row:
                   score_by_code = {c: s for c, s in zip(row.get("codes", []), row.get("calibrated_bps", []))}
               desired = weights_from_selected_scores(selected, code2idx, n_codes, score_by_code, market_mult, args.max_weight)
           else:
               desired = weights_from_selected(selected, code2idx, n_codes, market_mult, args.max_weight)
           # Defensive tilt can be global or gated to weak market states.
           defensive_tilt = getattr(args, "defensive_tilt", 0.0)
           defensive_tilt_market_mult_below = getattr(
               args,
               "defensive_tilt_market_mult_below",
               1.0,
           )
           if should_apply_defensive_tilt(
               defensive_tilt,
               market_mult,
               defensive_tilt_market_mult_below,
           ):
               desired = apply_defensive_tilt(desired, selected, code2idx, all_dates[day], close_df, idx_close, defensive_tilt)
           execution_result = apply_open_ledger_constraints(
               desired,
               current_shares,
               cash,
               equity_before_trade,
               open_df,
               close_df,
               adv_df,
               day,
               args,
               execution_masks=execution_masks,
               capture_trace=execution_trace_sink is not None,
           )
           if execution_trace_sink is None:
               new_shares, cash, executed_shares, exec_info = execution_result
           else:
               new_shares, cash, executed_shares, exec_info, execution_trace = execution_result
               execution_date = str(pd.Timestamp(all_dates[day]).date())
               for trace_row in execution_trace:
                   item = dict(trace_row)
                   item["date"] = execution_date
                   item["code"] = codes[int(item.pop("asset_index"))]
                   execution_trace_sink.append(item)
           live_idx = np.where(new_shares >= max(int(args.lot_size), 1))[0]
           live_codes = [codes[i] for i in live_idx]
           prev_set = set(current_selected)
           live_set = set(live_codes)
           for code in prev_set - live_set:
               closed_ages.append(holding_ages.get(code, 1))
               holding_ages.pop(code, None)
           for code in live_codes:
               holding_ages[code] = holding_ages.get(code, 0) + 1
           current_selected = live_codes
           current_shares = new_shares
           equity_after_trade = float(cash + np.dot(current_shares, marked_prices))
           invested_value = float(np.dot(current_shares, marked_prices))
           actual_weights = (
               current_shares * marked_prices / max(equity_after_trade, 1.0)
           )
           holdings = encode_holdings(codes, actual_weights)
           if position_trace_sink is not None:
               position_date = str(pd.Timestamp(all_dates[day]).date())
               for asset_index in live_idx:
                   position_trace_sink.append({
                       "date": position_date,
                       "code": codes[int(asset_index)],
                       "shares": float(current_shares[asset_index]),
                       "mark_price": float(marked_prices[asset_index]),
                       "market_value_cny": float(current_shares[asset_index] * marked_prices[asset_index]),
                       "weight": float(actual_weights[asset_index]),
                   })
           risk_diag = summarize_portfolio_risk(
               actual_weights,
               stock_beta,
               residual_vol,
           )
           diag_rows.append({
               "day": int(day),
               "date": str(all_dates[day]),
               "effective_target_frac": float(effective_target_frac),
               "target_n": int(target_n),
               "hold_n": int(hold_n),
               "kept_n": int(len(kept)),
               "selected_n": int(len(live_codes)),
               "desired_selected_n": int(len(selected)),
               "pre_industry_cap_selected_n": int(len(selected_before_industry_cap)),
               **industry_cap_diag,
               "max_new_names": int(getattr(args, "max_new_names", 0)),
               "max_new_names_mode": getattr(args, "max_new_names_mode", "legacy"),
               "exit_hold_frac": float(getattr(args, "exit_hold_frac", 0.0) or 0.0),
               "switch_gap_frac": float(getattr(args, "switch_gap_frac", 0.0) or 0.0),
               "desired_new_names": int(len(set(selected) - prev_set)),
               "current_selected_codes": ",".join(sorted(prev_set)),
               "selected_before_state_codes": ",".join(selected_before_state),
               "selected_after_state_codes": ",".join(selected),
               "retained_codes": ",".join(kept),
               **state_selection_diag,
               "gross_weight": invested_value / max(equity_after_trade, 1.0),
               "holdings": holdings,
               "cash_cny": float(cash),
               "equity_cny": equity_after_trade,
               "base_market_mult": float(base_market_mult),
               "market_mult": float(market_mult),
               "active_drawdown_throttle_scale": float(active_throttle_scale),
               "active_drawdown_throttle_step": active_drawdown_step,
               "active_drawdown_trailing_return": float(active_trailing_return)
               if np.isfinite(active_trailing_return)
               else np.nan,
               "active_drawdown_cooldown_remaining": int(active_drawdown_remaining),
               **active_condition_diag,
               **global_risk_state,
               "risk_obs_60d": int(risk_obs),
               **risk_diag,
               "avg_live_age": float(np.mean(list(holding_ages.values()))) if holding_ages else 0.0,
               **exec_info,
           })
       equity_curve[day] = float(cash + np.dot(current_shares, marked_prices))
       if active_drawdown_remaining > 0:
           active_drawdown_remaining -= 1
           if active_drawdown_remaining <= 0:
              active_drawdown_remaining_scale = 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
       returns = equity_curve[1:] / equity_curve[:-1] - 1.0
    returns[~np.isfinite(returns)] = 0.0
    start_idx = max(entry_days[0] - 1, 0)
    last_return_day = min(entry_days[-1] + 1, t_total - 1)
    returns_active = returns[start_idx:last_return_day]
    active_dates = all_dates[1:][start_idx:start_idx + len(returns_active)]
    if idx_daily is not None:
       benchmark_returns = (
           idx_daily.reindex(active_dates).fillna(0.0).to_numpy(dtype=np.float64)
       )
    else:
       benchmark_returns = np.zeros(len(returns_active), dtype=np.float64)
    benchmark_returns = benchmark_returns[:len(returns_active)]
    active_return = returns_active[:len(benchmark_returns)] - benchmark_returns

    closed_ages.extend(holding_ages.values())
    diag_df = pd.DataFrame(diag_rows)
    row = summarize_open_ledger_result(
       returns_active,
       diag_df,
       closed_ages,
       target_frac,
       hold_frac,
       args,
       benchmark_returns=benchmark_returns,
    )
    returns_df = pd.DataFrame({
       "date": active_dates,
       "return": returns_active,
       "benchmark_return": benchmark_returns,
       "active_return": active_return,
       "equity_cny": equity_curve[1:][start_idx:start_idx + len(returns_active)],
    })
    return row, returns_df, diag_df

def weights_from_selected_scores(selected, code2idx, n_codes, score_by_code, gross_weight, max_weight):
    weights = np.zeros(n_codes, dtype=np.float64)
    valid_indices = []
    scores = []
    for code in dict.fromkeys(selected):
        if code in code2idx and code in score_by_code:
            valid_indices.append(code2idx[code])
            scores.append(max(score_by_code[code], 0.0))
    if not valid_indices or sum(scores) <= 0:
        return weights_from_selected(selected, code2idx, n_codes, gross_weight, max_weight)
    scores_arr = np.array(scores, dtype=np.float64)
    raw = scores_arr / scores_arr.sum() * max(gross_weight, 0.0)
    capped = np.minimum(raw, max(max_weight, 0.0))
    excess = max(gross_weight, 0.0) - capped.sum()
    if excess > 1e-10:
        uncapped = capped < max_weight
        if uncapped.any():
            capped[uncapped] += excess * scores_arr[uncapped] / scores_arr[uncapped].sum()
            capped = np.minimum(capped, max_weight)
    for idx, w in zip(valid_indices, capped):
        weights[idx] = w
    return weights
