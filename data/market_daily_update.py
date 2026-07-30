"""Resumable daily ingestion into the transactional MarketDailyStore."""

from __future__ import annotations

import json
import msvcrt
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from data.api_utils import SafeAPICaller
from data.market_daily_store import MarketDailyStore, validate_market_daily_frame


FORMAL_BROAD_INDEX_CODES = (
    "000016.SH",
    "000300.SH",
    "000905.SH",
    "399006.SZ",
)
AKSHARE_BROAD_INDEX_SYMBOLS = {
    "000016.SH": "sh000016",
    "000300.SH": "sh000300",
    "000905.SH": "sh000905",
    "399006.SZ": "sz399006",
}


def _date_key(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


@contextmanager
def update_progress_lock(progress_path: str | Path):
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
        raise RuntimeError(f"another market-daily update is using {progress_path}") from exc
    try:
        yield
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def _normalize_tushare_frame(
    frame: pd.DataFrame,
    *,
    instrument_type: str,
    source: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"{source} returned no rows")
    renamed = frame.rename(
        columns={
            "ts_code": "code",
            "vol": "volume",
            "amount": "money",
        }
    ).copy()
    if "factor" not in renamed:
        renamed["factor"] = 1.0
    return validate_market_daily_frame(
        renamed,
        instrument_type=instrument_type,
        source=source,
    )


class TushareMarketDailyClient:
    """Fetch one A-share session and the four formal broad indices."""

    def __init__(self, token: str):
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
        self.equity_source = "tushare.daily"
        self.index_source = "tushare.index_daily"
        self.index_money_semantics = "tushare_amount_thousand_cny"

    def trade_dates(self, start_date: str, end_date: str) -> list[str]:
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

    def equity_daily(self, trade_date: str) -> pd.DataFrame:
        return self.caller(
            self.pro.daily,
            trade_date=trade_date,
            fields="ts_code,trade_date,open,high,low,close,vol,amount",
        )

    def index_daily(
        self,
        trade_date: str,
        codes: Sequence[str] = FORMAL_BROAD_INDEX_CODES,
    ) -> pd.DataFrame:
        pieces = []
        for code in codes:
            frame = self.caller(
                self.pro.index_daily,
                ts_code=code,
                start_date=trade_date,
                end_date=trade_date,
                fields="ts_code,trade_date,open,high,low,close,vol,amount",
            )
            if frame is None or frame.empty:
                raise ValueError(f"Tushare index_daily returned no row for {code}:{trade_date}")
            pieces.append(frame)
        return pd.concat(pieces, ignore_index=True)


class AkshareBroadIndexClient:
    """Fetch the four broad indices from the project's established AkShare API."""

    index_source = "akshare.stock_zh_index_daily"
    index_money_semantics = "source_unavailable_filled_zero"

    def __init__(self):
        self.caller = SafeAPICaller(
            min_interval=0.5,
            max_retries=3,
            retry_base_delay=4.0,
            jitter=(0.1, 0.2),
            data_source="akshare",
        )

    def index_daily(
        self,
        trade_date: str,
        codes: Sequence[str] = FORMAL_BROAD_INDEX_CODES,
    ) -> pd.DataFrame:
        import akshare as ak

        target = pd.Timestamp(trade_date).normalize()
        pieces = []
        for code in codes:
            symbol = AKSHARE_BROAD_INDEX_SYMBOLS.get(code)
            if symbol is None:
                raise ValueError(f"no AkShare symbol mapping for broad index {code}")
            frame = self.caller(ak.stock_zh_index_daily, symbol=symbol)
            if frame is None or frame.empty:
                raise ValueError(f"AkShare returned no history for {code}")
            frame = frame.copy()
            dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            selected = frame.loc[dates == target].copy()
            if selected.empty:
                raise ValueError(f"AkShare returned no row for {code}:{trade_date}")
            selected["trade_date"] = dates.loc[selected.index]
            selected["code"] = code
            selected["money"] = 0.0
            selected["factor"] = 1.0
            pieces.append(selected)
        return pd.concat(pieces, ignore_index=True)


class CompositeMarketDailyClient:
    """Use one calendar/equity client and an independently replaceable index client."""

    def __init__(self, equity_client: Any, index_client: Any):
        self.equity_client = equity_client
        self.index_client = index_client
        self.equity_source = getattr(equity_client, "equity_source", "tushare.daily")
        self.index_source = getattr(index_client, "index_source", "market.index_daily")
        self.index_money_semantics = getattr(
            index_client,
            "index_money_semantics",
            "source_defined",
        )

    def trade_dates(self, start_date: str, end_date: str) -> list[str]:
        return self.equity_client.trade_dates(start_date, end_date)

    def equity_daily(self, trade_date: str) -> pd.DataFrame:
        return self.equity_client.equity_daily(trade_date)

    def index_daily(self, trade_date: str, codes: Sequence[str]) -> pd.DataFrame:
        return self.index_client.index_daily(trade_date, codes)


def _load_progress(
    path: Path,
    *,
    store_root: Path,
    dates: Sequence[str],
    include_indices: bool,
    index_codes: Sequence[str],
    equity_source: str,
    index_source: str | None,
    index_money_semantics: str | None,
) -> dict[str, Any]:
    expected = {
        "store_root": str(store_root),
        "requested_dates": list(dates),
        "include_indices": bool(include_indices),
        "index_codes": list(index_codes) if include_indices else [],
        "equity_source": equity_source,
        "index_source": index_source if include_indices else None,
        "index_money_semantics": index_money_semantics if include_indices else None,
    }
    if not path.is_file():
        return {
            "schema": "market_daily_incremental_update_v1",
            **expected,
            "status": "running",
            "completed_dates": [],
            "dates": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"market-daily progress {key} does not match request")
    return payload


def run_incremental_update(
    *,
    client: Any,
    store_root: str | Path,
    dates: Sequence[Any],
    progress_path: str | Path,
    min_equity_rows: int = 3000,
    include_indices: bool = True,
    index_codes: Sequence[str] = FORMAL_BROAD_INDEX_CODES,
    allow_revision: bool = False,
) -> dict[str, Any]:
    store_root = Path(store_root).resolve()
    progress_path = Path(progress_path).resolve()
    normalized_dates = sorted({_date_key(value) for value in dates})
    if not normalized_dates:
        raise ValueError("at least one trade date is required")
    index_codes = tuple(str(code).strip().upper() for code in index_codes)
    equity_source = str(getattr(client, "equity_source", "tushare.daily"))
    index_source = str(getattr(client, "index_source", "tushare.index_daily"))
    index_money_semantics = str(
        getattr(client, "index_money_semantics", "source_defined")
    )

    with update_progress_lock(progress_path):
        progress = _load_progress(
            progress_path,
            store_root=store_root,
            dates=normalized_dates,
            include_indices=include_indices,
            index_codes=index_codes,
            equity_source=equity_source,
            index_source=index_source,
            index_money_semantics=index_money_semantics,
        )
        store = MarketDailyStore(store_root)
        completed = set(progress["completed_dates"])
        for trade_date in normalized_dates:
            if trade_date in completed:
                continue
            date_record = progress["dates"].setdefault(trade_date, {})
            try:
                equity = _normalize_tushare_frame(
                    client.equity_daily(trade_date),
                    instrument_type="equity",
                    source=equity_source,
                )
                if len(equity) < int(min_equity_rows):
                    raise ValueError(
                        f"Tushare returned only {len(equity)} equity rows for {trade_date}; "
                        f"minimum is {min_equity_rows}"
                    )
                index = None
                if include_indices:
                    index = _normalize_tushare_frame(
                        client.index_daily(trade_date, index_codes),
                        instrument_type="index",
                        source=index_source,
                    )
                    missing = sorted(set(index_codes) - set(index["code"]))
                    extra = sorted(set(index["code"]) - set(index_codes))
                    if missing or extra:
                        raise ValueError(
                            f"broad-index coverage mismatch; missing={missing}, extra={extra}"
                        )

                equity_result = store.commit_partition(
                    equity,
                    instrument_type="equity",
                    source=equity_source,
                    allow_revision=allow_revision,
                )
                date_record["equity"] = {
                    "status": equity_result.status,
                    "rows": equity_result.row_count,
                    "logical_sha256": equity_result.logical_sha256,
                }
                _write_json_atomic(progress_path, progress)

                if index is not None:
                    index_result = store.commit_partition(
                        index,
                        instrument_type="index",
                        source=index_source,
                        allow_revision=allow_revision,
                    )
                    date_record["index"] = {
                        "status": index_result.status,
                        "rows": index_result.row_count,
                        "logical_sha256": index_result.logical_sha256,
                        "codes": list(index_codes),
                        "money_semantics": index_money_semantics,
                    }
                    _write_json_atomic(progress_path, progress)

                date_record["status"] = "completed"
                date_record.pop("error", None)
                progress["completed_dates"].append(trade_date)
                progress["completed_dates"].sort()
                completed.add(trade_date)
                _write_json_atomic(progress_path, progress)
            except Exception as exc:
                date_record["status"] = "failed"
                date_record["error"] = f"{type(exc).__name__}: {exc}"
                progress["status"] = "failed"
                _write_json_atomic(progress_path, progress)
                raise

        progress["status"] = "completed"
        progress["active_store"] = store.active_state()
        coverage = progress["active_store"]["coverage"]
        if include_indices:
            equity_end = coverage.get("equity", {}).get("date_end")
            index_end = coverage.get("index", {}).get("date_end")
            progress["coverage_alignment"] = {
                "equity_date_end": equity_end,
                "index_date_end": index_end,
                "latest_dates_aligned": equity_end == index_end,
                "gap_reason": None if equity_end == index_end else "source_or_partial_update_gap",
            }
        _write_json_atomic(progress_path, progress)
        return progress
