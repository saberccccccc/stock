"""Download and freeze point-in-time Tushare ST status events for research."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from data.api_utils import SafeAPICaller, resolve_tushare_token
from core.research_protocol import RESEARCH_END_DATE
from data.st_status import (
    file_sha256,
    normalize_namechange_events,
    normalize_st_events,
    normalize_ts_code,
    parse_event_date,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Output root; defaults to data/raw for research or data/forward_raw for forward.",
    )
    parser.add_argument("--source-cache", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--endpoint", choices=("st", "namechange"), default="st")
    parser.add_argument("--dataset-role", choices=("research", "forward"), default="research")
    parser.add_argument("--output-name", default="st_status_events.csv")
    parser.add_argument("--manifest-name", default="st_status_events_manifest.json")
    parser.add_argument("--as-of-date", default=str(RESEARCH_END_DATE.date()))
    parser.add_argument(
        "--source-start-date",
        default="19900101",
        help="Earliest source interval/announcement date to request for namechange.",
    )
    parser.add_argument(
        "--ts-code-file",
        default="data/raw/stable_stocks.csv",
        help="CSV containing ts_code values for the official st-by-code fetch.",
    )
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--min-interval", type=float, default=1.0)
    parser.add_argument("--token", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def validate_dataset_role(as_of_date, dataset_role, research_cutoff=None):
    """Enforce the frozen research/forward date split before any API call."""
    as_of = parse_event_date(as_of_date)
    cutoff = parse_event_date(research_cutoff or RESEARCH_END_DATE)
    if pd.isna(as_of):
        raise ValueError(f"invalid --as-of-date: {as_of_date}")
    if dataset_role == "research" and as_of > cutoff:
        raise ValueError(
            f"research dataset cannot exceed frozen cutoff {cutoff.date()}: {as_of.date()}"
        )
    if dataset_role == "forward" and as_of <= cutoff:
        raise ValueError(
            f"forward dataset must be after frozen cutoff {cutoff.date()}: {as_of.date()}"
        )
    return as_of


def fetch_all_events(
    pro,
    caller,
    *,
    endpoint_name="st",
    page_size=1000,
    max_pages=100,
    checkpoint_dir=None,
    resume=False,
):
    """Fetch a paginated Tushare endpoint without printing credentials."""
    if page_size <= 0 or max_pages <= 0:
        raise ValueError("page_size and max_pages must be positive")
    endpoint = getattr(pro, endpoint_name, None)
    pages = []
    offset = 0
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else None
    checkpoint_meta = checkpoint_root / "checkpoint.json" if checkpoint_root else None
    if resume and checkpoint_root and checkpoint_meta and checkpoint_meta.is_file():
        metadata = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
        if metadata.get("endpoint", endpoint_name) != endpoint_name:
            raise ValueError("Tushare checkpoint endpoint does not match the requested endpoint")
        if int(metadata.get("page_size", page_size)) != int(page_size):
            raise ValueError("Tushare checkpoint page_size does not match the requested page_size")
        for page_path in sorted(checkpoint_root.glob("page_*.csv")):
            page = _load_cache(page_path)
            if page is not None and not page.empty:
                pages.append(page)
                offset += len(page)
        if metadata.get("complete") and pages:
            return pd.concat(pages, ignore_index=True), len(pages)
    for _ in range(max_pages):
        if endpoint is not None:
            frame = caller(endpoint, offset=offset, limit=page_size)
        else:
            query = getattr(pro, "query", None)
            if query is None:
                raise AttributeError(f"Tushare client exposes neither {endpoint_name} nor query")
            frame = caller(query, endpoint_name, offset=offset, limit=page_size)
        if frame is None:
            raise RuntimeError(f"Tushare returned no response for {endpoint_name} page offset={offset}")
        frame = pd.DataFrame(frame)
        if frame.empty:
            break
        pages.append(frame)
        offset += len(frame)
        if checkpoint_root:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            _atomic_write_csv(frame, checkpoint_root / f"page_{len(pages) - 1:05d}.csv")
            _atomic_write_json(
                {
                    "schema_version": 1,
                    "endpoint": endpoint_name,
                    "page_size": int(page_size),
                    "page_count": len(pages),
                    "next_offset": offset,
                    "complete": len(frame) < page_size,
                },
                checkpoint_meta,
            )
        if len(frame) < page_size:
            break
    else:
        raise RuntimeError(f"{endpoint_name} pagination reached max_pages; refusing a partial download")
    if not pages:
        return pd.DataFrame(), 0
    return pd.concat(pages, ignore_index=True), len(pages)


def fetch_all_st_events(pro, caller, **kwargs):
    return fetch_all_events(pro, caller, endpoint_name="st", **kwargs)


def fetch_all_namechange_events(pro, caller, **kwargs):
    return fetch_all_events(pro, caller, endpoint_name="namechange", **kwargs)


def _normalize_code_list(ts_codes):
    normalized = []
    seen = set()
    for value in ts_codes or []:
        code = normalize_ts_code(value)
        if code and code not in seen:
            normalized.append(code)
            seen.add(code)
    return normalized


def load_ts_codes(path):
    """Load and normalize a frozen code universe for the st-by-code fetch."""
    frame = _load_cache(path)
    if frame is None or frame.empty:
        raise ValueError(f"ts-code file is missing or empty: {path}")
    for column in ("ts_code", "code", "symbol"):
        if column in frame.columns:
            codes = _normalize_code_list(frame[column].tolist())
            if codes:
                return codes
    raise ValueError(f"ts-code file has no usable code column: {path}")


def fetch_st_events_by_codes(
    pro,
    caller,
    *,
    ts_codes,
    checkpoint_dir=None,
    resume=False,
):
    """Fetch the official ``st`` event feed one stock code at a time."""
    codes = _normalize_code_list(ts_codes)
    if not codes:
        raise ValueError("ts_codes must contain at least one usable stock code")
    endpoint = getattr(pro, "st", None)
    if endpoint is None:
        raise AttributeError("Tushare client exposes no st endpoint")

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else None
    checkpoint_meta = checkpoint_root / "checkpoint.json" if checkpoint_root else None
    code_fingerprint = hashlib.sha256("\n".join(codes).encode("utf-8")).hexdigest()
    if resume and checkpoint_meta and checkpoint_meta.is_file():
        metadata = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
        if metadata.get("endpoint") != "st_by_ts_code":
            raise ValueError("ST checkpoint endpoint does not match code fetch mode")
        if metadata.get("code_fingerprint") != code_fingerprint:
            raise ValueError("ST checkpoint code universe does not match the request")

    frames = []
    for index, code in enumerate(codes):
        code_path = checkpoint_root / f"code_{index:05d}.csv" if checkpoint_root else None
        if resume and code_path and code_path.is_file():
            frame = _load_cache(code_path)
        else:
            frame = caller(endpoint, ts_code=code)
            if frame is None:
                raise RuntimeError(f"Tushare returned no response for st ts_code={code}")
            frame = pd.DataFrame(frame)
            if code_path:
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                _atomic_write_csv(frame, code_path)
        if frame is not None and not frame.empty:
            frames.append(pd.DataFrame(frame))
        if checkpoint_meta:
            _atomic_write_json(
                {
                    "schema_version": 1,
                    "endpoint": "st_by_ts_code",
                    "code_fingerprint": code_fingerprint,
                    "code_count": len(codes),
                    "processed_count": index + 1,
                    "complete": index + 1 == len(codes),
                },
                checkpoint_meta,
            )
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result, len(codes)


def fetch_namechange_intervals(
    pro,
    caller,
    *,
    start_date,
    end_date,
    checkpoint_dir=None,
    resume=False,
):
    """Fetch ``namechange`` with its documented date-range parameters."""
    start = parse_event_date(start_date)
    end = parse_event_date(end_date)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("namechange source date range is invalid")
    endpoint = getattr(pro, "namechange", None)
    if endpoint is None:
        raise AttributeError("Tushare client exposes no namechange endpoint")

    start_text = start.strftime("%Y%m%d")
    end_text = end.strftime("%Y%m%d")
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else None
    checkpoint_meta = checkpoint_root / "checkpoint.json" if checkpoint_root else None
    page_path = checkpoint_root / "page_00000.csv" if checkpoint_root else None
    if resume and checkpoint_meta and checkpoint_meta.is_file() and page_path.is_file():
        metadata = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
        if (
            metadata.get("endpoint") != "namechange_date_range"
            or metadata.get("start_date") != start_text
            or metadata.get("end_date") != end_text
        ):
            raise ValueError("namechange checkpoint request does not match the request")
        return _load_cache(page_path), 1

    frame = caller(endpoint, start_date=start_text, end_date=end_text)
    if frame is None:
        raise RuntimeError(
            f"Tushare returned no response for namechange {start_text}-{end_text}"
        )
    frame = pd.DataFrame(frame)
    if checkpoint_root:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_csv(frame, page_path)
        _atomic_write_json(
            {
                "schema_version": 1,
                "endpoint": "namechange_date_range",
                "start_date": start_text,
                "end_date": end_text,
                "complete": True,
            },
            checkpoint_meta,
        )
    return frame, 1


def _atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_cache(path):
    path = Path(path)
    if not path.is_file():
        return None
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except (OSError, UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"unable to read ST source cache: {path}")


def main(argv=None):
    args = parse_args(argv)
    as_of = validate_dataset_role(args.as_of_date, args.dataset_role)
    data_dir_arg = args.data_dir or (
        "data/raw" if args.dataset_role == "research" else "data/forward_raw"
    )
    data_dir = ROOT / data_dir_arg
    output = data_dir / args.output_name
    manifest_path = data_dir / args.manifest_name
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite frozen ST event file: {output}")
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite frozen ST manifest: {manifest_path}")

    tracking_root = (
        "data/tracking_raw"
        if args.dataset_role == "research"
        else "data/forward_tracking_raw"
    )
    source_cache_arg = args.source_cache or f"{tracking_root}/{args.endpoint}_source.csv"
    checkpoint_arg = args.checkpoint_dir or f"{tracking_root}/{args.endpoint}_pages"
    checkpoint_dir = ROOT / checkpoint_arg
    source_cache = ROOT / source_cache_arg
    raw = _load_cache(source_cache) if args.resume else None
    page_count = 0
    fetched_at = datetime.now(timezone.utc).isoformat()
    if raw is None:
        import tushare as ts

        token = resolve_tushare_token(args.token, context="historical ST events")
        ts.set_token(token)
        pro = ts.pro_api()
        caller = SafeAPICaller(
            min_interval=args.min_interval,
            max_retries=3,
            retry_base_delay=4.0,
            jitter=(0.2, 0.5),
            data_source=f"tushare.{args.endpoint}",
            non_retryable_markers=(
                "permission",
                "权限不足",
                "没有权限",
                "积分不足",
                "forbidden",
                "unauthorized",
                "invalid token",
                "token无效",
            ),
        )
        if args.endpoint == "namechange":
            raw, page_count = fetch_namechange_intervals(
                pro,
                caller,
                start_date=args.source_start_date,
                end_date=as_of.strftime("%Y%m%d"),
                checkpoint_dir=checkpoint_dir,
                resume=args.resume,
            )
        else:
            ts_code_file = ROOT / args.ts_code_file
            ts_codes = load_ts_codes(ts_code_file)
            raw, page_count = fetch_st_events_by_codes(
                pro,
                caller,
                ts_codes=ts_codes,
                checkpoint_dir=checkpoint_dir,
                resume=args.resume,
            )
        _atomic_write_csv(raw, source_cache)
    else:
        ts_code_file = ROOT / args.ts_code_file
        ts_codes = load_ts_codes(ts_code_file) if args.endpoint == "st" else None

    normalizer = normalize_st_events if args.endpoint == "st" else normalize_namechange_events
    normalized = normalizer(
        raw,
        as_of_date=as_of,
        source_endpoint=f"tushare.{args.endpoint}",
        source_fetched_at=fetched_at,
        strict=False,
    )
    invalid_count = int(normalized.attrs.get("invalid_row_count", 0))
    if invalid_count:
        raise ValueError(
            f"{args.endpoint} source contains {invalid_count} undecidable rows; refusing to publish research data"
        )
    normalized.attrs.clear()
    _atomic_write_csv(normalized, output)
    output_hash = file_sha256(output)
    event_dates = pd.to_datetime(normalized["event_date"], errors="coerce")
    manifest = {
        "schema_version": 1,
        "dataset_role": args.dataset_role,
        "selection_allowed": args.dataset_role == "research",
        "research_cutoff": str(RESEARCH_END_DATE.date()),
        "data_dir": str(data_dir.resolve()),
        "source_endpoint": f"tushare.{args.endpoint}",
        "source_kind": "tushare_event_history" if args.endpoint == "st" else "tushare_namechange_intervals",
        "source_label": (
            "direct_historical_st_events"
            if args.endpoint == "st"
            else "由历史股票名称区间重建"
        ),
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "coverage_start": event_dates.min().strftime("%Y-%m-%d") if not event_dates.empty else None,
        "coverage_end": as_of.strftime("%Y-%m-%d"),
        "raw_row_count": int(len(raw)),
        "normalized_row_count": int(len(normalized)),
        "code_count": int(normalized["ts_code"].nunique()) if not normalized.empty else 0,
        "invalid_row_count": invalid_count,
        "page_count": int(page_count),
        "page_size": int(args.page_size),
        "fetch_mode": (
            "namechange_date_range" if args.endpoint == "namechange" else "st_by_ts_code"
        ),
        "source_start_date": args.source_start_date if args.endpoint == "namechange" else None,
        "ts_code_file": str((ROOT / args.ts_code_file).resolve()) if args.endpoint == "st" else None,
        "ts_code_count": int(len(ts_codes)) if args.endpoint == "st" else None,
        "source_cache": str(source_cache.resolve()),
        "source_cache_sha256": file_sha256(source_cache) if source_cache.exists() else None,
        "output": str(output.resolve()),
        "output_sha256": output_hash,
        "fetched_at": fetched_at,
        "note": (
            "Research copy is filtered by imp_date <= as_of_date. Future source rows, if any, "
            "remain outside data/raw in the tracking source cache. For namechange, status is "
            "reconstructed from displayed-name intervals and is not a direct st event feed."
        ),
    }
    _atomic_write_json(manifest, manifest_path)
    print(json.dumps({"output": str(output), "rows": len(normalized), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
