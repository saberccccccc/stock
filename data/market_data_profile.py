"""Read-only inventory and runtime profiling for daily market-data stores."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from backtest.ohlc_matrix_cache import (
    RAW_FIELDS,
    discover_stock_csvs,
    load_ohlc_matrix_meta,
    matrix_cache_is_current,
    source_signature,
)
from data.forward_daily_update import last_trade_date


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short").splitlines(),
    }


def _memory_snapshot() -> dict[str, Any]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "total_bytes": int(memory.total),
            "available_bytes": int(memory.available),
            "percent": float(memory.percent),
        }
    except ImportError:
        return {"unavailable": "psutil_not_installed"}


def inventory_csv_root(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    started = time.perf_counter()
    paths = discover_stock_csvs(root)
    discovery_seconds = time.perf_counter() - started

    started = time.perf_counter()
    signature = source_signature(paths)
    signature_seconds = time.perf_counter() - started

    started = time.perf_counter()
    last_dates = Counter()
    for csv_path in paths:
        value = last_trade_date(csv_path)
        last_dates[str(value.date()) if value is not None else "missing"] += 1
    tail_scan_seconds = time.perf_counter() - started

    return {
        "root": str(root),
        "stock_files": len(paths),
        "bytes": int(sum(item.stat().st_size for item in paths)),
        "signature": signature,
        "latest_dates_top20": last_dates.most_common(20),
        "timings_seconds": {
            "discover": discovery_seconds,
            "signature": signature_seconds,
            "tail_scan": tail_scan_seconds,
        },
    }


def profile_csv_range(
    path: str | Path,
    *,
    start_date: Any,
    end_date: Any,
    max_files: int | None = None,
) -> dict[str, Any]:
    root = Path(path).resolve()
    paths = discover_stock_csvs(root)
    if max_files is not None:
        paths = paths[: max(0, int(max_files))]
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    started = time.perf_counter()
    rows = 0
    bytes_read = 0
    failures = []
    for csv_path in paths:
        bytes_read += csv_path.stat().st_size
        try:
            frame = pd.read_csv(
                csv_path,
                usecols=["trade_date", *RAW_FIELDS],
            )
            dates = pd.to_datetime(frame["trade_date"], format="mixed", errors="coerce")
            rows += int(((dates >= start) & (dates <= end)).sum())
        except Exception as exc:
            failures.append({"path": str(csv_path), "error": str(exc)})
    elapsed = time.perf_counter() - started
    return {
        "root": str(root),
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "files_scanned": len(paths),
        "source_bytes_scanned": int(bytes_read),
        "rows_in_range": int(rows),
        "failures": failures,
        "elapsed_seconds": elapsed,
        "throughput_mib_per_second": (
            bytes_read / (1024**2) / elapsed if elapsed > 0 else None
        ),
        "cache_state": "single_pass_os_cache_unspecified",
        "materialization": "streamed_count_only_no_global_matrix",
    }


def inspect_matrix_cache(data_root: str | Path, cache_dir: str | Path) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    cache_dir = Path(cache_dir).resolve()
    meta = load_ohlc_matrix_meta(cache_dir)
    started = time.perf_counter()
    current = matrix_cache_is_current(data_root, cache_dir)
    check_seconds = time.perf_counter() - started
    files = [item for item in cache_dir.glob("*") if item.is_file()]
    return {
        "root": str(cache_dir),
        "bytes": int(sum(item.stat().st_size for item in files)),
        "files": len(files),
        "matches_data_root": bool(current),
        "identity_check_seconds": check_seconds,
        "meta": {
            "version": meta.get("version") if meta else None,
            "data_dir": meta.get("data_dir") if meta else None,
            "source_count": meta.get("source_count") if meta else None,
            "source_hash": meta.get("source_hash") if meta else None,
            "shape": meta.get("shape") if meta else None,
            "date_start": (meta.get("dates") or [None])[0] if meta else None,
            "date_end": (meta.get("dates") or [None])[-1] if meta else None,
            "fields": meta.get("fields") if meta else None,
        },
    }


def file_record(path: str | Path) -> dict[str, Any]:
    from experiments.recording import sha256_file

    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_baseline_profile(
    *,
    project_root: str | Path,
    csv_roots: Iterable[str | Path],
    benchmark_root: str | Path,
    benchmark_start: Any,
    benchmark_end: Any,
    matrix_cache_dir: str | Path,
    baseline_contract: str | Path,
    artifact_inventory: str | Path,
    max_files: int | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    benchmark_root = Path(benchmark_root).resolve()
    return {
        "schema": "market_data_runtime_baseline_v1",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "memory": _memory_snapshot(),
        },
        "source": _git_state(root),
        "csv_roots": [inventory_csv_root(path) for path in csv_roots],
        "csv_range_benchmark": profile_csv_range(
            benchmark_root,
            start_date=benchmark_start,
            end_date=benchmark_end,
            max_files=max_files,
        ),
        "matrix_cache": inspect_matrix_cache(benchmark_root, matrix_cache_dir),
        "parity_oracles": {
            "baseline_contract": file_record(baseline_contract),
            "artifact_inventory": file_record(artifact_inventory),
            "rule": "existing canonical ledger artifacts remain the MD0 parity oracle",
        },
    }


def write_profile(output_dir: str | Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "market_data_runtime_baseline.json"
    temp = json_path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(json_path)

    benchmark = payload["csv_range_benchmark"]
    matrix = payload["matrix_cache"]
    rows = [
        "# MD0 行情与运行时基准",
        "",
        "该报告只读冻结 CSV、OHLC cache 和正式 ledger parity oracle；未重建缓存或运行回测。",
        "",
        "## 数据库存",
        "",
        "| root | 股票文件 | GiB | 尾部扫描秒 |",
        "|---|---:|---:|---:|",
    ]
    for item in payload["csv_roots"]:
        rows.append(
            f"| `{item['root']}` | {item['stock_files']} | "
            f"{item['bytes'] / 1024**3:.3f} | "
            f"{item['timings_seconds']['tail_scan']:.3f} |"
        )
    rows.extend(
        [
            "",
            "## CSV 基准",
            "",
            f"- 区间：{benchmark['start_date']} 至 {benchmark['end_date']}",
            f"- 文件：{benchmark['files_scanned']}",
            f"- 扫描源字节：{benchmark['source_bytes_scanned'] / 1024**3:.3f} GiB",
            f"- 区间行数：{benchmark['rows_in_range']}",
            f"- 耗时：{benchmark['elapsed_seconds']:.3f} 秒",
            f"- 吞吐：{benchmark['throughput_mib_per_second']:.3f} MiB/s",
            f"- 失败：{len(benchmark['failures'])}",
            "",
            "## OHLC Cache",
            "",
            f"- 路径：`{matrix['root']}`",
            f"- 大小：{matrix['bytes'] / 1024**3:.3f} GiB",
            f"- 日期：{matrix['meta']['date_start']} 至 {matrix['meta']['date_end']}",
            f"- 与当前数据根一致：{matrix['matches_data_root']}",
            f"- identity 检查：{matrix['identity_check_seconds']:.3f} 秒",
            "",
            "## Parity Oracle",
            "",
            f"- baseline contract：`{payload['parity_oracles']['baseline_contract']['path']}`",
            f"- artifact inventory：`{payload['parity_oracles']['artifact_inventory']['path']}`",
            "- MD1-MD6 不得改变现有正式 ledger 的订单、成交、阻塞、成本、持仓和净值。",
            "",
        ]
    )
    md_path = output_dir / "MD0_MARKET_DATA_RUNTIME_BASELINE.md"
    md_path.write_text("\n".join(rows), encoding="utf-8")
    return json_path, md_path
