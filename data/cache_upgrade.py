"""Low-memory upgrade from a v13 feature cache to v14 multi-label cache."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from data.labels import PHYSICAL_LABEL_FAMILIES, build_forward_return_labels


SENTINEL = np.int16(-32768)
LABEL_SCALE = 1000


def resolve_cache_file(meta_path: Path, value: str) -> Path:
    """Resolve copied-cache paths by rebasing a missing path to the meta folder."""
    candidate = Path(value)
    if candidate.exists():
        return candidate
    rebased = meta_path.parent / candidate.name
    if rebased.exists():
        return rebased
    raise FileNotFoundError(f"cache file not found: {value} (also tried {rebased})")


def derive_v14_meta_path(source_meta_path: Path, last_date) -> Path:
    name = source_meta_path.name.replace("v13_config_key", "v14_multilabel_open")
    if not name.endswith("_meta.pkl"):
        raise ValueError(f"unexpected metadata filename: {source_meta_path.name}")
    stem = name[:-len("_meta.pkl")]
    end_tag = f"_end{pd.Timestamp(last_date).strftime('%Y%m%d')}"
    if end_tag not in stem:
        stem += end_tag
    return source_meta_path.with_name(stem + "_meta.pkl")


def _atomic_target(path: Path) -> Path:
    return path.with_name(path.name + ".building")


def _close_memmap(mm) -> None:
    mm.flush()
    mmap_obj = getattr(mm, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _replace_building(building: Path, final: Path, overwrite: bool) -> None:
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {final}")
        final.unlink()
    os.replace(building, final)


def _read_aligned_prices(csv_path: Path, all_dates: pd.DatetimeIndex):
    frame = pd.read_csv(csv_path)
    frame.columns = frame.columns.str.strip().str.lower()
    required = {"trade_date", "open", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.drop_duplicates("trade_date", keep="last").set_index("trade_date").sort_index()
    aligned = frame.reindex(all_dates)
    return aligned["open"].to_numpy(np.float32), aligned["close"].to_numpy(np.float32)


def _build_feature_validity(
    x_path: Path,
    shape,
    output_path: Path,
    stock_chunk: int,
):
    n_stocks, n_dates, x_dim = shape
    x_mm = np.memmap(x_path, dtype=np.int16, mode="r", shape=shape)
    valid_mm = np.memmap(output_path, dtype=np.uint8, mode="w+", shape=(n_dates, n_stocks))
    for start in range(0, n_stocks, stock_chunk):
        end = min(start + stock_chunk, n_stocks)
        valid_mm[:, start:end] = (x_mm[start:end, :, 0] != SENTINEL).T
        print(f"feature validity: {end}/{n_stocks}", flush=True)
    _close_memmap(valid_mm)
    del x_mm


def _write_raw_labels(
    all_codes: Iterable[str],
    all_dates: pd.DatetimeIndex,
    data_dir: Path,
    max_horizon: int,
    output_paths: Dict[str, Path],
):
    n_stocks = len(all_codes)
    n_dates = len(all_dates)
    raw_mmaps = {
        family: np.memmap(
            path, dtype=np.float32, mode="w+",
            shape=(n_stocks, n_dates, max_horizon),
        )
        for family, path in output_paths.items()
    }
    try:
        for stock_idx, code in enumerate(all_codes):
            csv_path = data_dir / f"{code}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"raw stock file not found: {csv_path}")
            opens, closes = _read_aligned_prices(csv_path, all_dates)
            labels = build_forward_return_labels(opens, closes, max_horizon)
            for family in PHYSICAL_LABEL_FAMILIES:
                raw_mmaps[family][stock_idx, :, :] = labels[family]
            if (stock_idx + 1) % 100 == 0 or stock_idx + 1 == n_stocks:
                print(f"raw labels: {stock_idx + 1}/{n_stocks}", flush=True)
    finally:
        for mm in raw_mmaps.values():
            _close_memmap(mm)


def _transpose_stock_to_date(source_path, temp_path, dtype, shape, stock_chunk):
    n_stocks, n_dates, max_horizon = shape
    source = np.memmap(source_path, dtype=dtype, mode="r", shape=shape)
    target = np.memmap(
        temp_path, dtype=dtype, mode="w+", shape=(n_dates, n_stocks, max_horizon)
    )
    for start in range(0, n_stocks, stock_chunk):
        end = min(start + stock_chunk, n_stocks)
        target[:, start:end, :] = source[start:end, :, :].transpose(1, 0, 2)
    _close_memmap(target)
    del source


def _transpose_date_to_stock(source_path, output_path, dtype, shape, stock_chunk):
    n_stocks, n_dates, max_horizon = shape
    source = np.memmap(
        source_path, dtype=dtype, mode="r", shape=(n_dates, n_stocks, max_horizon)
    )
    target = np.memmap(output_path, dtype=dtype, mode="w+", shape=shape)
    for start in range(0, n_stocks, stock_chunk):
        end = min(start + stock_chunk, n_stocks)
        target[start:end, :, :] = source[:, start:end, :].transpose(1, 0, 2)
    _close_memmap(target)
    del source


def _normalize_date_major(
    raw_date_path: Path,
    valid_path: Path,
    norm_date_path: Path,
    shape,
    min_stocks: int,
):
    n_stocks, n_dates, max_horizon = shape
    raw = np.memmap(
        raw_date_path, dtype=np.float32, mode="r", shape=(n_dates, n_stocks, max_horizon)
    )
    feature_valid = np.memmap(valid_path, dtype=np.uint8, mode="r", shape=(n_dates, n_stocks))
    norm = np.memmap(
        norm_date_path, dtype=np.int16, mode="w+", shape=(n_dates, n_stocks, max_horizon)
    )
    for date_idx in range(n_dates):
        values = raw[date_idx]
        output = np.full((n_stocks, max_horizon), SENTINEL, dtype=np.int16)
        base_valid = feature_valid[date_idx].astype(bool)
        for horizon in range(max_horizon):
            valid = base_valid & np.isfinite(values[:, horizon])
            if int(valid.sum()) < min_stocks:
                continue
            y = values[valid, horizon].astype(np.float32, copy=True)
            p_low, p_high = np.percentile(y, [1, 99])
            y = np.clip(y, p_low, p_high)
            y = (y - y.mean()) / (y.std() + 1e-8)
            output[valid, horizon] = (
                y * LABEL_SCALE
            ).clip(-32767, 32767).astype(np.int16)
        norm[date_idx] = output
        if (date_idx + 1) % 250 == 0 or date_idx + 1 == n_dates:
            print(f"normalize dates: {date_idx + 1}/{n_dates}", flush=True)
    _close_memmap(norm)
    del raw, feature_valid


def upgrade_v13_cache(
    source_meta_path,
    data_dir="data/raw",
    output_meta_path=None,
    overwrite=False,
    stock_chunk=32,
):
    """Upgrade labels while reusing all v13 feature/risk cache files."""
    source_meta_path = Path(source_meta_path)
    data_dir = Path(data_dir)
    with source_meta_path.open("rb") as handle:
        source = pickle.load(handle)

    if source.get("residualize", False):
        raise ValueError("residualized v13 caches require a dedicated residualization upgrade")
    all_codes = list(source["all_codes"])
    all_dates = pd.DatetimeIndex(source["all_dates"])
    n_stocks = len(all_codes)
    n_dates = len(all_dates)
    max_horizon = int(source["max_horizon"])
    shape = (n_stocks, n_dates, max_horizon)

    if output_meta_path is None:
        output_meta_path = derive_v14_meta_path(source_meta_path, all_dates[-1])
    output_meta_path = Path(output_meta_path)
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = output_meta_path.name[:-len("_meta.pkl")]

    reusable_keys = (
        "feat_path", "risk_path", "x_norm_path", "risk_full_path"
    )
    reusable = {
        key: resolve_cache_file(source_meta_path, source[key]) for key in reusable_keys
    }
    for key, path in reusable.items():
        print(f"reuse {key}: {path}")

    final_raw = {
        family: output_meta_path.parent / f"{prefix}_label_{family}_raw.dat"
        for family in PHYSICAL_LABEL_FAMILIES
    }
    final_norm = {
        family: output_meta_path.parent / f"{prefix}_label_{family}_norm.dat"
        for family in PHYSICAL_LABEL_FAMILIES
    }
    final_y = output_meta_path.parent / f"{prefix}_y_norm.dat"
    final_outputs = list(final_raw.values()) + list(final_norm.values()) + [final_y, output_meta_path]
    existing = [path for path in final_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"v14 output already exists: {existing[0]}")

    work_prefix = output_meta_path.parent / f".{prefix}_upgrade"
    valid_temp = Path(str(work_prefix) + "_feature_valid.tmp")
    raw_building = {family: _atomic_target(path) for family, path in final_raw.items()}
    norm_building = {family: _atomic_target(path) for family, path in final_norm.items()}
    y_building = _atomic_target(final_y)

    _build_feature_validity(
        reusable["x_norm_path"],
        (n_stocks, n_dates, int(source["x_dim"])),
        valid_temp,
        stock_chunk,
    )
    _write_raw_labels(
        all_codes, all_dates, data_dir, max_horizon, raw_building
    )

    for family in PHYSICAL_LABEL_FAMILIES:
        raw_date_temp = Path(str(work_prefix) + f"_{family}_raw_date.tmp")
        norm_date_temp = Path(str(work_prefix) + f"_{family}_norm_date.tmp")
        print(f"transpose/normalize family: {family}")
        _transpose_stock_to_date(
            raw_building[family], raw_date_temp, np.float32, shape, stock_chunk
        )
        _normalize_date_major(
            raw_date_temp, valid_temp, norm_date_temp, shape,
            int(source.get("min_stocks", 30)),
        )
        _transpose_date_to_stock(
            norm_date_temp, norm_building[family], np.int16, shape, stock_chunk
        )
        raw_date_temp.unlink()
        norm_date_temp.unlink()

    cc_norm = np.memmap(norm_building["cc"], dtype=np.int16, mode="r", shape=shape)
    y_mm = np.memmap(y_building, dtype=np.int16, mode="w+", shape=(n_stocks, n_dates))
    target_idx = min(int(source.get("target_horizon", 5)) - 1, max_horizon - 1)
    for start in range(0, n_stocks, stock_chunk):
        end = min(start + stock_chunk, n_stocks)
        y_mm[start:end, :] = cc_norm[start:end, :, target_idx]
    _close_memmap(y_mm)
    del cc_norm

    for family in PHYSICAL_LABEL_FAMILIES:
        _replace_building(raw_building[family], final_raw[family], overwrite)
        _replace_building(norm_building[family], final_norm[family], overwrite)
    _replace_building(y_building, final_y, overwrite)
    valid_temp.unlink()

    def portable(path: Path) -> str:
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path)

    upgraded = dict(source)
    upgraded.update({key: portable(path) for key, path in reusable.items()})
    upgraded["ret_path"] = portable(final_raw["cc"])
    upgraded["y_norm_path"] = portable(final_y)
    upgraded["y_seq_norm_path"] = portable(final_norm["cc"])
    upgraded["label_schema_version"] = 1
    upgraded["label_families"] = {
        family: {
            "raw_path": portable(final_raw[family]),
            "norm_path": portable(final_norm[family]),
            "date_shift": 0,
        }
        for family in PHYSICAL_LABEL_FAMILIES
    }
    upgraded["label_families"]["oo_lag1"] = {"alias_of": "oo", "date_shift": 1}
    upgraded["upgraded_from"] = portable(source_meta_path)

    meta_building = _atomic_target(output_meta_path)
    with meta_building.open("wb") as handle:
        pickle.dump(upgraded, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _replace_building(meta_building, output_meta_path, overwrite)
    print(f"v14 metadata: {output_meta_path}")
    return upgraded, output_meta_path
