"""Validated explicit access to an existing cross-section memmap bundle."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


REQUIRED_META_KEYS = {
    "all_codes",
    "all_dates",
    "industry_array",
    "x_dim",
    "risk_full_dim",
    "max_horizon",
    "x_norm_path",
    "risk_full_path",
    "y_norm_path",
    "y_seq_norm_path",
}


def _resolve_data_path(value: str | Path, *, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_explicit_cross_section_meta(
    meta_path: str | Path,
    *,
    project_root: str | Path,
    expected_input_dim: int | None = None,
    required_label_families: Sequence[str] = (),
    logical_end: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Load one declared cache without falling back to a raw-data rebuild.

    The pickle is a trusted local project artifact. Every referenced memmap is
    resolved and checked before the caller starts training or inference.
    """
    root = Path(project_root).resolve()
    path = _resolve_data_path(meta_path, project_root=root)
    if not path.is_file():
        raise FileNotFoundError(f"explicit cache metadata is missing: {path}")
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"cache metadata must be a mapping: {path}")
    meta = dict(loaded)
    missing_keys = sorted(REQUIRED_META_KEYS - set(meta))
    if missing_keys:
        raise ValueError(f"cache metadata is missing keys {missing_keys}: {path}")

    dates = pd.DatetimeIndex(pd.to_datetime(meta["all_dates"])).normalize()
    if dates.empty or not dates.is_monotonic_increasing:
        raise ValueError(f"cache dates must be non-empty and ordered: {path}")
    if expected_input_dim is not None and int(meta["x_dim"]) != int(expected_input_dim):
        raise ValueError(
            f"cache input dimension mismatch: expected={expected_input_dim} actual={meta['x_dim']}"
        )
    families = meta.get("label_families", {})
    missing_families = sorted(set(required_label_families) - set(families))
    if missing_families:
        raise ValueError(f"cache does not contain label families: {missing_families}")

    data_keys = ("x_norm_path", "risk_full_path", "y_norm_path", "y_seq_norm_path", "ret_path")
    referenced: list[Path] = []
    for key in data_keys:
        if meta.get(key):
            resolved = _resolve_data_path(meta[key], project_root=root)
            meta[key] = str(resolved)
            referenced.append(resolved)
    normalized_families = {}
    for name, raw_family in families.items():
        family = dict(raw_family)
        for key in ("raw_path", "norm_path"):
            if family.get(key):
                resolved = _resolve_data_path(family[key], project_root=root)
                family[key] = str(resolved)
                referenced.append(resolved)
        normalized_families[name] = family
    meta["label_families"] = normalized_families
    missing_files = sorted(str(item) for item in set(referenced) if not item.is_file())
    if missing_files:
        raise FileNotFoundError(f"explicit cache has missing data files: {missing_files[:3]}")

    effective_end = dates.max()
    if logical_end is not None:
        requested_end = pd.Timestamp(logical_end).normalize()
        if requested_end < dates.min() or requested_end > dates.max():
            raise ValueError(
                f"logical cache end {requested_end.date()} is outside physical coverage "
                f"{dates.min().date()}..{dates.max().date()}"
            )
        effective_end = requested_end
    meta["meta_path"] = str(path)
    meta["physical_data_start"] = str(dates.min().date())
    meta["physical_data_end"] = str(dates.max().date())
    meta["effective_data_end"] = str(effective_end.date())
    meta["cache_view_kind"] = (
        "physical_superset_logical_cutoff" if effective_end < dates.max() else "physical_full_view"
    )
    return meta
