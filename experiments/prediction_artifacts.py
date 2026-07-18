"""Hash-checked dated prediction artifacts for frozen and legacy alpha."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from alpha.io import iter_alpha_rows, resolve_row_scores
from core.research_protocol import get_split_spec
from experiments.model_adapters import PredictionFrame
from experiments.recording import sha256_file


DATED_PREDICTION_MANIFEST = "dated_prediction_manifest.json"


def resolve_split_alpha_path(
    project_root: str | Path,
    signal_path: str | Path,
    split: str,
) -> Path:
    """Resolve one registered signal using the same shapes accepted by the ledger."""

    root = Path(project_root).resolve()
    base = Path(signal_path)
    if not base.is_absolute():
        base = root / base
    direct = base / split / "alpha_policy.jsonl"
    if direct.is_file():
        return direct.resolve()
    manifest_path = (
        base
        if base.is_file() and base.name == "rolling_manifest.json"
        else base / "rolling_manifest.json"
    )
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        stitched = payload.get("split_alpha_paths", {}).get(split, {})
        stitched_path = Path(str(stitched.get("path", "")))
        if not stitched_path.is_absolute():
            stitched_path = manifest_path.parent / stitched_path
        if stitched_path.is_file():
            return stitched_path.resolve()
        spec = get_split_spec(split)
        declared_windows = {
            str(item.get("name")): item
            for item in payload.get("config", {}).get("windows", [])
        }
        matches = []
        for window in payload.get("windows", []):
            declared = declared_windows.get(str(window.get("name")), {})
            start = pd.Timestamp(declared.get("predict_start")) if declared.get("predict_start") else None
            end = pd.Timestamp(declared.get("predict_end")) if declared.get("predict_end") else None
            alpha_path = Path(str(window.get("alpha_path", "")))
            if not alpha_path.is_absolute():
                alpha_path = manifest_path.parent / alpha_path
            if (
                start is not None
                and end is not None
                and start <= spec.start
                and end >= spec.end
                and alpha_path.is_file()
            ):
                matches.append(alpha_path.resolve())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"multiple rolling alpha windows match split={split}: {matches}")
    if base.is_file() and base.suffix.lower() == ".jsonl":
        return base.resolve()
    raise FileNotFoundError(f"no alpha_policy.jsonl for split={split} under {base}")


def inspect_dated_alpha(
    path: str | Path,
    *,
    candidate_id: str,
    split: str,
) -> dict[str, Any]:
    """Validate one alpha file one date at a time without materializing it."""

    source = Path(path).resolve()
    spec = get_split_spec(split)
    first_date = None
    last_date = None
    dates = 0
    stock_rows = 0
    for row in iter_alpha_rows(source):
        date = pd.Timestamp(row["date"]).normalize()
        if date < spec.start or date > spec.end:
            raise ValueError(
                f"frozen prediction date {date.date()} is outside {split} "
                f"[{spec.start.date()}, {spec.end.date()}]"
            )
        values = np.asarray(resolve_row_scores(row), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"frozen prediction row {date.date()} has non-finite scores")
        PredictionFrame.from_daily_scores(
            [row],
            model_id=candidate_id,
            asof_time="close",
        )
        first_date = date if first_date is None else min(first_date, date)
        last_date = date if last_date is None else max(last_date, date)
        dates += 1
        stock_rows += len(row.get("codes", []))
    if dates == 0:
        raise ValueError(f"frozen prediction artifact is empty: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "schema": "prediction_frame_v1",
        "asof_policy": "signal_date_close_15_00",
        "dates": dates,
        "rows": stock_rows,
        "signal_start": str(first_date.date()),
        "signal_end": str(last_date.date()),
    }


def materialize_frozen_prediction_manifest(
    *,
    project_root: str | Path,
    candidates_csv: str | Path,
    candidate_ids: Iterable[str],
    splits: Iterable[str],
    output_dir: str | Path,
) -> Path:
    """Freeze registry lineage and dated alpha hashes without copying alpha."""

    root = Path(project_root).resolve()
    registry_path = Path(candidates_csv)
    if not registry_path.is_absolute():
        registry_path = root / registry_path
    registry_path = registry_path.resolve()
    frame = pd.read_csv(registry_path, dtype=str).fillna("")
    if "candidate_id" not in frame:
        raise ValueError("candidate registry is missing candidate_id")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate registry contains duplicate candidate_id")
    indexed = frame.set_index("candidate_id", drop=False)
    split_names = [str(value) for value in splits]
    if not split_names:
        raise ValueError("at least one split is required")
    candidates: dict[str, Mapping[str, Any]] = {}
    for raw_candidate in candidate_ids:
        candidate = str(raw_candidate).strip()
        if not candidate:
            raise ValueError("candidate_id cannot be empty")
        if candidate not in indexed.index:
            raise KeyError(f"unknown candidate_id: {candidate}")
        row = indexed.loc[candidate]
        signal_path = str(row.get("signal_path", "")).strip()
        if not signal_path:
            raise ValueError(f"candidate {candidate!r} has no signal_path")
        split_paths = {}
        for split in split_names:
            source = resolve_split_alpha_path(root, signal_path, split)
            split_paths[split] = inspect_dated_alpha(
                source,
                candidate_id=candidate,
                split=split,
            )
        candidates[candidate] = {
            "registry_status": str(row.get("status", "")),
            "selection_eligible": str(row.get("selection_eligible", "")).lower() == "true",
            "forward_observation_only": str(row.get("forward_observation_only", "")).lower() == "true",
            "signal_path": signal_path,
            "split_alpha_paths": split_paths,
        }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / DATED_PREDICTION_MANIFEST
    payload = {
        "schema_version": 1,
        "artifact_type": "frozen_dated_predictions",
        "source_mode": "registry_read_only",
        "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path)},
        "candidates": candidates,
        "training_executed": False,
        "promotion_performed": False,
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path
