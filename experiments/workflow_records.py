"""Build Qlib-aligned standard Records from one completed project Workflow."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from alpha.io import load_alpha_rows, resolve_row_scores
from experiments.recording import sha256_file
from experiments.record_templates import materialize_record_spec
from run.summarize_apm_attribution import load_industry_map, summarize_attribution


REQUIRED_LEDGER_ARTIFACTS = (
    "equity_curve",
    "positions",
    "orders",
    "rejections",
    "costs",
    "diagnostics",
)


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _main_candidate(config: Mapping[str, Any]) -> str:
    candidate = str(config.get("signal", {}).get("candidate_id", "")).strip()
    if candidate:
        return candidate
    candidates = config.get("model", {}).get("config", {}).get("candidate_ids", [])
    if len(candidates) != 1:
        raise ValueError("standard Records require one explicit main candidate")
    return str(candidates[0])


def resolve_prediction_paths(workflow_dir: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    rolling_manifest = workflow_dir / "model_signal" / "rolling_manifest.json"
    candidate = _main_candidate(config)
    dated_manifest = workflow_dir / "model_signal" / "dated_prediction_manifest.json"
    if rolling_manifest.is_file():
        payload = _load_json(rolling_manifest)
        sources = payload.get("split_alpha_paths", {})
    elif dated_manifest.is_file():
        payload = _load_json(dated_manifest)
        if payload.get("artifact_type") != "frozen_dated_predictions":
            raise ValueError("unsupported dated prediction manifest")
        try:
            sources = payload["candidates"][candidate]["split_alpha_paths"]
        except KeyError as exc:
            raise ValueError(
                f"dated prediction manifest has no candidate {candidate!r}"
            ) from exc
    else:
        raise FileNotFoundError(
            "standard Records require model_signal/rolling_manifest.json or "
            "model_signal/dated_prediction_manifest.json"
        )
    result = {}
    for split, item in sources.items():
        path = Path(str(item.get("path", ""))).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = str(item.get("sha256", "")).strip()
        if expected_hash and sha256_file(path) != expected_hash:
            raise ValueError(f"prediction artifact hash mismatch: {path}")
        result[str(split)] = path
    expected = list(config["evaluation"]["selection_splits"]) + list(
        config["evaluation"].get("observation_splits", [])
    )
    missing = set(expected) - set(result)
    if missing:
        raise ValueError(f"rolling prediction manifest is missing splits: {sorted(missing)}")
    return {split: result[split] for split in expected}


def resolve_cache_meta(workflow_dir: Path, config: Mapping[str, Any], project_root: Path) -> Path:
    configured = config.get("model", {}).get("config", {}).get("cache_meta")
    candidates = []
    if configured:
        candidates.append(_resolve_path(configured, project_root))
    model_manifest = workflow_dir / "model_signal" / "experiment_manifest.json"
    if model_manifest.is_file():
        metadata_path = _load_json(model_manifest).get("cache_contract", {}).get("metadata_path")
        if metadata_path:
            candidates.append(_resolve_path(metadata_path, project_root))
    for source in config.get("data", {}).get("sources", []):
        if source.get("provider") == "v14_memmap_v1":
            candidates.append(_resolve_path(source["root"], project_root))
    for path in candidates:
        if path.is_file() and path.suffix.lower() == ".pkl":
            return path
    raise FileNotFoundError("no concrete v14 cache metadata file is available for OOS labels")


def _resolve_label_raw_path(meta: Mapping[str, Any], meta_path: Path, family: str):
    families = meta.get("label_families", {})
    if family not in families:
        raise ValueError(f"cache does not contain label family {family!r}")
    view = families[family]
    physical = families[view.get("alias_of", family)]
    raw_path = Path(str(physical.get("raw_path", "")))
    if not raw_path.is_absolute():
        project_candidate = meta_path.parents[1] / raw_path if len(meta_path.parents) > 1 else raw_path
        sibling_candidate = meta_path.parent / raw_path.name
        raw_path = project_candidate if project_candidate.is_file() else sibling_candidate
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    return raw_path.resolve(), int(view.get("date_shift", 0))


def materialize_signal_inputs(
    prediction_paths: Mapping[str, Path],
    cache_meta: Path,
    label_family: str,
    horizon_index: int,
    candidate: str,
    output_dir: Path,
):
    with cache_meta.open("rb") as handle:
        meta = pickle.load(handle)
    raw_path, date_shift = _resolve_label_raw_path(meta, cache_meta, label_family)
    n_stocks, n_dates, max_horizon = len(meta["all_codes"]), len(meta["all_dates"]), int(meta["max_horizon"])
    if not 0 <= int(horizon_index) < max_horizon:
        raise ValueError("workflow label horizon is outside cache")
    labels = np.memmap(raw_path, dtype=np.float32, mode="r", shape=(n_stocks, n_dates, max_horizon))
    date_to_index = {pd.Timestamp(value).strftime("%Y-%m-%d"): i for i, value in enumerate(meta["all_dates"])}
    code_to_index = {str(code): i for i, code in enumerate(meta["all_codes"])}
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_out = output_dir / "prediction.jsonl"
    label_out = output_dir / "label.jsonl"
    metric_rows = []
    seen = set()
    row_count = 0
    with prediction_out.open("x", encoding="utf-8") as prediction_handle, label_out.open("x", encoding="utf-8") as label_handle:
        for split, path in prediction_paths.items():
            for row in load_alpha_rows(path):
                date = str(row["date"])
                if date in seen:
                    raise ValueError(f"duplicate prediction date across splits: {date}")
                seen.add(date)
                time_index = date_to_index.get(date)
                if time_index is None or time_index + date_shift >= n_dates:
                    raise ValueError(f"label date is unavailable for prediction date {date}")
                values = []
                for code in row["codes"]:
                    stock_index = code_to_index.get(str(code))
                    value = np.nan if stock_index is None else labels[stock_index, time_index + date_shift, horizon_index]
                    values.append(float(value) if np.isfinite(value) else None)
                prediction = dict(row)
                prediction.update({"split": split, "model_id": candidate, "asof_time": f"{date}T15:00:00"})
                label_row = {
                    "date": date,
                    "split": split,
                    "codes": list(row["codes"]),
                    "label": values,
                    "label_family": label_family,
                    "horizon_index": int(horizon_index),
                }
                prediction_handle.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                label_handle.write(json.dumps(label_row, ensure_ascii=False, allow_nan=False) + "\n")
                scores = np.asarray(resolve_row_scores(row), dtype=float)
                target = np.asarray([np.nan if value is None else value for value in values], dtype=float)
                valid = np.isfinite(scores) & np.isfinite(target)
                rank_ic = np.nan
                top_return = np.nan
                if int(valid.sum()) >= 3:
                    rank_ic = pd.Series(scores[valid]).rank().corr(pd.Series(target[valid]).rank())
                    top_n = max(1, int(np.ceil(valid.sum() * 0.006)))
                    top_return = float(target[valid][np.argsort(scores[valid])[-top_n:]].mean())
                metric_rows.append({
                    "date": date,
                    "split": split,
                    "stocks": int(len(scores)),
                    "valid_labels": int(valid.sum()),
                    "rank_ic": rank_ic,
                    "top0p6_return": top_return,
                })
                row_count += 1
    del labels
    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "signal_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    rank_ic = metrics["rank_ic"].dropna()
    summary = _json_safe({
        "rows": row_count,
        "rank_ic_mean": float(rank_ic.mean()) if len(rank_ic) else None,
        "rank_ic_ir": float(rank_ic.mean() / rank_ic.std(ddof=0)) if len(rank_ic) and rank_ic.std(ddof=0) > 0 else None,
        "top0p6_return_mean": float(metrics["top0p6_return"].mean()),
        "label_coverage": float(metrics["valid_labels"].sum() / max(metrics["stocks"].sum(), 1)),
    })
    return prediction_out, label_out, metrics_path, summary


def load_ledger_index(workflow_dir: Path, candidate: str) -> pd.DataFrame:
    frames = []
    for path in sorted((workflow_dir / "ledger").glob("**/path_artifact_index.csv")):
        frame = pd.read_csv(path)
        split = path.parent.name
        frame["split"] = split
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("Workflow ledger has no path_artifact_index.csv")
    result = pd.concat(frames, ignore_index=True)
    result = result.loc[result["alpha_name"].astype(str) == candidate].copy()
    if result.empty:
        raise ValueError(f"ledger path index has no rows for candidate {candidate!r}")
    missing = set(REQUIRED_LEDGER_ARTIFACTS) - set(result.columns)
    if missing:
        raise ValueError(f"ledger path index is missing artifacts: {sorted(missing)}")
    for column in REQUIRED_LEDGER_ARTIFACTS:
        for value in result[column]:
            if not Path(str(value)).is_file():
                raise FileNotFoundError(value)
    return result


def _json_safe_row(frame: pd.DataFrame, candidate: str, default_status: str):
    if frame.empty or "candidate" not in frame:
        return {"status": default_status}
    selected = frame.loc[frame["candidate"].astype(str) == candidate]
    if selected.empty:
        return {"status": default_status}
    return _json_safe({
        str(key): (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
        for key, value in selected.iloc[0].to_dict().items()
    })


def build_workflow_record_bundle(workflow_dir: str | Path) -> Path:
    workflow_dir = Path(workflow_dir).resolve()
    config = _load_json(workflow_dir / "workflow_config.json")
    compiled = _load_json(workflow_dir / "compiled_workflow.json")
    project_root = Path(compiled["project_root"]).resolve()
    candidate = _main_candidate(config)
    prediction_paths = resolve_prediction_paths(workflow_dir, config)
    cache_meta = resolve_cache_meta(workflow_dir, config, project_root)
    inputs = workflow_dir / "record_inputs"
    if inputs.exists():
        raise FileExistsError(f"immutable Record inputs already exist: {inputs}")
    label = config["dataset"]["label"]
    prediction, labels, signal_metrics, signal_summary = materialize_signal_inputs(
        prediction_paths,
        cache_meta,
        str(label["family"]),
        int(label["horizon_index"]),
        candidate,
        inputs / "signal",
    )
    ledger = load_ledger_index(workflow_dir, candidate)
    selection_splits = list(config["evaluation"]["selection_splits"])
    primary = ledger.loc[
        ledger["split"].eq(selection_splits[0])
        & ledger["stress"].eq("normal")
        & np.isclose(ledger["portfolio_value"].astype(float), 500_000)
    ]
    if len(primary) != 1:
        raise ValueError(f"expected one primary 50w normal ledger cell, found {len(primary)}")
    primary = primary.iloc[0]
    returns_df = pd.read_csv(primary["equity_curve"])
    diag_df = pd.read_csv(primary["diagnostics"])
    research_data = next(
        (_resolve_path(item["root"], project_root) for item in config["data"]["sources"] if item["role"] == "research_market"),
        project_root / "data" / "raw",
    )
    risk_summary, industry_df, style_df, slice_df = summarize_attribution(
        returns_df,
        diag_df,
        load_industry_map(project_root / "data" / "stock_industry.csv"),
        research_data,
    )
    risk_summary = _json_safe(risk_summary)
    risk_dir = inputs / "risk"
    risk_dir.mkdir(parents=True)
    risk_metrics = risk_dir / "risk_metrics.csv"
    pd.DataFrame([risk_summary]).to_csv(risk_metrics, index=False)
    industry_df.to_csv(risk_dir / "industry_exposure.csv", index=False)
    style_df.to_csv(risk_dir / "style_exposure.csv", index=False)
    slice_df.to_csv(risk_dir / "slice_metrics.csv", index=False)

    scorecard_dir = workflow_dir / "scorecard"
    scorecard_long = pd.read_csv(scorecard_dir / "registry_scorecard_long.csv")
    stress_frame = scorecard_long.loc[scorecard_long["candidate"].astype(str) == candidate].copy()
    stress_path = inputs / "stress_scorecard.csv"
    stress_frame.to_csv(stress_path, index=False)
    decisions = pd.read_csv(scorecard_dir / "registry_decisions.csv")
    forward_path = scorecard_dir / "registry_forward_summary.csv"
    forward = _read_optional_csv(forward_path)
    selection_result = _json_safe_row(decisions, candidate, "no_decision_row")
    forward_result = _json_safe_row(forward, candidate, "not_requested")
    decision_path = inputs / "decision_report.json"
    decision_path.write_text(
        json.dumps({"selection_result": selection_result, "forward_observation": forward_result}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    all_dates = [date for path in prediction_paths.values() for date in [row["date"] for row in load_alpha_rows(path)]]
    spec = {
        "schema_version": 1,
        "workflow": config,
        "records": {
            "signal": {
                "payload": {
                    "schema": "prediction_frame_v1",
                    "signal_start": min(all_dates), "signal_end": max(all_dates),
                    "rows": int(signal_summary["rows"]),
                    "asof_start": f"{min(all_dates)}T15:00:00", "asof_end": f"{max(all_dates)}T15:00:00",
                },
                "artifacts": {"prediction": str(prediction), "label": str(labels)},
            },
            "signal_analysis": {
                "payload": {
                    "selection_splits": selection_splits,
                    "observation_splits": list(config["evaluation"].get("observation_splits", [])),
                    "metrics": signal_summary,
                },
                "artifacts": {"signal_metrics": str(signal_metrics)},
            },
            "portfolio": {
                "payload": {
                    "execution_adapter": "official_open_ledger", "fill_price": "open",
                    "backtest_start": str(primary["backtest_start"]), "backtest_end": str(primary["backtest_end"]),
                    "metrics": {key: value for key, value in risk_summary.items() if isinstance(value, (int, float))},
                },
                "artifacts": {
                    name: str(primary[name])
                    for name in ("equity_curve", "positions", "orders", "rejections", "costs", "diagnostics")
                },
            },
            "risk_attribution": {
                "payload": {"metrics": risk_summary},
                "artifacts": {
                    "risk_metrics": str(risk_metrics),
                    "industry_exposure": str(risk_dir / "industry_exposure.csv"),
                    "style_exposure": str(risk_dir / "style_exposure.csv"),
                    "slice_metrics": str(risk_dir / "slice_metrics.csv"),
                },
            },
            "stress": {
                "payload": {
                    "capitals": sorted(stress_frame["capital"].dropna().astype(int).unique().tolist()),
                    "stresses": sorted(stress_frame["stress"].dropna().astype(str).unique().tolist()),
                    "cells": int(len(stress_frame)),
                },
                "artifacts": {"stress_scorecard": str(stress_path)},
            },
            "decision": {
                "payload": {
                    "selection_splits": selection_splits,
                    "observation_splits": list(config["evaluation"].get("observation_splits", [])),
                    "selection_result": selection_result,
                    "forward_observation": forward_result,
                    "forward_used_for_selection": False,
                },
                "artifacts": {"decision_report": str(decision_path)},
            },
        },
    }
    spec_path = inputs / "standard_record_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return materialize_record_spec(spec_path, workflow_dir)
