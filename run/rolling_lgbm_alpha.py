"""Run a Qlib-style rolling LightGBM baseline on frozen v14 PIT samples."""

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import load_alpha_rows, write_alpha_rows
from core.config import DataConfig
from core.research_protocol import RESEARCH_END_DATE, SPLIT_SPECS
from data.labels import label_end_offset
from data.dataset_runtime import ProjectDataset, build_v14_rolling_dataset
from data.pipeline import AGG_NAMES, INDUSTRY_REL_FEATURES
from data.pipeline import build_cross_section_dataset
from data.providers import DataView, V14MemmapProvider
from data.rolling_samples import iter_rolling_samples
from experiments.factor_baselines import get_factor_baseline
from experiments.model_adapters import LightGBMModelAdapter
from experiments.recording import (
    MANIFEST_NAME,
    append_event,
    canonical_json_hash,
    create_experiment,
    declared_range,
    finalize_artifact_index,
    not_applicable_range,
    record_artifact,
    sha256_file,
    validate_manifest_for_formal_use,
)
from experiments.rolling import RollingWindow, build_split_alpha_files, resolve_window_indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qlib_style_rolling_lgbm_oo_lag1.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", action="append", default=None)
    parser.add_argument("--experiment-id", default=None, help="Required for a non-dry-run immutable experiment.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--replace-dry-run",
        action="store_true",
        help="Allow a formal run to remove a prior dry-run rolling manifest in the same output directory.",
    )
    return parser.parse_args(argv)


def load_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"data", "model", "windows"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"rolling config missing: {sorted(missing)}")
    return value


def build_data_config(spec):
    data = spec["data"]
    flags = data.get("feature_flags", {})
    cfg = DataConfig(data_dir=data.get("data_dir", "data/raw"), research_end_date=data["research_end"])
    cfg.allow_cache_superset = True
    cfg.label_family = data["label_family"]
    cfg.use_technical_features = bool(flags.get("technical", False))
    cfg.use_market_features = bool(flags.get("market", True))
    cfg.use_macro_features = bool(flags.get("macro", False))
    cfg.use_fundamental_features = bool(flags.get("fundamental", False))
    cfg.use_fundamental_quality_features = bool(flags.get("fundamental_quality", False))
    cfg.use_shareholder_features = bool(flags.get("shareholder", False))
    cfg.use_restricted_features = bool(flags.get("restricted", False))
    return cfg


def resolve_dataset_runtime(spec):
    value = str(spec.get("data", {}).get("dataset_runtime", "legacy_iter"))
    if value not in {"legacy_iter", "project_dataset"}:
        raise ValueError(f"unsupported LightGBM dataset_runtime: {value}")
    return value


def resolve_model_runtime(spec):
    value = str(spec.get("model", {}).get("runtime", "legacy_trainer"))
    if value not in {"legacy_trainer", "project_adapter"}:
        raise ValueError(f"unsupported LightGBM model runtime: {value}")
    return value


def _segment_samples(
    meta,
    indices,
    label_family,
    horizon_index,
    feature_indices=None,
    *,
    dataset: ProjectDataset | None = None,
    segment: str | None = None,
    data_key: str = "raw",
):
    if dataset is None:
        return iter_rolling_samples(
            meta,
            indices,
            label_family,
            horizon_index,
            feature_indices,
            include_risk=False,
            include_industry=False,
        )
    if not segment:
        raise ValueError("Dataset sample source requires a named segment")
    return dataset.prepare(segment, data_key=data_key)


def collect_rows(
    meta,
    indices,
    label_family,
    horizon_index,
    max_rows,
    seed,
    feature_indices=None,
    *,
    dataset: ProjectDataset | None = None,
    segment: str | None = None,
):
    quota = max(1, int(np.ceil(max_rows / max(len(indices), 1))))
    xs, ys = [], []
    for sample in _segment_samples(
        meta,
        indices,
        label_family,
        horizon_index,
        feature_indices,
        dataset=dataset,
        segment=segment,
        data_key="learn" if dataset is not None else "raw",
    ):
        take = min(quota, len(sample["y"]))
        rng = np.random.default_rng(int(seed) + sample["time_index"])
        selected = rng.choice(len(sample["y"]), size=take, replace=False)
        xs.append(sample["X"][selected])
        ys.append(sample["y"][selected])
    if not xs:
        raise ValueError("window has no label-safe samples")
    return np.vstack(xs)[:max_rows], np.concatenate(ys)[:max_rows]


def train_window(
    meta,
    indices,
    spec,
    label_family,
    horizon_index,
    feature_indices=None,
    *,
    dataset: ProjectDataset | None = None,
):
    model_spec = spec["model"]
    train_x, train_y = collect_rows(meta, indices["train"], label_family, horizon_index, model_spec["max_train_rows"], model_spec["seed"], feature_indices, dataset=dataset, segment="train")
    valid_x, valid_y = collect_rows(meta, indices["valid"], label_family, horizon_index, model_spec["max_valid_rows"], model_spec["seed"] + 1, feature_indices, dataset=dataset, segment="valid")
    train_rows, valid_rows = len(train_y), len(valid_y)
    params = {key: model_spec[key] for key in ("objective", "learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l2", "seed")}
    params.update({
        "metric": "l2",
        "verbosity": -1,
        "force_col_wise": True,
        "num_threads": int(model_spec.get("num_threads", 4)),
    })
    train = lgb.Dataset(train_x, label=train_y, free_raw_data=True)
    valid = lgb.Dataset(valid_x, label=valid_y, reference=train, free_raw_data=True)
    train.construct()
    valid.construct()
    del train_x, train_y, valid_x, valid_y
    gc.collect()
    model = lgb.train(params, train, num_boost_round=model_spec["num_boost_round"], valid_sets=[valid], callbacks=[lgb.early_stopping(model_spec["early_stopping_rounds"], verbose=False)])
    return model, {"train_rows": train_rows, "valid_rows": valid_rows, "best_iteration": int(model.best_iteration)}


def predict_rows(
    meta,
    indices,
    label_family,
    horizon_index,
    model,
    feature_indices=None,
    *,
    dataset: ProjectDataset | None = None,
    segment: str = "predict",
):
    rows = []
    for sample in _segment_samples(
        meta,
        indices,
        label_family,
        horizon_index,
        feature_indices,
        dataset=dataset,
        segment=segment,
        data_key="infer" if dataset is not None else "raw",
    ):
        score = model.predict(sample["X"], num_iteration=model.best_iteration)
        order = np.argsort(score)[::-1]
        rows.append({"date": sample["date"], "codes": np.asarray(sample["codes"], dtype=object)[order].tolist(), "alpha": score[order].astype(float).tolist(), "n_stocks": len(score)})
    return rows


def build_v14_feature_layout(meta):
    """Return names for the expanded v14 matrix in cache order."""
    columns = list(meta["feature_cols"])
    high_count = int(meta.get("high_feat_dim", len(columns)))
    high_columns = columns[:high_count]
    extra_columns = columns[high_count:]
    layout = [f"agg:{agg}:{column}" for agg in AGG_NAMES for column in high_columns]
    layout.extend(f"extra:last:{column}" for column in extra_columns)
    layout.extend(f"extra:qoq:{column}" for column in extra_columns)
    high_layout = [f"agg:{agg}:{column}" for agg in AGG_NAMES for column in high_columns]
    layout.extend(f"rank:{name}" for name in high_layout)
    layout.extend(
        f"industry_rel:agg:last:{column}"
        for column in INDUSTRY_REL_FEATURES
        if column in high_columns
    )
    if len(layout) != int(meta["x_dim"]):
        raise ValueError(f"v14 feature layout mismatch: names={len(layout)} x_dim={meta['x_dim']}")
    return layout


def resolve_feature_indices(meta, spec):
    name = spec.get("data", {}).get("factor_baseline")
    columns = list(meta["feature_cols"])
    if "x_dim" not in meta:
        # Keep the lightweight contract used by unit tests and legacy callers.
        if not name:
            return None, columns, "v14_full"
        baseline = get_factor_baseline(name)
        missing = [feature for feature in baseline["features"] if feature not in columns]
        if missing:
            raise ValueError(f"factor baseline features missing from cache: {missing}")
        return [columns.index(feature) for feature in baseline["features"]], list(baseline["features"]), name
    layout = build_v14_feature_layout(meta)
    if not name:
        return None, layout, "v14_full"
    baseline = get_factor_baseline(name)
    missing = [feature for feature in baseline["features"] if feature not in columns]
    if missing:
        raise ValueError(f"factor baseline features missing from cache: {missing}")
    selected_names = [f"agg:last:{feature}" for feature in baseline["features"]]
    return [layout.index(feature) for feature in selected_names], selected_names, name


def _window_scope(windows, cache_contract):
    train_start = max(
        min(window.train_start for window in windows),
        str(cache_contract["data_start"]),
    )
    train_end = max(window.train_end for window in windows)
    valid_start = min(window.valid_start for window in windows)
    valid_end = max(window.valid_end for window in windows)
    signal_start = min(window.predict_start for window in windows)
    signal_end = max(window.predict_end for window in windows)
    roles = []
    for split, split_spec in SPLIT_SPECS.items():
        if any(
            str(split_spec.start.date()) <= window.predict_start <= str(split_spec.end.date())
            for window in windows
        ):
            roles.append(
                {
                    "split": split,
                    "selection_eligible": split_spec.selection_eligible,
                    "forward_used": split_spec.is_forward,
                }
            )
    if not roles:
        roles.append(
            {
                "split": "historical_oos",
                "role": "research_oos_preselection",
                "selection_eligible": False,
                "forward_used": False,
            }
        )
    contract_hash = canonical_json_hash(cache_contract)
    return {
        "stage": "model_signal",
        "data_sources": [
            {
                "role": "feature_cache",
                "root": cache_contract.get("metadata_path") or "v14_cache_metadata",
                "fingerprint": {"kind": "cache_contract_sha256", "value": contract_hash},
            }
        ],
        "ranges": {
            "feature_warmup": declared_range(cache_contract["data_start"], train_start),
            "train": declared_range(train_start, train_end),
            "valid": declared_range(valid_start, valid_end),
            "signal": declared_range(signal_start, signal_end),
            "backtest": not_applicable_range("ledger evaluation is a separate experiment stage"),
        },
        "max_data_date": signal_end,
        "split_roles": roles,
        "transform": {
            "state_sha256": contract_hash,
            "fit_range": not_applicable_range("v14 uses dated cross-sectional transforms"),
        },
        "lineage": {},
    }


def build_experiment_contract(
    spec,
    meta,
    feature_set,
    feature_names,
    label_end_offset,
    windows=None,
    provider_manifest=None,
):
    contract = {
        "config": spec,
        "protocol": {
            "research_end": str(RESEARCH_END_DATE.date()),
            "cache_read_ceiling": spec["data"]["research_end"],
            "label_family": spec["data"]["label_family"],
            "horizon_index": int(spec["data"]["horizon_index"]),
            "label_end_offset": int(label_end_offset),
            "selection_splits": ["val_2024", "test_2025"],
            "forward_is_observation_only": True,
        },
        "cache_contract": {
            "metadata_path": str(meta.get("meta_path", "")),
            "data_start": str(meta["all_dates"][0]),
            "data_end": str(meta["all_dates"][-1]),
            "data_scope": {
                "role": "research",
                "effective_start": str(meta["all_dates"][0]),
                "effective_end": str(meta["all_dates"][-1]),
                "physical_coverage": "see_data_boundary_audit",
                "complete": True,
            },
            "x_dim": int(meta["x_dim"]),
            "feature_set": feature_set,
            "feature_names": feature_names,
            "provider_manifest": dict(provider_manifest or {}),
        },
    }
    if windows:
        contract["experiment_scope"] = _window_scope(windows, contract["cache_contract"])
    return contract


def build_v14_provider(spec, meta, windows):
    dates = pd.DatetimeIndex(pd.to_datetime(meta["all_dates"])).normalize()
    data_root = Path(spec["data"].get("data_dir", "data/raw"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    task_start = max(dates.min(), min(pd.Timestamp(window.train_start) for window in windows))
    task_end = max(pd.Timestamp(window.predict_end) for window in windows)
    evaluation_start = min(pd.Timestamp(window.predict_start) for window in windows)
    view = DataView.create(
        name=f"{spec.get('name', 'rolling')}:selection",
        physical_root=data_root,
        feature_warmup_start=dates.min(),
        feature_warmup_end=task_start,
        task_start=task_start,
        task_end=task_end,
        evaluation_start=evaluation_start,
        evaluation_end=task_end,
        max_data_date=spec["data"]["research_end"],
    )
    provider = V14MemmapProvider(
        meta=meta,
        meta_path=meta.get("meta_path", ""),
        data_view=view,
    )
    for window in windows:
        for start, end in (
            (window.train_start, window.train_end),
            (window.valid_start, window.valid_end),
            (window.predict_start, window.predict_end),
        ):
            provider.date_indices(start, end)
    return provider


def write_rolling_progress(path, *, config_sha256, window_entries):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "windows": list(window_entries),
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def load_rolling_progress(path, *, config_sha256):
    target = Path(path)
    if not target.is_file():
        return []
    payload = json.loads(target.read_text(encoding="utf-8-sig"))
    if payload.get("config_sha256") != config_sha256:
        raise ValueError("rolling progress config hash mismatch")
    entries = payload.get("windows")
    if not isinstance(entries, list):
        raise ValueError("rolling progress windows must be a list")
    for entry in entries:
        for path_field, hash_field in (
            ("alpha_path", "alpha_sha256"),
            ("model_path", "model_sha256"),
        ):
            artifact = Path(str(entry.get(path_field, "")))
            if not artifact.is_file() or sha256_file(artifact) != entry.get(hash_field):
                raise ValueError(f"rolling progress artifact mismatch: {artifact}")
    return entries


def main(argv=None):
    args, spec = parse_args(argv), None
    spec = load_config(args.config)
    windows = [RollingWindow.from_mapping(value) for value in spec["windows"]]
    if args.window:
        windows = [window for window in windows if window.name in set(args.window)]
    if not windows:
        raise ValueError("no rolling windows selected")
    meta = build_cross_section_dataset(build_data_config(spec), use_cache=True)
    if not isinstance(meta, dict):
        raise RuntimeError("rolling runner requires a v14 precomputed cache")
    provider = build_v14_provider(spec, meta, windows)
    provider_manifest = provider.manifest()
    family, horizon = spec["data"]["label_family"], int(spec["data"]["horizon_index"])
    feature_indices, feature_names, feature_set = resolve_feature_indices(meta, spec)
    dataset_runtime = resolve_dataset_runtime(spec)
    model_runtime = resolve_model_runtime(spec)
    if model_runtime == "project_adapter" and dataset_runtime != "project_dataset":
        raise ValueError("project_adapter requires data.dataset_runtime=project_dataset")
    offset = label_end_offset(family, horizon)
    output, manifest = Path(args.output_dir), {
        "run_mode": "dry_run" if args.dry_run else "formal",
        "config": spec,
        "label_end_offset": offset,
        "feature_set": feature_set,
        "feature_names": feature_names,
        "dataset_runtime": dataset_runtime,
        "model_runtime": model_runtime,
        "windows": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and not args.experiment_id:
        raise ValueError("--experiment-id is required for a non-dry-run rolling experiment")
    manifest_path = output / "rolling_manifest.json"
    progress_path = output / "rolling_progress.json"
    if not args.dry_run and manifest_path.exists() and args.resume:
        validate_manifest_for_formal_use(output / MANIFEST_NAME)
        print(manifest_path.read_text(encoding="utf-8"))
        return
    if not args.dry_run and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        is_formal = existing.get("run_mode") == "formal" or any(
            entry.get("model_path") or entry.get("alpha_path")
            for entry in existing.get("windows", [])
        )
        if is_formal:
            raise FileExistsError(f"formal rolling manifest already exists: {manifest_path}")
        if not args.replace_dry_run:
            raise FileExistsError(
                f"dry-run rolling manifest exists; use a clean output directory or --replace-dry-run: {manifest_path}"
            )
        manifest_path.unlink()
    contract = build_experiment_contract(
        spec,
        meta,
        feature_set,
        feature_names,
        offset,
        windows,
        provider_manifest=provider_manifest,
    )
    config_hash = canonical_json_hash(contract["config"])
    if not args.dry_run:
        experiment_manifest_path = output / MANIFEST_NAME
        resume_existing = args.resume and experiment_manifest_path.is_file()
        if args.resume and not resume_existing and progress_path.exists():
            raise FileNotFoundError(
                f"rolling progress exists without its frozen experiment manifest: {experiment_manifest_path}"
            )
        if resume_existing:
            experiment_manifest = json.loads(experiment_manifest_path.read_text(encoding="utf-8-sig"))
            if experiment_manifest.get("config_sha256") != config_hash:
                raise ValueError("resume config does not match the frozen experiment manifest")
            manifest["windows"] = load_rolling_progress(
                progress_path,
                config_sha256=config_hash,
            )
            append_event(
                output,
                status="running",
                event_type="rolling_training_resumed",
                details={"completed_windows": len(manifest["windows"])},
            )
        else:
            create_experiment(
                output,
                experiment_id=args.experiment_id,
                config=contract["config"],
                protocol=contract["protocol"],
                cache_contract=contract["cache_contract"],
                project_root=ROOT,
                formal=True,
                experiment_scope=contract["experiment_scope"],
            )
            record_artifact(output, name="config", path=ROOT / args.config, kind="training_config")
            append_event(output, status="running", event_type="rolling_training_started")
    try:
        completed_names = {entry["name"] for entry in manifest["windows"]}
        for window in windows:
            if window.name in completed_names:
                continue
            indices = resolve_window_indices(meta["all_dates"], window, offset)
            entry = {"name": window.name, "counts": {key: len(value) for key, value in indices.items()}}
            if not args.dry_run:
                dataset = None
                if dataset_runtime == "project_dataset":
                    dataset = build_v14_rolling_dataset(
                        provider=provider,
                        segment_indices=indices,
                        label_family=family,
                        horizon_index=horizon,
                        feature_indices=feature_indices,
                    )
                    dataset.fit()
                adapter = None
                if model_runtime == "project_adapter":
                    adapter_config = dict(spec["model"])
                    adapter_config.update(
                        {
                            "sampling_mode": "daily_seeded_quota_v1",
                            "segment_date_counts": {
                                "train": len(indices["train"]),
                                "valid": len(indices["valid"]),
                            },
                        }
                    )
                    adapter = LightGBMModelAdapter(
                        model_id=f"{spec.get('name', 'rolling_lgbm')}:{window.name}",
                        config=adapter_config,
                    )
                    fit_result = adapter.fit(dataset)
                    model = adapter.model
                    metrics = {
                        key: fit_result.metrics[key]
                        for key in ("train_rows", "valid_rows", "best_iteration", "sampling_mode")
                    }
                    prediction = adapter.predict(dataset, "predict")
                    rows = [
                        {
                            "date": row["date"],
                            "codes": row["codes"],
                            "alpha": row["alpha"],
                            "n_stocks": row["n_stocks"],
                        }
                        for row in prediction.to_alpha_rows(tie_breaker="legacy_numpy")
                    ]
                    for row in rows:
                        row["date"] = str(pd.Timestamp(row["date"]))
                else:
                    model, metrics = train_window(
                        meta,
                        indices,
                        spec,
                        family,
                        horizon,
                        feature_indices,
                        dataset=dataset,
                    )
                    rows = predict_rows(
                        meta,
                        indices["predict"],
                        family,
                        horizon,
                        model,
                        feature_indices,
                        dataset=dataset,
                    )
                window_dir = output / "windows" / window.name
                window_dir.mkdir(parents=True, exist_ok=True)
                path = window_dir / "alpha_raw.jsonl"
                model_path = window_dir / "model.txt"
                write_alpha_rows(path, [{**row, "date": str(row["date"])} for row in rows])
                if adapter is not None:
                    adapter.save_state(model_path)
                else:
                    model.save_model(str(model_path), num_iteration=model.best_iteration)
                record_artifact(output, name=f"alpha:{window.name}", path=path, kind="dated_alpha")
                record_artifact(output, name=f"model:{window.name}", path=model_path, kind="lightgbm_model")
                entry.update(
                    metrics
                    | {
                        "alpha_path": str(path.resolve()),
                        "model_path": str(model_path.resolve()),
                        "alpha_sha256": sha256_file(path),
                        "model_sha256": sha256_file(model_path),
                    }
                )
            manifest["windows"].append(entry)
            if not args.dry_run:
                write_rolling_progress(
                    progress_path,
                    config_sha256=config_hash,
                    window_entries=manifest["windows"],
                )
        if args.dry_run:
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        else:
            split_alpha = build_split_alpha_files(manifest["windows"], output)
            manifest["split_alpha_paths"] = split_alpha
            with manifest_path.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
            for split, item in split_alpha.items():
                record_artifact(
                    output,
                    name=f"stitched_alpha:{split}",
                    path=item["path"],
                    kind="stitched_dated_alpha",
                )
            record_artifact(output, name="rolling_manifest", path=manifest_path, kind="rolling_manifest")
            record_artifact(output, name="rolling_progress", path=progress_path, kind="rolling_progress")
            append_event(output, status="completed", event_type="rolling_training_completed")
            finalize_artifact_index(output)
    except Exception:
        if not args.dry_run:
            append_event(output, status="failed", event_type="rolling_training_failed", details={"traceback": traceback.format_exc()})
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
