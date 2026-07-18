"""One-window immutable acceptance run for the strong Torch Model Adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cache_metadata import load_explicit_cross_section_meta
from data.dataset_runtime import ProcessorChains, build_v14_strong_rolling_dataset
from data.providers import DataView, V14MemmapProvider
from experiments.model_adapters import TorchStrongAlphaAdapter
from experiments.recording import (
    append_event,
    create_experiment,
    finalize_artifact_index,
    record_artifact,
    sha256_file,
)
from experiments.rolling import RollingWindow, resolve_window_indices, validate_window
from experiments.strong_adapter_runtime import ExistingStrongTrainerDelegate
from experiments.strong_rolling import build_smoke_train_command, load_json, validate_profile


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="configs/reconstructed_multi_downside_e19_profile_v1.json")
    parser.add_argument("--schedule", default="configs/monthly_rolling_compact_4y6m1m_2024_2025.json")
    parser.add_argument("--cache-meta", default="cache/cross_section_v14_multilabel_open_tech_market_funda_shareh_restr_macro_all_s40_t5_h10_min30_mad_rawlab_end20260518_meta.pkl")
    parser.add_argument("--window", default="oos_2024_01")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--min-free-gib", type=float, default=0.75)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _select_window(schedule, name: str) -> RollingWindow:
    matches = [RollingWindow.from_mapping(item) for item in schedule["windows"] if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"rolling schedule must contain exactly one window named {name}")
    window = matches[0]
    validate_window(window, research_end=schedule["data"]["research_end"])
    return window


def _write_legacy_alpha(path: Path, prediction) -> Path:
    rows = prediction.to_alpha_rows(tie_breaker="legacy_numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            legacy = {
                "date": str(pd.Timestamp(row["date"])),
                "codes": row["codes"],
                "alpha": row["alpha"],
                "n_stocks": row["n_stocks"],
            }
            handle.write(json.dumps(legacy, ensure_ascii=False) + "\n")
    return path


def build_contract(args, profile_path, schedule_path, cache_meta, output, profile, window):
    training_dir = output / "windows" / window.name / "training"
    command = build_smoke_train_command(
        python=sys.executable,
        profile=profile,
        window=window,
        output_dir=training_dir,
        cache_meta=cache_meta,
        epochs=1,
        device=args.device,
    )
    return {
        "schema": "strong_model_adapter_smoke_v1",
        "mode": "one_window_one_epoch_adapter_acceptance",
        "profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
        "schedule": {"path": str(schedule_path), "sha256": sha256_file(schedule_path)},
        "data_cache": {"path": str(cache_meta), "sha256": sha256_file(cache_meta)},
        "window": window.__dict__,
        "train_command": command,
        "adapter": {
            "name": "torch_strong_alpha",
            "score_transform": "v9_rank",
            "prediction_universe": "label_free_x_risk_valid",
            "trainer_delegate": "existing_run_train_v1",
        },
        "selection_allowed": False,
        "promotion_allowed": False,
    }


def main(argv=None):
    args = parse_args(argv)
    profile_path = _resolve(ROOT, args.profile)
    schedule_path = _resolve(ROOT, args.schedule)
    cache_meta = _resolve(ROOT, args.cache_meta)
    output = _resolve(ROOT, args.output_dir)
    profile = load_json(profile_path)
    schedule = load_json(schedule_path)
    validate_profile(profile)
    window = _select_window(schedule, args.window)
    contract = build_contract(args, profile_path, schedule_path, cache_meta, output, profile, window)
    if args.dry_run:
        print(json.dumps(contract, ensure_ascii=False, indent=2, default=str))
        return
    if output.exists():
        raise FileExistsError(f"immutable adapter smoke output already exists: {output}")

    expected_dim = int(profile["confirmed"]["architecture"]["input_dim"])
    inferred = profile["inferred"]
    meta = load_explicit_cross_section_meta(
        cache_meta,
        project_root=ROOT,
        expected_input_dim=expected_dim,
        required_label_families=(inferred["label_family"], inferred["lag1_label_family"]),
        logical_end=window.predict_end,
    )
    offset = int(schedule["monthly_schedule"]["label_end_offset"])
    indices = resolve_window_indices(meta["all_dates"], window, offset)
    expected_counts = next(item for item in schedule["monthly_schedule"]["window_counts"] if item["name"] == window.name)
    actual_counts = {name: len(values) for name, values in indices.items()}
    for name in ("train", "valid", "predict"):
        if actual_counts[name] != int(expected_counts[name]):
            raise ValueError(
                f"schedule index count drift for {name}: expected={expected_counts[name]} actual={actual_counts[name]}"
            )

    view = DataView.create(
        name=f"strong_adapter_smoke:{window.name}",
        physical_root=ROOT / "data" / "raw",
        feature_warmup_start=window.train_start,
        feature_warmup_end=window.train_end,
        task_start=window.train_start,
        task_end=window.predict_end,
        evaluation_start=window.predict_start,
        evaluation_end=window.predict_end,
        max_data_date=window.predict_end,
    )
    provider = V14MemmapProvider(meta=meta, meta_path=cache_meta, data_view=view)
    dataset = build_v14_strong_rolling_dataset(
        provider=provider,
        segment_indices=indices,
        label_family=inferred["label_family"],
        horizon_indices=inferred["horizon_indices"],
        target_horizon_index=max(inferred["horizon_indices"]),
        include_raw_returns=True,
        include_lag1_labels=True,
        lag1_label_family=inferred["lag1_label_family"],
        processors=ProcessorChains(),
    )

    checkpoint = output / "windows" / window.name / "training" / "ultimate_v7_best.pt"
    alpha = output / "windows" / window.name / "alpha_raw.jsonl"
    create_experiment(
        output,
        experiment_id=output.name,
        config=contract,
        protocol={"selection_allowed": False, "promotion_allowed": False},
        cache_contract={
            "data_start": str(pd.Timestamp(meta["all_dates"][0]).date()),
            "data_end": window.predict_end,
            "physical_data_end": meta["physical_data_end"],
            "cache_meta": contract["data_cache"],
        },
        project_root=ROOT,
    )
    append_event(output, status="running", event_type="strong_adapter_smoke_started")
    try:
        delegate = ExistingStrongTrainerDelegate(
            command=contract["train_command"],
            project_root=ROOT,
            checkpoint_path=checkpoint,
            horizon_indices=inferred["horizon_indices"],
            expected_epoch=1,
            expected_input_dim=expected_dim,
            device=args.device,
            min_free_gib=args.min_free_gib,
        )
        adapter = TorchStrongAlphaAdapter(
            model_id=f"strong_adapter_smoke:{window.name}",
            config={
                "device": args.device,
                "score_transform": "v9_rank",
                "checkpoint_rule": "internal_rawtopstable_pilot_only_no_formal_selection",
            },
            fit_delegate=delegate,
        )
        fit_result = adapter.fit(dataset)
        prediction = adapter.predict(dataset, "predict")
        _write_legacy_alpha(alpha, prediction)

        fit_metrics_path = output / "fit_metrics.json"
        fit_metrics_path.write_text(
            json.dumps(dict(fit_result.metrics), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        prediction_manifest_path = output / "prediction_manifest.json"
        prediction_manifest_path.write_text(
            json.dumps(prediction.manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record_artifact(output, name="trainer_checkpoint", path=checkpoint, kind="torch_selected_checkpoint")
        record_artifact(output, name="raw_alpha", path=alpha, kind="prediction_frame_compatible_alpha")
        record_artifact(output, name="fit_metrics", path=fit_metrics_path, kind="model_fit_metrics")
        record_artifact(output, name="prediction_manifest", path=prediction_manifest_path, kind="prediction_manifest")
        append_event(
            output,
            status="completed",
            event_type="strong_adapter_smoke_completed",
            details={
                "window": window.name,
                "alpha_sha256": sha256_file(alpha),
                "checkpoint_sha256": sha256_file(checkpoint),
                "prediction_dates": prediction.manifest()["dates"],
                "selection_allowed": False,
            },
        )
    except BaseException as exc:
        append_event(
            output,
            status="failed",
            event_type="strong_adapter_smoke_failed",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    finally:
        finalize_artifact_index(output)
    print(json.dumps({"output": str(output), "alpha": str(alpha)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
