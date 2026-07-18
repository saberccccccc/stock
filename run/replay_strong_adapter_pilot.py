"""Replay completed cross-state strong checkpoints through the governed Adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import load_v9_checkpoint
from backtest.runtime import build_v9_backtest_config
from data.cache_metadata import load_explicit_cross_section_meta
from data.dataset_runtime import ProcessorChains, build_v14_strong_rolling_dataset
from data.providers import DataView, V14MemmapProvider
from data.transform_contract import build_v14_transform_contract
from experiments.model_adapters import TorchStrongAlphaAdapter
from experiments.provenance import (
    RuntimeMetricsSampler,
    command_manifest,
    environment_manifest,
    source_manifest,
    validate_provenance_bundle,
    write_provenance_bundle,
)
from experiments.recording import (
    append_event,
    create_experiment,
    finalize_artifact_index,
    record_artifact,
    sha256_file,
)
from experiments.rolling import RollingWindow, resolve_window_indices
from experiments.strong_rolling import load_json, validate_profile


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pilot",
        default="reports/experiments/strong_e19_staged_pilot_20260717",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _write_legacy_alpha(path: Path, prediction) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in prediction.to_alpha_rows(tie_breaker="legacy_numpy"):
            payload = {
                "date": str(pd.Timestamp(row["date"])),
                "codes": row["codes"],
                "alpha": row["alpha"],
                "n_stocks": row["n_stocks"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def main(argv=None):
    args = parse_args(argv)
    source = _resolve(args.source_pilot)
    output = _resolve(args.output_dir)
    contract = load_json(source / "staged_pilot_contract.json")
    progress = load_json(source / "staged_pilot_progress.json")
    profile_path = Path(contract["profile"]["path"])
    schedule_path = Path(contract["schedule"]["path"])
    cache_meta = Path(contract["data_cache"]["path"])
    profile = load_json(profile_path)
    schedule = load_json(schedule_path)
    validate_profile(profile)
    specs = contract["windows"]
    replay_contract = {
        "schema": "strong_adapter_cross_state_replay_v1",
        "source_pilot": {
            "path": str(source),
            "contract_sha256": contract["contract_sha256"],
            "progress_sha256": sha256_file(source / "staged_pilot_progress.json"),
        },
        "profile": contract["profile"],
        "schedule": contract["schedule"],
        "data_cache": contract["data_cache"],
        "windows": [spec["window"] for spec in specs],
        "adapter": "torch_strong_alpha",
        "score_transform": "v9_rank",
        "selection_allowed": False,
        "promotion_allowed": False,
    }
    if args.dry_run:
        print(json.dumps(replay_contract, ensure_ascii=False, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"immutable replay output already exists: {output}")

    inferred = profile["inferred"]
    expected_dim = int(profile["confirmed"]["architecture"]["input_dim"])
    last_predict_end = max(spec["window"]["predict_end"] for spec in specs)
    meta = load_explicit_cross_section_meta(
        cache_meta,
        project_root=ROOT,
        expected_input_dim=expected_dim,
        required_label_families=(inferred["label_family"], inferred["lag1_label_family"]),
        logical_end=last_predict_end,
    )
    create_experiment(
        output,
        experiment_id=output.name,
        config=replay_contract,
        protocol={"selection_allowed": False, "promotion_allowed": False},
        cache_contract={
            "data_start": str(pd.Timestamp(meta["all_dates"][0]).date()),
            "data_end": last_predict_end,
            "physical_data_end": meta["physical_data_end"],
            "cache_meta": contract["data_cache"],
        },
        project_root=ROOT,
        parent_experiment_ids=(source.name,),
    )
    append_event(output, status="running", event_type="strong_adapter_replay_started")
    results = []
    processor_states = []
    sampler = RuntimeMetricsSampler(interval_seconds=0.25)
    sampler.__enter__()
    runtime_finished = False
    try:
        offset = int(schedule["monthly_schedule"]["label_end_offset"])
        for spec in specs:
            window = RollingWindow.from_mapping(spec["window"])
            indices = resolve_window_indices(meta["all_dates"], window, offset)
            view = DataView.create(
                name=f"strong_adapter_replay:{window.name}",
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
            dataset.fit()
            processor_states.append(
                {"window": window.name, "state": dataset.handler.state_payload()}
            )
            sample = next(iter(dataset.prepare("train", data_key="infer")))
            checkpoint = Path(
                progress["windows"][window.name]["stages"]["multi_downside_e19"]
                ["selected_checkpoint"]["path"]
            )
            runtime_config = build_v9_backtest_config()
            runtime_config.horizon_indices = tuple(inferred["horizon_indices"])
            model, _, regime_dim = load_v9_checkpoint(
                str(checkpoint), [sample], runtime_config, args.device
            )
            adapter = TorchStrongAlphaAdapter(
                model_id=f"strong_adapter_replay:{window.name}",
                config={
                    "device": args.device,
                    "regime_dim": regime_dim,
                    "score_transform": "v9_rank",
                },
                model=model,
            )
            alpha = output / "windows" / window.name / "alpha_raw.jsonl"
            _write_legacy_alpha(alpha, adapter.predict(dataset, "predict"))
            oracle = Path(progress["windows"][window.name]["alpha"]["path"])
            result = {
                "window": window.name,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "oracle_alpha_path": str(oracle),
                "oracle_alpha_sha256": sha256_file(oracle),
                "adapter_alpha_path": str(alpha),
                "adapter_alpha_sha256": sha256_file(alpha),
                "byte_equal": alpha.read_bytes() == oracle.read_bytes(),
                "rows": len([line for line in alpha.read_text(encoding="utf-8").splitlines() if line]),
            }
            if not result["byte_equal"]:
                raise ValueError(f"Adapter alpha differs from staged oracle for {window.name}")
            results.append(result)
            record_artifact(
                output,
                name=f"adapter_alpha:{window.name}",
                path=alpha,
                kind="prediction_frame_compatible_alpha",
            )
            append_event(
                output,
                status="running",
                event_type="strong_adapter_window_replayed",
                details={"window": window.name, "rows": result["rows"], "byte_equal": True},
            )

        summary_path = output / "replay_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema": "strong_adapter_cross_state_replay_summary_v1",
                    "all_byte_equal": all(item["byte_equal"] for item in results),
                    "windows": results,
                    "selection_allowed": False,
                    "promotion_allowed": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record_artifact(output, name="replay_summary", path=summary_path, kind="adapter_parity_summary")
        runtime = sampler.finish(
            status="completed",
            details={
                "windows": len(results),
                "prediction_rows": sum(item["rows"] for item in results),
                "cache_hit": True,
                "training_launched": False,
            },
        )
        runtime_finished = True
        provenance_dir = output / "provenance"
        provenance_bundle = write_provenance_bundle(
            provenance_dir,
            environment=environment_manifest(ROOT),
            source=source_manifest(
                ROOT,
                (
                    __file__,
                    "experiments/model_adapters.py",
                    "experiments/provenance.py",
                    "data/dataset_runtime.py",
                    "data/rolling_samples.py",
                    "data/providers.py",
                    "backtest/engine.py",
                    "backtest/runtime.py",
                ),
                exclude_paths=(output,),
            ),
            data={
                "schema": "data_manifest_v1",
                "cache_meta": contract["data_cache"],
                "physical_coverage": {
                    "start": meta["physical_data_start"],
                    "end": meta["physical_data_end"],
                },
                "logical_view": {
                    "start": min(spec["window"]["train_start"] for spec in specs),
                    "end": last_predict_end,
                    "role": "rolling_train_valid_oos_replay",
                },
                "universe": {
                    "codes": len(meta["all_codes"]),
                    "dates": len(meta["all_dates"]),
                    "input_dim": int(meta["x_dim"]),
                    "risk_dim": int(meta["risk_full_dim"]),
                },
                "label_families": [inferred["label_family"], inferred["lag1_label_family"]],
                "windows": [spec["window"] for spec in specs],
                "pit_status": "frozen_v14_cache_with_declared_logical_view",
            },
            feature_transform={
                "schema": "feature_transform_manifest_v1",
                "cache_transform": build_v14_transform_contract(meta, meta_path=cache_meta),
                "processor_states": processor_states,
                "fit_policy": "each rolling window fits processors on Train only; empty chains preserve cache state",
            },
            command=command_manifest(
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                replay_contract,
            ),
            runtime=runtime,
        )
        validate_provenance_bundle(provenance_bundle)
        for path in sorted(provenance_dir.glob("*.json")):
            record_artifact(
                output,
                name=f"provenance:{path.stem}",
                path=path,
                kind="provenance_manifest_v1",
            )
        append_event(
            output,
            status="completed",
            event_type="strong_adapter_cross_state_replay_completed",
            details={"windows": len(results), "all_byte_equal": True},
        )
    except BaseException as exc:
        if not runtime_finished:
            sampler.finish(
                status="failed",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            runtime_finished = True
        append_event(
            output,
            status="failed",
            event_type="strong_adapter_cross_state_replay_failed",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    finally:
        finalize_artifact_index(output)
    print(json.dumps({"output": str(output), "windows": len(results), "all_byte_equal": True}))


if __name__ == "__main__":
    main()
