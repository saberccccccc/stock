import json
from pathlib import Path

import pytest

from experiments.recording import validate_manifest_for_formal_use
from experiments.workflow import compile_workflow, validate_workflow_config
from run.compile_experiment_workflow import main as compile_main
from run.execute_experiment_workflow import execute_workflow


def _config(data_root):
    return {
        "schema_version": 1,
        "experiment_id": "workflow-test",
        "data": {"sources": [{"role": "market", "root": str(data_root)}], "max_data_date": "2025-12-31"},
        "ranges": {
            "feature_warmup": {"start": "2020-01-01", "end": "2023-12-31"},
            "train": {"start": "2020-01-01", "end": "2023-12-31"},
            "valid": {"start": "2024-01-01", "end": "2024-12-31"},
            "signal": {"start": "2024-01-01", "end": "2025-12-31"},
            "backtest": {"start": "2024-01-01", "end": "2025-12-31"},
        },
        "features": {"set": "test", "transform_state_sha256": "transform-test"},
        "labels": {"family": "oo_lag1", "horizon_index": 4, "tail_purge_days": 7},
        "windows": {"type": "frozen"},
        "model": {"adapter": "frozen_registry_candidate", "candidate_ids": ["candidate"]},
        "checkpoint": {"rule": "frozen"},
        "alpha": {"candidate_ids": ["candidate"]},
        "strategy": {"adapter": "retention"},
        "ledger": {"adapter": "official_open_ledger", "execution_mode": "realistic", "stresses": ["normal", "lag1", "cost2x", "capacity_3pct"], "capitals": [500000, 1000000]},
        "evaluation": {"splits": ["val_2024", "test_2025"]},
        "reports": {"scorecard": "registry"},
    }


def test_compile_workflow_builds_project_native_stage_graph(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("date,close\n2024-01-01,1\n", encoding="utf-8")

    compiled = compile_workflow(_config(data), project_root=tmp_path, output_dir=tmp_path / "out", python="python")

    assert [stage["adapter"] for stage in compiled["stages"]] == [
        "frozen_dated_predictions",
        "official_open_ledger",
        "registry_scorecard",
    ]
    assert compiled["stages"][0]["command"][1] == "run/materialize_frozen_predictions.py"
    assert compiled["stages"][0]["command"].count("--split") == 2
    ledger = compiled["stages"][1]
    assert "run/official_backtest_from_registry.py" in ledger["command"]
    assert compiled["scope"]["split_roles"][0]["selection_eligible"] is True
    assert "--research-data-dir" in ledger["command"]
    assert "--reports-csv" in ledger["command"]
    assert "--append-registry" in ledger["command"]
    assert ledger["command"][ledger["command"].index("--ohlc-backend") + 1] == "legacy"
    assert "--candidate-id" in compiled["stages"][-1]["command"]
    assert compiled["stages"][-1]["command"].count("--expected-split") == 2


def test_workflow_rejects_qlib_or_proxy_execution(tmp_path):
    config = _config(tmp_path)
    config["ledger"]["adapter"] = "qlib_executor"

    with pytest.raises(ValueError, match="official_open_ledger"):
        validate_workflow_config(config)


def test_forward_workflow_requires_parent_frozen_before_2026(tmp_path):
    config = _config(tmp_path)
    config["data"]["max_data_date"] = "2026-06-30"
    config["ranges"]["signal"]["end"] = "2026-06-30"
    config["ranges"]["backtest"]["end"] = "2026-06-30"
    config["evaluation"] = {
        "splits": ["val_2024", "test_2025", "forward_2026"],
        "parent_fit_end": "2026-01-01",
        "parent_selection_end": "2025-12-31",
    }

    with pytest.raises(ValueError, match="parent_fit_end"):
        validate_workflow_config(config)


def test_rolling_workflow_inserts_local_candidate_registry_stage(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("date,close\n2024-01-01,1\n", encoding="utf-8")
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _config(data)
    config["model"] = {"adapter": "rolling_lgbm_alpha", "config": str(model_config)}
    config["alpha"] = {
        "candidate_id": "new_rolling",
        "comparison_candidate_ids": ["baseline"],
    }

    compiled = compile_workflow(
        config,
        project_root=tmp_path,
        output_dir=tmp_path / "out",
        python="python",
    )

    assert [stage["name"] for stage in compiled["stages"]] == [
        "model_signal",
        "candidate_registry",
        "realistic_ledger",
        "scorecard",
    ]
    assert compiled["stages"][2]["depends_on"] == ["candidate_registry"]
    assert "--candidates-csv" in compiled["stages"][2]["command"]


def test_compile_entrypoint_freezes_formal_manifest(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("date,close\n2024-01-01,1\n", encoding="utf-8")
    config_path = tmp_path / "workflow.json"
    config_path.write_text(json.dumps(_config(data)), encoding="utf-8")
    output = tmp_path / "compiled"

    import run.compile_experiment_workflow as runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    compile_main(["--config", str(config_path), "--output-dir", str(output), "--python", "python"])

    compiled = json.loads((output / "compiled_workflow.json").read_text(encoding="utf-8"))
    assert compiled["experiment_id"] == "workflow-test"
    assert validate_manifest_for_formal_use(output / "experiment_manifest.json")["complete"] is True

    with pytest.raises(FileExistsError, match="not empty"):
        compile_main(["--config", str(config_path), "--output-dir", str(output), "--python", "python"])


def test_executor_records_noop_stage_receipt_and_resumes(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("date,close\n2024-01-01,1\n", encoding="utf-8")
    config = _config(data)
    output = tmp_path / "compiled"
    config_path = tmp_path / "workflow.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    import run.compile_experiment_workflow as compiler
    monkeypatch.setattr(compiler, "ROOT", tmp_path)
    compile_main(["--config", str(config_path), "--output-dir", str(output), "--python", "python"])
    compiled_path = output / "compiled_workflow.json"
    import run.execute_experiment_workflow as executor
    monkeypatch.setattr(executor.subprocess, "run", lambda command, cwd: type("Result", (), {"returncode": 0})())

    first = execute_workflow(compiled_path, project_root=tmp_path)
    second = execute_workflow(compiled_path, project_root=tmp_path, resume=True)

    assert first["status"] == "completed"
    assert second["completed_stages"] == ["model_signal", "realistic_ledger", "scorecard"]
    receipt = json.loads((output / "stage_receipts" / "model_signal.json").read_text(encoding="utf-8"))
    assert receipt["command_sha256"]


def test_executor_resumes_failed_rolling_stage_without_overwriting_history(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("date,close\n2024-01-01,1\n", encoding="utf-8")
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _config(data)
    config["model"] = {"adapter": "rolling_lgbm_alpha", "config": str(model_config)}
    config["alpha"] = {"candidate_id": "rolling"}
    output = tmp_path / "compiled"
    config_path = tmp_path / "workflow.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    import run.compile_experiment_workflow as compiler
    monkeypatch.setattr(compiler, "ROOT", tmp_path)
    compile_main(["--config", str(config_path), "--output-dir", str(output), "--python", "python"])
    compiled_path = output / "compiled_workflow.json"
    import run.execute_experiment_workflow as executor
    commands = []
    returncodes = iter([1, 0, 0, 0, 0])

    def fake_run(command, cwd):
        commands.append(list(command))
        return type("Result", (), {"returncode": next(returncodes)})()

    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="model_signal failed"):
        execute_workflow(compiled_path, project_root=tmp_path)

    failed = list((output / "stage_receipts").glob("model_signal.failed.*.json"))
    assert len(failed) == 1
    assert not (output / "stage_receipts" / "model_signal.json").exists()

    result = execute_workflow(compiled_path, project_root=tmp_path, resume=True)

    assert result["status"] == "completed"
    assert "--resume" in commands[1]
    assert failed[0].is_file()
    assert (output / "stage_receipts" / "model_signal.json").is_file()


def _v2_config(data_root, model_config):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "workflow_v2_golden.json").read_text(encoding="utf-8"))
    config["experiment_id"] = "workflow-v2-test"
    config["data"]["sources"] = [
        {"role": "research_market", "provider": "ohlcv_matrix_v1", "root": str(data_root), "pit": True}
    ]
    config["model"]["config"]["path"] = str(model_config)
    config["governance"]["observation_splits"] = []
    config["evaluation"]["observation_splits"] = []
    config["ranges"]["signal"]["end"] = "2025-12-31"
    config["ranges"]["backtest"]["end"] = "2025-12-31"
    config["ranges"]["feature_warmup"]["end"] = "2025-12-31"
    config["data"]["max_data_date"] = "2025-12-31"
    config["dataset"]["segments"].pop("forward")
    return config


def test_workflow_v2_compiles_to_existing_safe_stage_graph(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("date,close\n2024-01-01,1\n", encoding="utf-8")
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _v2_config(data, model_config)

    validation = validate_workflow_config(config)
    compiled = compile_workflow(config, project_root=tmp_path, output_dir=tmp_path / "out", python="python")

    assert validation["source_schema_version"] == 2
    assert compiled["workflow_schema_version"] == 2
    assert [stage["name"] for stage in compiled["stages"]] == [
        "model_signal", "candidate_registry", "realistic_ledger", "scorecard", "standard_records"
    ]
    assert compiled["stages"][-1]["depends_on"] == ["scorecard"]
    assert compiled["stages"][-1]["command"][1] == "run/materialize_workflow_records.py"
    assert compiled["config_sha256"]


def test_workflow_v2_propagates_monthly_execution_backend(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _v2_config(data, model_config)
    config["ledger"].update(
        {
            "ohlc_backend": "monthly",
            "market_daily_store_root": "data/market_daily_candidate_v2",
            "ohlc_monthly_cache_dir": "cache/monthly",
        }
    )

    compiled = compile_workflow(
        config, project_root=tmp_path, output_dir=tmp_path / "out", python="python"
    )
    command = compiled["stages"][2]["command"]

    assert command[command.index("--ohlc-backend") + 1] == "monthly"
    assert command[command.index("--ohlc-monthly-cache-dir") + 1] == "cache/monthly"


def test_workflow_v2_propagates_dual_read_observation(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _v2_config(data, model_config)
    config["ledger"].update(
        {
            "ohlc_backend": "monthly",
            "ohlc_shadow_backend": "csv",
            "ohlc_shadow_report": "reports/dual_read.json",
        }
    )

    compiled = compile_workflow(
        config, project_root=tmp_path, output_dir=tmp_path / "out", python="python"
    )
    command = compiled["stages"][2]["command"]

    assert command[command.index("--ohlc-shadow-backend") + 1] == "csv"
    assert command[command.index("--ohlc-shadow-report") + 1] == "reports/dual_read.json"


def test_workflow_v2_rejects_incomplete_dual_read_contract(tmp_path):
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _v2_config(tmp_path, model_config)
    config["ledger"]["ohlc_shadow_backend"] = "csv"

    with pytest.raises(ValueError, match="requires both"):
        validate_workflow_config(config)


def test_workflow_v2_rejects_forward_as_selection(tmp_path):
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _v2_config(tmp_path, model_config)
    config["governance"]["selection_splits"] = ["val_2024", "test_2025", "forward_2026"]

    with pytest.raises(ValueError, match="schema error"):
        validate_workflow_config(config)


def test_workflow_v2_torch_strong_adapter_compiles_to_existing_runner(tmp_path):
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _v2_config(tmp_path, model_config)
    config["model"]["adapter"] = "torch_strong_alpha"
    profile = tmp_path / "profile.json"
    schedule = tmp_path / "schedule.json"
    cache = tmp_path / "cache.pkl"
    for path in (profile, schedule, cache):
        path.write_text("{}", encoding="utf-8")
    config["model"]["config"] = {
        "profile_path": str(profile),
        "schedule_path": str(schedule),
        "cache_meta": str(cache),
        "transition": "exact",
        "windows": ["oos_2024_01"],
        "device": "cuda",
    }

    validation = validate_workflow_config(config)
    compiled = compile_workflow(
        config, project_root=tmp_path, output_dir=tmp_path / "out", python="python"
    )
    stage = compiled["stages"][0]
    assert validation["declared_model_adapter"] == "torch_strong_alpha"
    assert stage["adapter"] == "torch_strong_alpha"
    assert stage["command"][1] == "run/rolling_strong_staged_pilot.py"
    assert stage["command"][stage["command"].index("--windows") + 1] == "oos_2024_01"
    assert [item["name"] for item in compiled["stages"]] == [
        "model_signal", "candidate_registry", "realistic_ledger", "scorecard", "standard_records"
    ]


def test_workflow_v2_torch_strong_requires_explicit_artifact_contract(tmp_path):
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _v2_config(tmp_path, model_config)
    config["model"]["adapter"] = "torch_strong_alpha"
    config["model"]["config"] = {"profile_path": "profile.json"}

    with pytest.raises(ValueError, match="missing model.config fields"):
        validate_workflow_config(config)


def test_workflow_v2_frozen_rejects_main_candidate_outside_artifacts(tmp_path):
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _v2_config(tmp_path, model_config)
    config["model"] = {
        "adapter": "frozen_artifact",
        "config": {"candidate_ids": ["frozen_a"]},
        "seed": 1,
        "resource_limits": {"ram_gb": 16, "gpu_vram_gb": 8, "threads": 4},
    }
    config["processors"]["state_policy"] = "frozen_parent_no_refit"
    config["signal"]["candidate_id"] = "frozen_b"

    with pytest.raises(ValueError, match="must be present"):
        validate_workflow_config(config)


def test_workflow_v2_frozen_requires_no_refit_processor_policy(tmp_path):
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _v2_config(tmp_path, model_config)
    config["model"] = {
        "adapter": "frozen_artifact",
        "config": {"candidate_ids": ["frozen"]},
        "seed": 1,
        "resource_limits": {"ram_gb": 16, "gpu_vram_gb": 8, "threads": 4},
    }
    config["signal"]["candidate_id"] = "frozen"

    with pytest.raises(ValueError, match="frozen_parent_no_refit"):
        validate_workflow_config(config)


def test_compile_entrypoint_accepts_and_freezes_workflow_v2(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("date,close\n2024-01-01,1\n", encoding="utf-8")
    model_config = tmp_path / "rolling.json"
    model_config.write_text("{}", encoding="utf-8")
    config = _v2_config(data, model_config)
    config_path = tmp_path / "workflow_v2.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "compiled_v2"

    import run.compile_experiment_workflow as compiler

    monkeypatch.setattr(compiler, "ROOT", tmp_path)
    compile_main(["--config", str(config_path), "--output-dir", str(output), "--python", "python"])

    compiled = json.loads((output / "compiled_workflow.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((output / "workflow_config.json").read_text(encoding="utf-8"))
    assert compiled["workflow_schema_version"] == 2
    assert manifest["protocol"]["type"] == "declarative_workflow_v2"
    assert manifest["protocol"]["splits"] == ["val_2024", "test_2025"]
    assert frozen == config
    assert validate_manifest_for_formal_use(output / "experiment_manifest.json")["complete"] is True
