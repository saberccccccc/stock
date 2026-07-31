import json
from pathlib import Path

import pandas as pd
import pytest

from backtest.market_data_contract import (
    ExecutionMarketDataContract,
    configured_default_backend,
)
from experiments.recording import sha256_file
from experiments.shadow_daily import (
    _replay_market_data_contract,
    build_sweep_command,
    materialize_daily_packets,
    run_daily_shadow,
    validate_daily_shadow_run,
)
from experiments.shadow_lifecycle import create_shadow_lifecycle


def _workflow(tmp_path):
    workflow = tmp_path / "workflow"
    records = workflow / "records"
    records.mkdir(parents=True)
    entries = {}
    for name in ("signal", "signal_analysis", "portfolio", "risk_attribution", "stress", "decision"):
        manifest = records / name / "record_manifest.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({"schema": "standard_record_v1"}) + "\n", encoding="utf-8")
        entries[name] = {"manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest)}
    (records / "bundle_manifest.json").write_text(
        json.dumps({"schema": "standard_record_bundle_v1", "records": entries}) + "\n",
        encoding="utf-8",
    )
    for name in ("workflow_config.json", "compiled_workflow.json", "experiment_manifest.json"):
        (workflow / name).write_text("{}\n", encoding="utf-8")
    return workflow


def _lifecycle(tmp_path):
    lifecycle = tmp_path / "lifecycle"
    create_shadow_lifecycle(
        lifecycle,
        lifecycle_id="daily-test",
        candidate_id="candidate",
        workflow_dir=_workflow(tmp_path),
        actor="tester",
        reason="test",
    )
    return lifecycle


def _execution(tmp_path):
    execution = tmp_path / "execution"
    paths = execution / "paths"
    paths.mkdir(parents=True)
    frames = {
        "equity_curve": pd.DataFrame([{"date": "2026-01-06", "return": 0.01, "active_return": 0.02, "equity_cny": 505000}]),
        "diagnostics": pd.DataFrame([{"date": "2026-01-06", "holdings": 2, "gross_weight": 0.5, "turnover": 0.5}]),
        "positions": pd.DataFrame([{"date": "2026-01-06", "code": "A", "weight": 0.25}]),
        "orders": pd.DataFrame([{"date": "2026-01-06", "code": "A", "status": "filled"}]),
        "rejections": pd.DataFrame(columns=["date", "code", "status"]),
        "costs": pd.DataFrame([{"date": "2026-01-06", "code": "A", "total_cost_cny": 5.0}]),
    }
    descriptor = {
        "sweep_key_sha256": "abc",
        "alpha_name": "candidate",
        "stress": "normal",
        "portfolio_value": 500000.0,
    }
    for name, frame in frames.items():
        path = paths / f"{name}.csv"
        frame.to_csv(path, index=False)
        descriptor[name] = str(path)
    pd.DataFrame([descriptor]).to_csv(execution / "path_artifact_index.csv", index=False)
    return execution


def _alpha(tmp_path):
    path = tmp_path / "alpha.jsonl"
    path.write_text(json.dumps({"date": "2026-01-05", "codes": ["A", "B"], "alpha": [0.2, 0.1]}) + "\n", encoding="utf-8")
    return path


def test_build_sweep_command_fixes_realistic_single_path_contract(tmp_path):
    command = build_sweep_command(
        project_root=tmp_path,
        alpha_path=tmp_path / "alpha.jsonl",
        candidate_id="candidate",
        execution_dir=tmp_path / "execution",
        data_dir=tmp_path / "data",
        start_date="2026-01-05",
        end_date="2026-01-30",
        max_data_date="2026-01-30",
        portfolio_value=500000,
        market_data=ExecutionMarketDataContract(
            backend="monthly",
            market_daily_store_root=tmp_path / "store",
            monthly_cache_root=tmp_path / "monthly",
        ),
        python_executable="python",
    )
    assert command[command.index("--execution-constraint-mode") + 1] == "realistic"
    assert command[command.index("--stresses") + 1] == "normal"
    assert command[command.index("--portfolio-values") + 1] == "500000.0"
    assert "--save-path-details" in command
    assert command[command.index("--ohlc-backend") + 1] == "monthly"
    assert command[command.index("--ohlc-monthly-cache-dir") + 1] == str(tmp_path / "monthly")


def test_materialize_and_validate_daily_shadow_packet(tmp_path):
    lifecycle = _lifecycle(tmp_path)
    snapshot = __import__("experiments.shadow_lifecycle", fromlist=["validate_shadow_lifecycle"]).validate_shadow_lifecycle(lifecycle)
    run_dir = tmp_path / "run"
    execution = _execution(run_dir)
    daily, semantic = materialize_daily_packets(
        run_dir=run_dir,
        lifecycle_snapshot=snapshot,
        alpha_path=_alpha(tmp_path),
        candidate_id="candidate",
        start_date="2026-01-05",
        end_date="2026-01-05",
        portfolio_value=500000,
        execution_dir=execution,
    )
    assert len(daily) == 1
    packet = json.loads(Path(daily[0]["manifest"]["path"]).read_text(encoding="utf-8"))
    assert packet["execution_date"] == "2026-01-06"
    assert packet["checks"]["data_ready"] is True
    assert packet["execution_contract"]["constraint_mode"] == "realistic"
    assert semantic

    second_run = tmp_path / "second-run"
    second_execution = _execution(second_run)
    _, second_semantic = materialize_daily_packets(
        run_dir=second_run,
        lifecycle_snapshot=snapshot,
        alpha_path=tmp_path / "alpha.jsonl",
        candidate_id="candidate",
        start_date="2026-01-05",
        end_date="2026-01-05",
        portfolio_value=500000,
        execution_dir=second_execution,
    )
    assert second_semantic == semantic


def test_prepared_lifecycle_rejects_formal_observation_but_allows_replay(tmp_path):
    lifecycle = _lifecycle(tmp_path)
    alpha = _alpha(tmp_path)
    run_dir = tmp_path / "run"
    _execution(run_dir)
    with pytest.raises(ValueError, match="state=shadow"):
        run_daily_shadow(
            project_root=tmp_path,
            lifecycle_dir=lifecycle,
            run_dir=tmp_path / "formal",
            alpha_path=alpha,
            candidate_id="candidate",
            data_dir=tmp_path,
            start_date="2026-01-05",
            end_date="2026-01-05",
            max_data_date="2026-01-06",
            portfolio_value=500000,
            mode="shadow_observation",
            actor="tester",
            reason="test",
            execute=False,
        )
    manifest = run_daily_shadow(
        project_root=tmp_path,
        lifecycle_dir=lifecycle,
        run_dir=run_dir,
        alpha_path=alpha,
        candidate_id="candidate",
        data_dir=tmp_path,
        start_date="2026-01-05",
        end_date="2026-01-05",
        max_data_date="2026-01-06",
        portfolio_value=500000,
        mode="historical_replay",
        execute=False,
    )
    validated = validate_daily_shadow_run(run_dir)
    assert manifest.is_file()
    assert validated["recorded_to_lifecycle"] is False
    assert validated["market_data"]["backend"] == configured_default_backend()


def test_historical_monthly_replay_requires_frozen_candidate_paths():
    with pytest.raises(ValueError, match="frozen store and cache roots"):
        _replay_market_data_contract(
            {"market_data": {"backend": "monthly"}}
        )

    contract = _replay_market_data_contract(
        {
            "market_data": {
                "backend": "monthly",
                "market_daily_store_root": "data/frozen_store",
                "monthly_cache_root": "cache/frozen_cache",
            }
        }
    )

    assert contract.market_daily_store_root == "data/frozen_store"
    assert contract.monthly_cache_root == "cache/frozen_cache"
