import json

from alpha.io import write_alpha_rows
from backtest.market_data_contract import ExecutionMarketDataContract
from experiments.ledger_evidence import build_ledger_command, validate_experiment_alpha
from experiments.recording import (
    append_event,
    create_experiment,
    declared_range,
    finalize_artifact_index,
    load_events,
    not_applicable_range,
    record_artifact,
)
from run.evaluate_experiment_alpha import main as evaluate_main


def test_research_evidence_uses_frozen_realistic_ledger_contract(tmp_path):
    alpha = tmp_path / "alpha.jsonl"
    write_alpha_rows(alpha, [{"date": "2024-01-02", "codes": ["000001.SZ"], "alpha": [1.0]}])

    info = validate_experiment_alpha(alpha, "val_2024")
    command = build_ledger_command(
        alpha_path=alpha,
        experiment_id="demo",
        split="val_2024",
        output_dir=tmp_path / "ledger",
        python="python",
        market_data=ExecutionMarketDataContract(backend="csv"),
    )

    assert info["days"] == 1
    assert "--execution-constraint-mode" in command
    assert command[command.index("--execution-constraint-mode") + 1] == "realistic"
    assert command[command.index("--max-data-date") + 1] == "2024-12-31"
    assert command[command.index("--ohlc-backend") + 1] == "csv"


def test_experiment_adapter_records_a_dry_run(tmp_path):
    alpha = tmp_path / "alpha.jsonl"
    write_alpha_rows(alpha, [{"date": "2024-01-02", "codes": ["000001.SZ"], "alpha": [1.0]}])
    experiment = tmp_path / "experiment"
    create_experiment(
        experiment,
        experiment_id="demo",
        config={"model": "test"},
        protocol={"selection_split": "val_2024"},
        cache_contract={"cache": "test"},
        project_root=tmp_path,
        formal=True,
        experiment_scope={
            "stage": "model_signal",
            "data_sources": [
                {"role": "test_cache", "root": str(tmp_path), "fingerprint": "test-hash"}
            ],
            "ranges": {
                "feature_warmup": not_applicable_range("unit test"),
                "train": declared_range("2020-01-01", "2022-12-31"),
                "valid": declared_range("2023-01-01", "2023-12-31"),
                "signal": declared_range("2024-01-01", "2024-12-31"),
                "backtest": not_applicable_range("prepared by adapter"),
            },
            "max_data_date": "2024-12-31",
            "split_roles": [
                {"split": "val_2024", "selection_eligible": True, "forward_used": False}
            ],
            "transform": {
                "state_sha256": "test-transform",
                "fit_range": not_applicable_range("unit test"),
            },
            "lineage": {},
        },
    )
    model = experiment / "model.txt"
    model.write_text("model", encoding="utf-8")
    record_artifact(experiment, name="model", path=model, kind="test_model")
    append_event(experiment, status="completed", event_type="model_completed")
    finalize_artifact_index(experiment)

    evaluate_main([
        "--experiment-dir", str(experiment),
        "--experiment-id", "demo",
        "--alpha-path", str(alpha),
        "--split", "val_2024",
        "--dry-run",
    ])

    request = json.loads((experiment / "ledger" / "val_2024" / "ledger_request.json").read_text(encoding="utf-8"))
    assert request["execution_contract"] == "realistic_open_price_share_ledger"
    assert request["alpha"]["signal_start"] == "2024-01-02"
    assert any(event["event_type"] == "ledger_dry_run_prepared" for event in load_events(experiment))
