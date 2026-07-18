import json
from pathlib import Path

import pytest

from experiments.record_templates import (
    RecordBuild,
    RecordBundleRunner,
    RecordContext,
    materialize_record_spec,
    standard_record_templates,
)


def _workflow():
    return {
        "evaluation": {
            "selection_splits": ["val_2024", "test_2025"],
            "observation_splits": ["forward_2026"],
        }
    }


def _producer(name, payload, artifact_names):
    def produce(context, parents, output_dir):
        artifacts = {}
        for artifact_name in artifact_names:
            path = output_dir / f"{artifact_name}.json"
            path.write_text(json.dumps({"record": name, "artifact": artifact_name}) + "\n", encoding="utf-8")
            artifacts[artifact_name] = path
        return RecordBuild(payload=payload, artifacts=artifacts)

    return produce


def _producers(*, forward_used=False):
    return {
        "signal": _producer(
            "signal",
            {
                "schema": "prediction_frame_v1",
                "signal_start": "2024-01-02",
                "signal_end": "2026-06-30",
                "rows": 100,
                "asof_start": "2024-01-02T15:00:00",
                "asof_end": "2026-06-30T15:00:00",
            },
            ["prediction", "label"],
        ),
        "signal_analysis": _producer(
            "signal_analysis",
            {
                "selection_splits": ["val_2024", "test_2025"],
                "observation_splits": ["forward_2026"],
                "metrics": {"rank_ic": 0.08, "top30_return": 0.12},
            },
            ["signal_metrics"],
        ),
        "portfolio": _producer(
            "portfolio",
            {
                "execution_adapter": "official_open_ledger",
                "fill_price": "open",
                "backtest_start": "2024-01-02",
                "backtest_end": "2026-06-30",
                "metrics": {"sharpe": 1.5},
            },
            ["equity_curve", "positions", "orders", "rejections", "costs"],
        ),
        "risk_attribution": _producer(
            "risk_attribution",
            {"metrics": {"beta": 0.8, "specific_vol": 0.2, "industry_hhi": 0.1}},
            ["risk_metrics"],
        ),
        "stress": _producer(
            "stress",
            {
                "capitals": [500_000, 1_000_000],
                "stresses": ["normal", "lag1", "cost2x", "capacity_3pct"],
                "cells": 24,
            },
            ["stress_scorecard"],
        ),
        "decision": _producer(
            "decision",
            {
                "selection_splits": ["val_2024", "test_2025"],
                "observation_splits": ["forward_2026"],
                "selection_result": {"status": "hold"},
                "forward_observation": {"status": "observed_only"},
                "forward_used_for_selection": forward_used,
            },
            ["decision_report"],
        ),
    }


def test_record_dependency_failure_is_explicit(tmp_path):
    context = RecordContext.from_workflow(tmp_path / "experiment", _workflow())
    runner = RecordBundleRunner(context, standard_record_templates(_producers()))

    with pytest.raises(RuntimeError, match="missing parent records"):
        runner.run_record("portfolio")


def test_complete_bundle_has_standard_layout_hashes_and_dependencies(tmp_path):
    context = RecordContext.from_workflow(tmp_path / "experiment", _workflow())
    runner = RecordBundleRunner(context, standard_record_templates(_producers()))
    bundle_path = runner.run_all()

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert list(bundle["records"]) == [
        "signal", "signal_analysis", "portfolio", "risk_attribution", "stress", "decision"
    ]
    assert bundle["selection_splits"] == ["val_2024", "test_2025"]
    assert bundle["observation_splits"] == ["forward_2026"]
    decision = json.loads((context.experiment_dir / "records" / "decision" / "record_manifest.json").read_text(encoding="utf-8"))
    assert set(decision["depends_on"]) == {"signal_analysis", "risk_attribution", "stress"}
    assert (context.experiment_dir / "records" / "portfolio" / "orders.json").is_file()


def test_forward_cannot_enter_decision_selection(tmp_path):
    context = RecordContext.from_workflow(tmp_path / "experiment", _workflow())
    runner = RecordBundleRunner(context, standard_record_templates(_producers(forward_used=True)))

    for name in ("signal", "signal_analysis", "portfolio", "risk_attribution", "stress"):
        runner.run_record(name)
    with pytest.raises(ValueError, match="Forward observation"):
        runner.run_record("decision")
    assert not (context.experiment_dir / "records" / "decision").exists()


def test_portfolio_record_rejects_nonofficial_execution(tmp_path):
    producers = _producers()
    bad_payload = {
        "execution_adapter": "qlib_executor",
        "fill_price": "close",
        "backtest_start": "2024-01-02",
        "backtest_end": "2025-12-31",
        "metrics": {},
    }
    producers["portfolio"] = _producer(
        "portfolio", bad_payload, ["equity_curve", "positions", "orders", "rejections", "costs"]
    )
    context = RecordContext.from_workflow(tmp_path / "experiment", _workflow())
    runner = RecordBundleRunner(context, standard_record_templates(producers))
    runner.run_record("signal")

    with pytest.raises(ValueError, match="official open-price ledger"):
        runner.run_record("portfolio")


def test_record_module_does_not_embed_a_second_ledger():
    source = (Path(__file__).resolve().parents[1] / "experiments" / "record_templates.py").read_text(encoding="utf-8")
    assert "from backtest" not in source
    assert "import backtest" not in source


def test_record_spec_materializer_resolves_relative_artifacts(tmp_path):
    artifact_dir = tmp_path / "source_artifacts"
    artifact_dir.mkdir()
    records = {}
    for name, producer in _producers().items():
        build_dir = tmp_path / "build" / name
        build_dir.mkdir(parents=True)
        build = producer(None, {}, build_dir)
        records[name] = {
            "payload": dict(build.payload),
            "artifacts": {
                artifact_name: str(Path(path).relative_to(tmp_path))
                for artifact_name, path in build.artifacts.items()
            },
        }
    spec = {"schema_version": 1, "workflow": _workflow(), "records": records}
    spec_path = tmp_path / "record_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    bundle = materialize_record_spec(spec_path, tmp_path / "experiment")

    assert bundle.is_file()
    assert json.loads(bundle.read_text(encoding="utf-8"))["schema"] == "standard_record_bundle_v1"
