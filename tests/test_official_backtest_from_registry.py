import json
from types import SimpleNamespace

import pandas as pd
import pytest

import run.official_backtest_from_registry as official
from run.official_backtest_from_registry import (
    append_report_registry,
    build_command,
    record_detailed_ledger_artifacts,
    raise_for_failed_backtests,
)


def make_args():
    return SimpleNamespace(
        target_fracs="0.006",
        hold_fracs="0.10",
        rebalance_bands="0.20",
        stresses="normal",
        portfolio_values="500000",
        max_new_names_list="5",
        exit_hold_fracs="0",
        switch_gap_fracs="0",
        execution_mode="realistic",
        research_data_dir="data/raw",
        forward_data_dir="data/forward_raw",
        resume=False,
    )


def command_value(command, flag):
    return command[command.index(flag) + 1]


def test_official_backtest_uses_frozen_data_for_selection_splits(tmp_path):
    command = build_command(make_args(), "test_2025", "candidate=alpha.jsonl", tmp_path)

    assert command_value(command, "--data-dir") == "data/raw"
    assert command_value(command, "--max-data-date") == "2025-12-31"
    assert "--save-path-details" in command


def test_official_backtest_uses_forward_data_for_forward_split(tmp_path):
    command = build_command(make_args(), "forward_2026", "candidate=alpha.jsonl", tmp_path)

    assert command_value(command, "--data-dir") == "data/forward_raw"
    assert command_value(command, "--max-data-date") == "2026-06-30"


def test_split_alpha_path_resolves_rolling_manifest_window(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "ROOT", tmp_path)
    alpha = tmp_path / "alpha.jsonl"
    alpha.write_text('{"date":"2024-01-02","codes":[],"alpha":[]}\n', encoding="utf-8")
    manifest = tmp_path / "rolling_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config": {
                    "windows": [
                        {
                            "name": "predict_2024",
                            "predict_start": "2024-01-01",
                            "predict_end": "2024-12-31",
                        }
                    ]
                },
                "windows": [{"name": "predict_2024", "alpha_path": str(alpha)}],
            }
        ),
        encoding="utf-8",
    )

    assert official.split_alpha_path(manifest, "val_2024") == alpha.resolve()


def test_registry_writer_derives_forward_role_from_split(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "ROOT", tmp_path)
    monkeypatch.setattr(
        official,
        "validate_manifest_for_formal_use",
        lambda *args, **kwargs: {"complete": True},
    )
    manifest = tmp_path / "experiment_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    summary = tmp_path / "reports" / "candidate" / "normal" / "open_ledger_summary.csv"
    summary.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "portfolio_value": 500000,
                "execution_constraint_mode": "realistic",
                "signal_start": "2026-01-05",
                "signal_end": "2026-06-30",
                "backtest_start": "2026-01-06",
                "backtest_end": "2026-06-29",
            }
        ]
    ).to_csv(summary, index=False)

    append_report_registry("registry/reports.csv", "forward_2026", [summary], manifest)

    row = pd.read_csv(tmp_path / "registry" / "reports.csv").iloc[0]
    assert bool(row["selection_eligible"]) is False
    assert bool(row["is_forward"]) is True
    assert row["evidence_class"] == "formal_experiment"


def test_registry_writer_rejects_result_outside_declared_split(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "ROOT", tmp_path)
    monkeypatch.setattr(official, "validate_manifest_for_formal_use", lambda *args, **kwargs: {"complete": True})
    manifest = tmp_path / "experiment_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    summary = tmp_path / "reports" / "candidate" / "normal" / "open_ledger_summary.csv"
    summary.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "portfolio_value": 500000,
                "execution_constraint_mode": "realistic",
                "signal_start": "2026-01-05",
                "signal_end": "2026-07-01",
                "backtest_start": "2026-01-06",
                "backtest_end": "2026-07-01",
            }
        ]
    ).to_csv(summary, index=False)

    with pytest.raises(ValueError, match="outside"):
        append_report_registry("registry/reports.csv", "forward_2026", [summary], manifest)


def test_detailed_ledger_artifacts_are_recorded_from_index(tmp_path, monkeypatch):
    artifact_names = (
        "equity_curve",
        "diagnostics",
        "positions",
        "orders",
        "rejections",
        "costs",
    )
    row = {"sweep_key_sha256": "abc123"}
    for name in artifact_names:
        path = tmp_path / f"{name}.csv"
        path.write_text("x\n1\n", encoding="utf-8")
        row[name] = str(path)
    index = tmp_path / "path_artifact_index.csv"
    pd.DataFrame([row]).to_csv(index, index=False)
    recorded = []
    monkeypatch.setattr(
        official,
        "record_artifact",
        lambda experiment_dir, **kwargs: recorded.append(kwargs),
    )

    count = record_detailed_ledger_artifacts(tmp_path, "val_2024", index)

    assert count == 6
    assert len(recorded) == 7
    assert recorded[0]["kind"] == "ledger_path_artifact_index_v1"
    assert {item["name"].rsplit(":", 1)[-1] for item in recorded[1:]} == set(artifact_names)


def test_official_wrapper_propagates_subprocess_failure():
    with pytest.raises(RuntimeError, match="subprocess failed"):
        raise_for_failed_backtests(
            [{"split": "val_2024", "status": "failed", "returncode": 1, "output_dir": "out"}]
        )
