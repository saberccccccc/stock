import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.workflow_records import (
    _read_optional_csv,
    build_workflow_record_bundle,
    materialize_signal_inputs,
)


def _write_alpha(path, date, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"date": date, "codes": ["A", "B", "C"], "alpha": scores}) + "\n",
        encoding="utf-8",
    )


def _write_cache(tmp_path):
    raw_path = tmp_path / "labels.dat"
    labels = np.memmap(raw_path, dtype=np.float32, mode="w+", shape=(3, 2, 1))
    labels[:, 0, 0] = [0.03, 0.01, -0.02]
    labels[:, 1, 0] = [-0.01, 0.02, 0.04]
    labels.flush()
    del labels
    meta = {
        "all_codes": ["A", "B", "C"],
        "all_dates": [pd.Timestamp("2024-01-02"), pd.Timestamp("2025-01-02")],
        "max_horizon": 1,
        "label_families": {"oo": {"raw_path": str(raw_path), "date_shift": 0}},
    }
    meta_path = tmp_path / "meta.pkl"
    with meta_path.open("wb") as handle:
        pickle.dump(meta, handle)
    return meta_path


def test_materialize_signal_inputs_aligns_raw_labels(tmp_path):
    alpha = tmp_path / "alpha.jsonl"
    _write_alpha(alpha, "2024-01-02", [3.0, 2.0, 1.0])

    prediction, label, metrics, summary = materialize_signal_inputs(
        {"val_2024": alpha}, _write_cache(tmp_path), "oo", 0, "candidate", tmp_path / "out"
    )

    label_row = json.loads(label.read_text(encoding="utf-8").splitlines()[0])
    assert label_row["label"] == pytest.approx([0.03, 0.01, -0.02])
    assert json.loads(prediction.read_text(encoding="utf-8").splitlines()[0])["model_id"] == "candidate"
    assert pd.read_csv(metrics).iloc[0]["rank_ic"] == pytest.approx(1.0)
    assert summary["label_coverage"] == 1.0


def test_materialize_signal_inputs_applies_oo_lag1_trading_date_shift(tmp_path):
    alpha = tmp_path / "alpha.jsonl"
    _write_alpha(alpha, "2024-01-02", [3.0, 2.0, 1.0])
    meta_path = _write_cache(tmp_path)
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    meta["label_families"]["oo_lag1"] = {"alias_of": "oo", "date_shift": 1}
    with meta_path.open("wb") as handle:
        pickle.dump(meta, handle)

    _, label, _, _ = materialize_signal_inputs(
        {"val_2024": alpha}, meta_path, "oo_lag1", 0, "candidate", tmp_path / "lag1"
    )

    label_row = json.loads(label.read_text(encoding="utf-8").splitlines()[0])
    assert label_row["label"] == pytest.approx([-0.01, 0.02, 0.04])


def test_empty_optional_forward_csv_means_not_requested(tmp_path):
    path = tmp_path / "forward.csv"
    path.write_bytes(b"")

    assert _read_optional_csv(path).empty


def test_completed_workflow_materializes_six_record_bundle(tmp_path):
    project = tmp_path / "project"
    workflow = project / "workflow"
    (workflow / "model_signal").mkdir(parents=True)
    data_dir = project / "data" / "raw"
    data_dir.mkdir(parents=True)
    meta_path = _write_cache(project)
    val_alpha = workflow / "model_signal" / "signals" / "val_2024" / "alpha_policy.jsonl"
    test_alpha = workflow / "model_signal" / "signals" / "test_2025" / "alpha_policy.jsonl"
    _write_alpha(val_alpha, "2024-01-02", [3.0, 2.0, 1.0])
    _write_alpha(test_alpha, "2025-01-02", [1.0, 2.0, 3.0])
    (workflow / "model_signal" / "rolling_manifest.json").write_text(
        json.dumps({
            "split_alpha_paths": {
                "val_2024": {"path": str(val_alpha)},
                "test_2025": {"path": str(test_alpha)},
            }
        }),
        encoding="utf-8",
    )
    config = {
        "signal": {"candidate_id": "candidate"},
        "model": {"config": {"cache_meta": str(meta_path)}},
        "dataset": {"label": {"family": "oo", "horizon_index": 0}},
        "data": {"sources": [{"role": "research_market", "root": str(data_dir)}]},
        "evaluation": {"selection_splits": ["val_2024", "test_2025"], "observation_splits": []},
    }
    (workflow / "workflow_config.json").write_text(json.dumps(config), encoding="utf-8")
    (workflow / "compiled_workflow.json").write_text(
        json.dumps({"project_root": str(project)}), encoding="utf-8"
    )

    detail_dir = workflow / "details"
    detail_dir.mkdir()
    returns = detail_dir / "returns.csv"
    diagnostics = detail_dir / "diagnostics.csv"
    pd.DataFrame([{
        "date": "2024-01-03", "return": 0.01, "benchmark_return": 0.0,
        "active_return": 0.01, "equity_cny": 505000.0,
    }]).to_csv(returns, index=False)
    pd.DataFrame([{
        "date": "2024-01-03", "holdings": "", "cost": 0.001,
        "commission": 0.0002, "stamp_tax": 0.0003, "slippage": 0.0005,
        "market_mult": 1.0, "gross_weight": 0.9,
        "portfolio_beta_60d": 0.8, "portfolio_beta_per_gross_60d": 0.88,
        "portfolio_specific_vol_60d": 0.2,
    }]).to_csv(diagnostics, index=False)
    artifact_paths = {"equity_curve": returns, "diagnostics": diagnostics}
    for name in ("positions", "orders", "rejections", "costs"):
        path = detail_dir / f"{name}.csv"
        pd.DataFrame([{"date": "2024-01-03", "code": "A"}]).to_csv(path, index=False)
        artifact_paths[name] = path

    score_rows = []
    for split in ("val_2024", "test_2025"):
        index_dir = workflow / "ledger" / "run" / "_sweeps" / split
        index_dir.mkdir(parents=True)
        rows = []
        for stress in ("normal", "lag1", "cost2x", "capacity_3pct"):
            for capital in (500000, 1000000):
                rows.append({
                    "sweep_key_sha256": f"{split}-{stress}-{capital}",
                    "alpha_name": "candidate", "stress": stress,
                    "portfolio_value": capital,
                    "signal_start": "2024-01-02", "signal_end": "2025-01-02",
                    "backtest_start": "2024-01-03", "backtest_end": "2025-01-03",
                    **{name: str(path) for name, path in artifact_paths.items()},
                })
                score_rows.append({"candidate": "candidate", "split": split, "stress": stress, "capital": capital})
        pd.DataFrame(rows).to_csv(index_dir / "path_artifact_index.csv", index=False)
    scorecard = workflow / "scorecard"
    scorecard.mkdir()
    pd.DataFrame(score_rows).to_csv(scorecard / "registry_scorecard_long.csv", index=False)
    pd.DataFrame([{"candidate": "candidate", "decision": "hold"}]).to_csv(
        scorecard / "registry_decisions.csv", index=False
    )
    pd.DataFrame(columns=["candidate"]).to_csv(scorecard / "registry_forward_summary.csv", index=False)

    bundle = build_workflow_record_bundle(workflow)

    payload = json.loads(bundle.read_text(encoding="utf-8"))
    assert list(payload["records"]) == [
        "signal", "signal_analysis", "portfolio", "risk_attribution", "stress", "decision"
    ]
    assert (workflow / "records" / "portfolio" / "record_manifest.json").is_file()
    assert json.loads((workflow / "records" / "stress" / "record_manifest.json").read_text(encoding="utf-8"))["payload"]["cells"] == 16
