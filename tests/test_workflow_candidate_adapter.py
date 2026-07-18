import json

import pandas as pd

from experiments.recording import (
    append_event,
    create_experiment,
    declared_range,
    finalize_artifact_index,
    not_applicable_range,
    record_artifact,
)
from run.materialize_workflow_candidates import materialize_candidates


def test_materialize_experiment_local_rolling_candidate(tmp_path):
    experiment = tmp_path / "rolling"
    create_experiment(
        experiment,
        experiment_id="rolling",
        config={},
        protocol={},
        cache_contract={},
        project_root=tmp_path,
        formal=True,
        experiment_scope={
            "stage": "model_signal",
            "data_sources": [{"role": "cache", "root": str(tmp_path), "fingerprint": "hash"}],
            "ranges": {
                "feature_warmup": not_applicable_range("test"),
                "train": declared_range("2020-01-01", "2022-12-31"),
                "valid": declared_range("2023-01-01", "2023-12-31"),
                "signal": declared_range("2024-01-01", "2024-12-31"),
                "backtest": not_applicable_range("test"),
            },
            "max_data_date": "2024-12-31",
            "split_roles": [
                {"split": "val_2024", "selection_eligible": True, "forward_used": False}
            ],
            "transform": {
                "state_sha256": "transform",
                "fit_range": not_applicable_range("test"),
            },
            "lineage": {},
        },
    )
    rolling = experiment / "rolling_manifest.json"
    rolling.write_text(json.dumps({"windows": []}), encoding="utf-8")
    record_artifact(experiment, name="rolling_manifest", path=rolling, kind="rolling_manifest")
    append_event(experiment, status="completed", event_type="rolling_completed")
    finalize_artifact_index(experiment)
    source = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "baseline",
                "display_name": "Baseline",
                "family": "base",
                "status": "formal_baseline",
                "selection_eligible": True,
                "forward_observation_only": True,
                "signal_path": "signals",
                "backtest_path": "",
                "execution_mode": "realistic",
                "base_alpha": "",
                "created_at": "2026-01-01",
                "notes": "",
            }
        ]
    ).to_csv(source, index=False)

    output = materialize_candidates(
        rolling_experiment_dir=experiment,
        candidate_id="new_model",
        comparison_candidate_ids=["baseline"],
        source_candidates_csv=source,
        output=tmp_path / "local_candidates.csv",
    )

    frame = pd.read_csv(output)
    assert frame["candidate_id"].tolist() == ["baseline", "new_model"]
    assert frame.loc[frame["candidate_id"].eq("new_model"), "signal_path"].iloc[0] == str(rolling)
