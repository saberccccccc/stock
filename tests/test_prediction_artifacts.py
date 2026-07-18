import json
from pathlib import Path

import pandas as pd
import pytest

from alpha.io import write_alpha_rows
from experiments.prediction_artifacts import (
    DATED_PREDICTION_MANIFEST,
    materialize_frozen_prediction_manifest,
    resolve_split_alpha_path,
)
from experiments.workflow_records import resolve_prediction_paths


def _write_registry(root: Path):
    signal = root / "signals" / "frozen"
    for split, date in (("val_2024", "2024-01-03"), ("test_2025", "2025-01-03")):
        write_alpha_rows(
            signal / split / "alpha_policy.jsonl",
            [{"date": date, "codes": ["B", "A"], "alpha": [0.1, 0.4]}],
        )
    registry = root / "registry" / "candidates.csv"
    registry.parent.mkdir()
    pd.DataFrame(
        [
            {
                "candidate_id": "frozen",
                "status": "legacy",
                "selection_eligible": "true",
                "forward_observation_only": "true",
                "signal_path": str(signal.relative_to(root)),
            }
        ]
    ).to_csv(registry, index=False)
    return registry, signal


def test_frozen_prediction_manifest_is_read_only_hash_checked_and_split_aware(tmp_path):
    registry, signal = _write_registry(tmp_path)
    output = tmp_path / "workflow" / "model_signal"

    path = materialize_frozen_prediction_manifest(
        project_root=tmp_path,
        candidates_csv=registry,
        candidate_ids=["frozen"],
        splits=["val_2024", "test_2025"],
        output_dir=output,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload["candidates"]["frozen"]
    assert path.name == DATED_PREDICTION_MANIFEST
    assert payload["training_executed"] is False
    assert payload["promotion_performed"] is False
    assert candidate["split_alpha_paths"]["val_2024"]["rows"] == 2
    assert candidate["split_alpha_paths"]["test_2025"]["signal_end"] == "2025-01-03"
    assert list(output.iterdir()) == [path]
    assert resolve_split_alpha_path(tmp_path, signal, "val_2024").is_file()


def test_workflow_records_accept_dated_manifest_and_reject_source_mutation(tmp_path):
    registry, signal = _write_registry(tmp_path)
    workflow = tmp_path / "workflow"
    materialize_frozen_prediction_manifest(
        project_root=tmp_path,
        candidates_csv=registry,
        candidate_ids=["frozen"],
        splits=["val_2024", "test_2025"],
        output_dir=workflow / "model_signal",
    )
    config = {
        "model": {"candidate_ids": ["frozen"]},
        "signal": {"candidate_id": "frozen"},
        "evaluation": {
            "selection_splits": ["val_2024", "test_2025"],
            "observation_splits": [],
        },
    }

    resolved = resolve_prediction_paths(workflow, config)
    assert list(resolved) == ["val_2024", "test_2025"]

    write_alpha_rows(
        signal / "val_2024" / "alpha_policy.jsonl",
        [{"date": "2024-01-03", "codes": ["A"], "alpha": [9.0]}],
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_prediction_paths(workflow, config)


def test_frozen_prediction_manifest_rejects_out_of_split_dates(tmp_path):
    registry, signal = _write_registry(tmp_path)
    write_alpha_rows(
        signal / "val_2024" / "alpha_policy.jsonl",
        [{"date": "2025-01-03", "codes": ["A"], "alpha": [0.1]}],
    )

    with pytest.raises(ValueError, match="outside val_2024"):
        materialize_frozen_prediction_manifest(
            project_root=tmp_path,
            candidates_csv=registry,
            candidate_ids=["frozen"],
            splits=["val_2024"],
            output_dir=tmp_path / "out",
        )


def test_frozen_prediction_manifest_accepts_code_mapped_alpha(tmp_path):
    registry, signal = _write_registry(tmp_path)
    write_alpha_rows(
        signal / "val_2024" / "alpha_policy.jsonl",
        [
            {
                "date": "2024-01-03",
                "codes": ["B", "A"],
                "alpha": {"A": 0.4, "B": 0.1},
            }
        ],
    )

    path = materialize_frozen_prediction_manifest(
        project_root=tmp_path,
        candidates_csv=registry,
        candidate_ids=["frozen"],
        splits=["val_2024"],
        output_dir=tmp_path / "out",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["candidates"]["frozen"]["split_alpha_paths"]["val_2024"]["rows"] == 2
