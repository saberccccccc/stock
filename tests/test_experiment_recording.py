import json

import pytest

from experiments.recording import (
    ARTIFACT_INDEX_NAME,
    EVENTS_NAME,
    MANIFEST_NAME,
    append_event,
    artifact_descriptor,
    canonical_json_hash,
    create_experiment,
    finalize_artifact_index,
    infer_data_scope,
    load_events,
    declared_range,
    not_applicable_range,
    record_artifact,
    source_state,
    validate_manifest_for_formal_use,
)
from experiments.recording import _is_excluded_status_line


def _create(tmp_path):
    return create_experiment(
        tmp_path,
        experiment_id="demo",
        config={"seed": 7, "model": "m0"},
        protocol={"selection_splits": ["val_2024", "test_2025"]},
        cache_contract={"cache_id": "v14-demo", "data_end": "2026-05-18"},
        project_root=tmp_path,
    )


def test_create_experiment_writes_immutable_manifest_and_created_event(tmp_path):
    manifest_path = _create(tmp_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = load_events(tmp_path)

    assert manifest_path.name == MANIFEST_NAME
    assert manifest["experiment_id"] == "demo"
    assert manifest["config_sha256"] == canonical_json_hash({"seed": 7, "model": "m0"})
    assert "source_state" in manifest
    assert manifest["data_scope"]["effective_end"] == "2026-05-18"
    assert manifest["data_scope"]["complete"] is False
    assert manifest["experiment_class"] == "exploratory"
    assert events[0]["status"] == "created"

    with pytest.raises(FileExistsError):
        _create(tmp_path)


def test_data_scope_preserves_explicit_physical_and_effective_ranges():
    scope = infer_data_scope(
        {"research_end": "2026-05-18"},
        {
            "data_scope": {
                "role": "research",
                "effective_start": "2010-01-04",
                "effective_end": "2026-05-18",
                "physical_coverage": {"start": "2010-01-04", "end": "2026-06-29"},
                "complete": True,
            }
        },
    )

    assert scope["physical_coverage"]["end"] == "2026-06-29"
    assert scope["effective_end"] == "2026-05-18"
    assert scope["complete"] is True



def _formal_scope():
    return {
        "stage": "model_signal",
        "data_sources": [
            {"role": "feature_cache", "root": "cache/meta.pkl", "fingerprint": "abc123"}
        ],
        "ranges": {
            "feature_warmup": declared_range("2019-01-01", "2019-12-31"),
            "train": declared_range("2020-01-01", "2022-12-31"),
            "valid": declared_range("2023-01-01", "2023-12-31"),
            "signal": declared_range("2024-01-01", "2024-12-31"),
            "backtest": not_applicable_range("ledger is a separate stage"),
        },
        "max_data_date": "2024-12-31",
        "split_roles": [
            {
                "split": "val_2024",
                "selection_eligible": True,
                "forward_used": False,
            }
        ],
        "transform": {
            "state_sha256": "transform123",
            "fit_range": not_applicable_range("daily cross-sectional transform"),
        },
        "lineage": {},
    }


def test_formal_experiment_requires_complete_scope(tmp_path):
    with pytest.raises(ValueError, match="data_sources"):
        create_experiment(
            tmp_path,
            experiment_id="formal",
            config={},
            protocol={},
            cache_contract={},
            project_root=tmp_path,
            formal=True,
            experiment_scope={"stage": "model_signal"},
        )


def test_formal_experiment_manifest_passes_revalidation(tmp_path):
    path = create_experiment(
        tmp_path,
        experiment_id="formal",
        config={"model": "lgbm"},
        protocol={"selection_splits": ["val_2024"]},
        cache_contract={"cache": "demo"},
        project_root=tmp_path,
        formal=True,
        experiment_scope=_formal_scope(),
    )

    artifact = tmp_path / "model.txt"
    artifact.write_text("model", encoding="utf-8")
    record_artifact(tmp_path, name="model", path=artifact, kind="test_model")
    append_event(tmp_path, status="completed", event_type="formal_completed")
    finalize_artifact_index(tmp_path)

    result = validate_manifest_for_formal_use(path)

    assert result["complete"] is True
    assert result["has_forward"] is False


def test_formal_use_rejects_missing_artifact_index(tmp_path):
    path = create_experiment(
        tmp_path,
        experiment_id="formal-no-artifacts",
        config={},
        protocol={},
        cache_contract={},
        project_root=tmp_path,
        formal=True,
        experiment_scope=_formal_scope(),
    )

    append_event(tmp_path, status="completed", event_type="formal_completed")
    with pytest.raises(ValueError, match="artifact index is missing"):
        validate_manifest_for_formal_use(path)


def test_formal_use_rejects_noncompleted_terminal_state(tmp_path):
    path = create_experiment(
        tmp_path,
        experiment_id="formal-running",
        config={},
        protocol={},
        cache_contract={},
        project_root=tmp_path,
        formal=True,
        experiment_scope=_formal_scope(),
    )
    artifact = tmp_path / "model.txt"
    artifact.write_text("model", encoding="utf-8")
    record_artifact(tmp_path, name="model", path=artifact, kind="test_model")
    finalize_artifact_index(tmp_path)

    with pytest.raises(ValueError, match="not completed"):
        validate_manifest_for_formal_use(path)

    result = validate_manifest_for_formal_use(
        path,
        require_completed=False,
        require_current_index=False,
    )
    assert result["complete"] is True


def test_formal_use_rejects_stale_artifact_index(tmp_path):
    path = create_experiment(
        tmp_path,
        experiment_id="formal-stale-index",
        config={},
        protocol={},
        cache_contract={},
        project_root=tmp_path,
        formal=True,
        experiment_scope=_formal_scope(),
    )
    artifact = tmp_path / "model.txt"
    artifact.write_text("model", encoding="utf-8")
    record_artifact(tmp_path, name="model", path=artifact, kind="test_model")
    append_event(tmp_path, status="completed", event_type="formal_completed")
    finalize_artifact_index(tmp_path)
    append_event(tmp_path, status="completed", event_type="late_completed_event")

    with pytest.raises(ValueError, match="stale"):
        validate_manifest_for_formal_use(path)


def test_forward_formal_scope_requires_pre_2026_parent_freeze(tmp_path):
    scope = _formal_scope()
    scope["split_roles"] = [
        {"split": "forward_2026", "selection_eligible": False, "forward_used": True}
    ]
    scope["ranges"]["signal"] = declared_range("2026-01-01", "2026-06-30")
    scope["max_data_date"] = "2026-06-30"
    scope["lineage"] = {
        "parent_fit_end": "2026-01-01",
        "parent_selection_end": "2025-12-31",
    }

    with pytest.raises(ValueError, match="parent_fit_end"):
        create_experiment(
            tmp_path,
            experiment_id="bad-forward",
            config={},
            protocol={},
            cache_contract={},
            project_root=tmp_path,
            formal=True,
            experiment_scope=scope,
        )


def test_record_artifact_keeps_manifest_static_and_builds_final_index(tmp_path):
    _create(tmp_path)
    alpha = tmp_path / "alpha.jsonl"
    alpha.write_text('{"date":"2024-01-02","codes":[]}\n', encoding="utf-8")

    record_artifact(tmp_path, name="alpha", path=alpha, kind="alpha_jsonl")
    append_event(tmp_path, status="completed", event_type="run_completed")
    index_path = finalize_artifact_index(tmp_path)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_path.name == ARTIFACT_INDEX_NAME
    assert index["artifacts"][0]["name"] == "alpha"
    assert index["artifacts"][0]["artifact"]["sha256"]
    assert len(load_events(tmp_path)) == 3
    assert (tmp_path / EVENTS_NAME).is_file()

    rebuilt = finalize_artifact_index(tmp_path)
    rebuilt_index = json.loads(rebuilt.read_text(encoding="utf-8"))
    assert rebuilt_index["artifacts"] == index["artifacts"]


def test_artifact_descriptor_can_skip_full_hash_for_large_cache_contract(tmp_path):
    artifact = tmp_path / "cache_meta.pkl"
    artifact.write_bytes(b"meta")

    descriptor = artifact_descriptor(artifact, include_sha256=False)

    assert descriptor["bytes"] == 4
    assert "sha256" not in descriptor


def test_source_state_is_stable_for_same_working_tree(tmp_path):
    first = source_state(tmp_path)
    second = source_state(tmp_path)

    assert first["working_tree_status_sha256"] == second["working_tree_status_sha256"]
    assert "source_revision" in first


def test_status_fingerprint_can_exclude_its_own_experiment_output(tmp_path):
    output = tmp_path / "reports" / "exp"

    assert _is_excluded_status_line(
        "?? reports/exp/experiment_manifest.json", tmp_path, (output,)
    )
    assert not _is_excluded_status_line(
        " M source.py", tmp_path, (output,)
    )


def test_append_event_rejects_unknown_status_and_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        append_event(tmp_path, status="running", event_type="bad")
    _create(tmp_path)
    with pytest.raises(ValueError, match="unknown experiment status"):
        append_event(tmp_path, status="unknown", event_type="bad")
