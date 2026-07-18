import json
from pathlib import Path

from experiments.recording import append_event, create_experiment, record_artifact, sha256_file
from run.materialize_strong_rolling_manifest import materialize


def test_materializer_preserves_exploratory_status_and_source_hashes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    alpha = source / "alpha.jsonl"
    alpha.write_text(
        '{"date":"2024-01-02","codes":["000001.SZ"],"alpha":[1.0]}\n',
        encoding="utf-8",
    )
    model = source / "model.pt"
    model.write_bytes(b"model")
    create_experiment(
        source,
        experiment_id="source",
        config={"kind": "pilot"},
        protocol={"selection_allowed": False},
        cache_contract={},
        project_root=tmp_path,
    )
    record_artifact(source, name="alpha", path=alpha, kind="alpha")
    append_event(source, status="completed", event_type="completed")
    artifact = lambda path: {"path": str(path), "sha256": sha256_file(path)}
    contract = {
        "contract_sha256": "contract",
        "profile": {"path": "p", "sha256": "p"},
        "schedule": {"path": "s", "sha256": "s"},
        "windows": [{"window": {
            "name": "oos_2024_01", "train_start": "2019-01-01", "train_end": "2022-12-31",
            "valid_start": "2023-01-01", "valid_end": "2023-12-31",
            "predict_start": "2024-01-02", "predict_end": "2024-01-31"
        }}],
    }
    progress = {"windows": {"oos_2024_01": {
        "alpha": artifact(alpha),
        "stages": {"multi_downside_e19": {
            "exact_checkpoint": artifact(model), "selected_checkpoint": artifact(model)
        }},
    }}}
    (source / "staged_pilot_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (source / "staged_pilot_progress.json").write_text(json.dumps(progress), encoding="utf-8")

    manifest_path = materialize(source, tmp_path / "adapted", "adapted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = json.loads((tmp_path / "adapted" / "experiment_manifest.json").read_text(encoding="utf-8"))

    assert manifest["learner_adapter"] == "torch_strong_alpha"
    assert manifest["split_alpha_paths"]["val_2024"]["rows"] == 1
    assert experiment["experiment_class"] == "exploratory"
    assert experiment["formal_completeness"]["complete"] is False
