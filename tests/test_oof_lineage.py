import json

import pytest

from experiments.oof_lineage import build_lineage_manifest, inspect_component
from run.build_oof_blends import _parse_weights


def _write_component(tmp_path, *, train_end="2022-12-29", prediction_date="2024-01-02"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    alpha = tmp_path / "alpha.jsonl"
    alpha.write_text(
        json.dumps({"date": prediction_date, "codes": ["000001.SZ"], "alpha": [1.0]}) + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "model.txt"
    model.write_text("model", encoding="utf-8")
    manifest = tmp_path / "rolling_manifest.json"
    payload = {
        "run_mode": "formal",
        "label_end_offset": 7,
        "feature_set": "compact",
        "config": {
            "data": {
                "research_end": "2026-05-18",
                "label_family": "oo_lag1",
                "horizon_index": 4,
            },
            "windows": [
                {
                    "name": "predict_2024",
                    "train_start": "2010-01-01",
                    "train_end": train_end,
                    "valid_start": "2023-01-03",
                    "valid_end": "2023-12-28",
                    "predict_start": "2024-01-01",
                    "predict_end": "2024-12-31",
                }
            ],
        },
        "windows": [
            {
                "name": "predict_2024",
                "counts": {"predict": 1},
                "alpha_path": str(alpha),
                "model_path": str(model),
            }
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_inspect_component_records_model_lineage(tmp_path):
    manifest = _write_component(tmp_path)

    result = inspect_component("compact", manifest)

    window = result["windows"][0]
    assert window["model_id"] == "compact:predict_2024"
    assert window["train_end"] == "2022-12-29"
    assert window["prediction_lineage"][0]["prediction_date"] == "2024-01-02"
    assert window["alpha_sha256"]
    assert result["formal_evidence"] == "executed_rolling_manifest"


def test_inspect_component_accepts_legacy_manifest_with_completed_record(tmp_path):
    manifest = _write_component(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("run_mode")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps({"protocol": {"research_end": "2026-05-18"}}),
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"event_type": "rolling_training_completed", "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "artifact_index.json").write_text(
        json.dumps({"artifacts": [{"name": "rolling_manifest"}]}),
        encoding="utf-8",
    )

    result = inspect_component("legacy", manifest)

    assert result["formal_evidence"] == "legacy_completed_experiment_record"


def test_inspect_component_does_not_trust_invalid_governance_formal_parent(tmp_path):
    manifest = _write_component(tmp_path)
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps({"experiment_class": "formal"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="experiment_scope"):
        inspect_component("invalid-formal-parent", manifest)


def test_inspect_component_rejects_training_after_prediction(tmp_path):
    manifest = _write_component(tmp_path, train_end="2024-01-02")

    with pytest.raises(ValueError, match="unordered"):
        inspect_component("bad", manifest)


def test_build_lineage_rejects_contract_mismatch(tmp_path):
    first = _write_component(tmp_path / "first")
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = _write_component(second_dir)
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["config"]["data"]["label_family"] = "close"
    second.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="label_family"):
        build_lineage_manifest(
            [("a", first), ("b", second)],
            output_path=tmp_path / "lineage.json",
        )


def test_blend_weights_accept_json_numbers():
    assert _parse_weights([0.25, 0.75], 2) == [0.25, 0.75]
