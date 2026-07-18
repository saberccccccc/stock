import json

from experiments.tuning import build_trial_config, load_tuning_spec


def test_tuning_spec_allows_one_model_override(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"max_trials": 1, "trials": [{"trial_id": "t01", "overrides": {"num_leaves": 15}}]}), encoding="utf-8")
    spec = load_tuning_spec(path)
    assert spec["trials"][0]["trial_id"] == "t01"


def test_trial_config_preserves_data_and_changes_only_model_component():
    base = {"name": "base", "data": {"research_end": "2026-05-18"}, "model": {"num_leaves": 31}}
    trial = build_trial_config(base, {"trial_id": "t01", "overrides": {"num_leaves": 15}})
    assert trial["data"] == base["data"]
    assert trial["model"]["num_leaves"] == 15
    assert trial["tuning"]["trial_id"] == "t01"
