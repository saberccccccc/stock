import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = ROOT / "reports" / "qlib_alignment_20260717" / "qlib_alignment_matrix.json"
SCHEMA = ROOT / "schemas" / "workflow_v2.schema.json"
GOLDEN = ROOT / "configs" / "workflow_v2_golden.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_alignment_matrix_has_complete_unique_component_mapping():
    matrix = _load(ALIGNMENT)
    assert matrix["qlib_reference"]["commit"] == "d5379c520f66a39953bad76234a7019a72796fd0"
    components = matrix["components"]
    ids = [item["id"] for item in components]
    assert len(ids) == len(set(ids))
    assert {
        "workflow_task", "provider", "datahandler_lp", "dataseth", "model",
        "recorder", "record_templates", "rolling", "rolling_ensemble",
        "strategy", "executor", "online_manager",
    }.issubset(ids)
    allowed = set(matrix["status_values"])
    for component in components:
        assert component["status"] in allowed
        assert component["qlib_source"]
        assert component["project_modules"]
        assert component["decision"]
        assert component["phase"].startswith("Q")
        assert component["acceptance_tests"]
        for project_path in component["project_modules"]:
            assert (ROOT / project_path).exists(), project_path


def test_workflow_v2_schema_and_golden_config_are_valid():
    schema = _load(SCHEMA)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(_load(GOLDEN))


def test_golden_workflow_preserves_a_share_research_invariants():
    config = _load(GOLDEN)
    assert config["governance"]["selection_splits"] == ["val_2024", "test_2025"]
    assert config["governance"]["observation_splits"] == ["forward_2026"]
    assert config["governance"]["forward_selection_allowed"] is False
    assert config["checkpoint"]["selection_segment"] == "valid"
    assert config["checkpoint"]["oos_selection_allowed"] is False
    assert config["ledger"]["adapter"] == "official_open_ledger"
    assert config["ledger"]["fill_price"] == "open"
    assert set(config["ledger"]["capitals"]) == {500_000, 1_000_000}
    assert set(config["ledger"]["stresses"]) == {"normal", "lag1", "cost2x", "capacity_3pct"}
    assert config["evaluation"]["ic_role"] == "diagnostic_only"


def test_workflow_v2_is_explicitly_a_draft_not_runtime_schema_v1():
    schema = _load(SCHEMA)
    golden = _load(GOLDEN)
    assert schema["properties"]["schema_version"]["const"] == 2
    assert golden["schema_version"] == 2

    from experiments.workflow import WORKFLOW_SCHEMA_VERSION

    assert WORKFLOW_SCHEMA_VERSION == 1
