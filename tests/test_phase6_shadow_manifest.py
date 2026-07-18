import json

from run.create_phase6_shadow_manifest import REQUIRED_ARTIFACTS, build_config, build_rollback_policy


def test_phase6_shadow_config_is_observation_only():
    config = build_config()

    assert config["formal_baseline"] == "ledger_path_v3_t0001_nolookahead"
    assert config["research_boundary"]["selection_splits"] == ["val_2024", "test_2025"]
    assert config["research_boundary"]["research_end"] == "2025-12-31"
    assert config["research_boundary"]["forward_start"] == "2026-01-01"
    assert config["research_boundary"]["forward_end"] == "2026-06-30"
    assert config["research_boundary"]["parent_fit_end"] == "2025-12-31"
    assert config["research_boundary"]["forward_selection_allowed"] is False
    assert config["activation"]["automatic_trading"] is False
    assert config["activation"]["automatic_retraining"] is False
    assert config["activation"]["automatic_promotion"] is False
    assert all(candidate["promotion_allowed"] is False for candidate in config["conditional_candidates"])
    assert config["scorecard_contract"]["forward_role"] == "observation_only"
    assert config["research_boundary"]["physical_cache_policy"] == "logical_views_over_physical_superset"
    assert "selection_eligible" in config["scorecard_contract"]["required_fields"]


def test_phase6_requires_boundary_audits():
    assert "reports/qlib_research_framework_20260712/data_boundary_audit_research_20260716.json" in REQUIRED_ARTIFACTS
    assert "reports/qlib_research_framework_20260712/data_boundary_audit_forward_20260716.json" in REQUIRED_ARTIFACTS


def test_phase6_rollback_requires_manual_formal_baseline_fallback():
    policy = build_rollback_policy()

    assert policy["automatic_action"] == "none"
    assert policy["manual_fallback"] == "ledger_path_v3_t0001_nolookahead"
    assert "manifest_or_artifact_hash_mismatch" in policy["hard_integrity_conditions"]
    assert "source_state_changed_after_freeze" in policy["hard_integrity_conditions"]
    assert "never retune from forward data" in policy["response"]
    assert policy["report_only_monitoring_thresholds"]["rolling_20d_active_return_below"] == -0.05


def test_phase6_config_is_json_serializable():
    json.dumps({"config": build_config(), "rollback": build_rollback_policy()})
