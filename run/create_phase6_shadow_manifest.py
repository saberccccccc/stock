"""Freeze a research-only Phase 6 shadow manifest.

This creates provenance and rollback artifacts for observation preparation. It
does not register a candidate, generate forward alpha, retrain a model, or
activate automatic trading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.recording import (
    append_event,
    create_experiment,
    declared_range,
    finalize_artifact_index,
    not_applicable_range,
    record_artifact,
    sha256_file,
)
from core.research_protocol import (
    FORWARD_END_DATE,
    FORWARD_START_DATE,
    RESEARCH_END_DATE,
    SELECTION_SPLITS,
)


REQUIRED_ARTIFACTS = (
    "QLIB_ADOPTION_PLAN_20260712.md",
    "ARCHITECTURE.md",
    "RESEARCH_PROTOCOL.md",
    "experiments/recording.py",
    "backtest/open_ledger.py",
    "configs/state_aware_portfolio_pilot_20260715.json",
    "reports/qlib_research_framework_20260712/state_aware_phase5_closure_20260715.md",
    "reports/qlib_research_framework_20260712/state_aware_pilot_risk_rank_delta_20260715.csv",
    "reports/qlib_research_framework_20260712/state_aware_pilot_risk_suppress_delta_20260715.csv",
    "reports/experiments/oof_baseline_components_20260715/blends/compact_v14_eq_rank.jsonl",
    "reports/qlib_research_framework_20260712/data_boundary_audit_research_20260716.json",
    "reports/qlib_research_framework_20260712/data_boundary_audit_forward_20260716.json",
    "registry/baselines.yaml",
    "registry/decision_rules.json",
)


def build_config():
    return {
        "schema_version": 1,
        "shadow_type": "research_observation_only",
        "formal_baseline": "ledger_path_v3_t0001_nolookahead",
        "conditional_candidates": [
            {
                "id": "compact_v14_eq_rank_risk_rank_t035_p010",
                "status": "conditional_research",
                "promotion_allowed": False,
            },
            {
                "id": "compact_v14_eq_rank_risk_suppress_d015",
                "status": "exploratory",
                "promotion_allowed": False,
            },
        ],
        "research_boundary": {
            "research_end": str(RESEARCH_END_DATE.date()),
            "forward_start": str(FORWARD_START_DATE.date()),
            "forward_end": str(FORWARD_END_DATE.date()),
            "selection_splits": list(SELECTION_SPLITS),
            "forward_selection_allowed": False,
            "parent_fit_end": str(RESEARCH_END_DATE.date()),
            "parent_selection_end": str(RESEARCH_END_DATE.date()),
            "physical_cache_policy": "logical_views_over_physical_superset",
        },
        "alpha": {
            "id": "compact_v14_eq_rank",
            "mode": "chronological_rank_mean",
            "source": "reports/experiments/oof_baseline_components_20260715/blends/compact_v14_eq_rank.jsonl",
        },
        "execution": {
            "mode": "realistic",
            "price_field": "open",
            "ledger": "backtest.open_ledger",
            "stresses": ["normal", "lag1", "cost2x", "capacity_3pct"],
            "capital_cny": [500000, 1000000],
        },
        "activation": {
            "automatic_trading": False,
            "automatic_retraining": False,
            "automatic_promotion": False,
            "current_forward_activation_allowed": False,
            "required_source_state": "clean_or_explicit_snapshot",
            "reason": "Phase 5 has no formally promoted candidate and the current worktree is not a clean release snapshot.",
        },
        "scorecard_contract": {
            "required_fields": [
                "signal_start",
                "signal_end",
                "backtest_start",
                "backtest_end",
                "split",
                "is_forward",
                "selection_eligible",
                "model_alpha_contribution",
                "execution_rejection_contribution",
                "cost_contribution",
                "portfolio_constraint_contribution",
            ],
            "forward_role": "observation_only",
        },
    }


def build_rollback_policy():
    return {
        "schema_version": 1,
        "automatic_action": "none",
        "manual_fallback": "ledger_path_v3_t0001_nolookahead",
        "hard_integrity_conditions": [
            "manifest_or_artifact_hash_mismatch",
            "alpha_date_outside_forward_boundary_or_duplicate_date",
            "missing_required_ohlc_or_execution_mask",
            "ledger_cash_share_reconciliation_failure",
            "source_state_changed_after_freeze",
        ],
        "report_only_monitoring_thresholds": {
            "rolling_20d_active_return_below": -0.05,
            "rolling_20d_drawdown_excess_vs_baseline_above": 0.05,
            "execution_rejection_rate_above": 0.25,
        },
        "response": "manual_review_then_fallback_to_formal_baseline; never retune from forward data",
    }


def _write_exclusive(path: Path, value):
    with path.open("x", encoding="utf-8") as handle:
        if isinstance(value, str):
            handle.write(value)
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/experiments/phase6_shadow_bundle_20260715",
    )
    parser.add_argument(
        "--experiment-id",
        default="phase6_shadow_bundle_20260715",
    )
    parser.add_argument("--project-root", default=str(ROOT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"shadow output directory is not empty: {output_dir}")

    missing = [path for path in REQUIRED_ARTIFACTS if not (project_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required Phase 6 artifacts are missing: {missing}")

    config = build_config()
    protocol = {
        "research_end": str(RESEARCH_END_DATE.date()),
        "forward_start": str(FORWARD_START_DATE.date()),
        "forward_end": str(FORWARD_END_DATE.date()),
        "selection_splits": list(SELECTION_SPLITS),
        "observation_splits": ["forward_2026"],
        "execution_mode": "realistic",
        "forward_policy": "observation_only",
    }
    cache_contract = {
        "research_data_root": "data/raw",
        "forward_data_root": "data/forward_raw",
        "research_data_policy": "logical_views_over_physical_superset",
        "data_scope": {
            "research": {
                "role": "research",
                "effective_start": "2010-01-04",
                "effective_end": str(RESEARCH_END_DATE.date()),
                "physical_coverage": {"start": "2010-01-04", "end": "2026-06-29"},
                "complete": True,
            },
            "forward": {
                "role": "forward",
                "effective_start": str(FORWARD_START_DATE.date()),
                "effective_end": str(FORWARD_END_DATE.date()),
                "physical_coverage": {"start": "2010-01-04", "end": "2026-06-30"},
                "complete": True,
            },
        },
        "research_boundary_audit": "reports/qlib_research_framework_20260712/data_boundary_audit_research_20260716.json",
        "forward_boundary_audit": "reports/qlib_research_framework_20260712/data_boundary_audit_forward_20260716.json",
        "alpha_cache": "reports/experiments/oof_baseline_components_20260715/blends/compact_v14_eq_rank.jsonl",
        "historical_st_coverage": "declared_execution_coverage_gate",
    }
    alpha_path = project_root / cache_contract["alpha_cache"]
    scope = {
        "stage": "shadow_preparation",
        "data_sources": [
            {
                "role": "research_compatibility_root",
                "root": str((project_root / "data/raw").resolve()),
                "fingerprint": {
                    "kind": "boundary_audit_sha256",
                    "value": sha256_file(project_root / cache_contract["research_boundary_audit"]),
                },
            },
            {
                "role": "forward_compatibility_root",
                "root": str((project_root / "data/forward_raw").resolve()),
                "fingerprint": {
                    "kind": "boundary_audit_sha256",
                    "value": sha256_file(project_root / cache_contract["forward_boundary_audit"]),
                },
            },
        ],
        "ranges": {
            "feature_warmup": not_applicable_range("owned by frozen parent alpha"),
            "train": not_applicable_range("shadow preparation does not fit a model"),
            "valid": not_applicable_range("shadow preparation does not select a checkpoint"),
            "signal": declared_range(FORWARD_START_DATE.date(), FORWARD_END_DATE.date()),
            "backtest": declared_range(FORWARD_START_DATE.date(), FORWARD_END_DATE.date()),
        },
        "max_data_date": str(FORWARD_END_DATE.date()),
        "split_roles": [
            {
                "split": "forward_2026",
                "selection_eligible": False,
                "forward_used": True,
            }
        ],
        "transform": {
            "state_sha256": sha256_file(alpha_path),
            "fit_range": not_applicable_range("owned by frozen parent alpha"),
        },
        "lineage": {
            "parent_fit_end": str(RESEARCH_END_DATE.date()),
            "parent_selection_end": str(RESEARCH_END_DATE.date()),
        },
    }
    create_experiment(
        output_dir,
        experiment_id=args.experiment_id,
        config=config,
        protocol=protocol,
        cache_contract=cache_contract,
        project_root=project_root,
        parent_experiment_ids=(
            "oof_baseline_components_20260715",
            "state_aware_portfolio_pilot_20260715",
        ),
        formal=True,
        experiment_scope=scope,
    )

    rollback_path = output_dir / "rollback_policy.json"
    _write_exclusive(rollback_path, build_rollback_policy())
    activation_log_path = output_dir / "shadow_activation_log.jsonl"
    _write_exclusive(
        activation_log_path,
        json.dumps(
            {
                "event_type": "shadow_prepared",
                "status": "not_active",
                "reason": "no Phase 5 proposal has formal promotion status",
                "formal_fallback": "ledger_path_v3_t0001_nolookahead",
            },
            ensure_ascii=False,
        )
        + "\n",
    )

    for relative in REQUIRED_ARTIFACTS:
        record_artifact(
            output_dir,
            name=relative,
            path=project_root / relative,
            kind="phase6_parent_artifact",
        )
    record_artifact(output_dir, name="rollback_policy", path=rollback_path, kind="rollback_policy")
    record_artifact(output_dir, name="shadow_activation_log", path=activation_log_path, kind="activation_log")
    append_event(
        output_dir,
        status="completed",
        event_type="shadow_manifest_frozen",
        details={
            "activation_allowed": False,
            "forward_selection_allowed": False,
            "reason": "research-only Phase 6 preparation",
        },
    )
    index_path = finalize_artifact_index(output_dir)
    print(json.dumps({"manifest": str(output_dir / "experiment_manifest.json"), "artifact_index": str(index_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
