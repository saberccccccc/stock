import json
import hashlib

import pandas as pd

from data.nt6_market_backend_closure import inspect_nt6_market_backend_closure
from data.execution_market_backend_policy import promote_policy, rollback_policy
from data.market_daily_store import MarketDailyStore


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _policy():
    return {
        "schema": "execution_market_backend_policy_v2",
        "state": "legacy_active",
        "active_backend": "legacy",
        "candidate_backend": "monthly",
        "candidate_store_root": "candidate_store",
        "candidate_monthly_cache_root": "candidate_cache",
        "rollback_backend": "legacy",
        "shadow_backend": "csv",
        "required_dual_read_splits": [
            "val_2024",
            "test_2025",
            "forward_2026",
        ],
        "evidence": {
            "behavior_parity": "parity.json",
            "performance_acceptance": "acceptance.json",
            "call_site_audit": "calls.json",
            "dual_read_reports": {
                split: f"dual.{split}.json"
                for split in ("val_2024", "test_2025", "forward_2026")
            },
        },
    }


def _write_complete_evidence(root):
    store_root = root / "candidate_store"
    cache_root = root / "candidate_cache"
    cache_root.mkdir()
    store = MarketDailyStore(store_root)
    store.commit_partition(
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-07-01",
                    "code": "000001.SZ",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.2,
                    "volume": 1000.0,
                    "money": 10000.0,
                    "factor": 1.0,
                }
            ]
        ),
        instrument_type="equity",
        source="test",
    )
    active = store.active_state()
    _write(
        root / "incremental.json",
        {
            "schema": "market_daily_incremental_benchmark_v1",
            "status": "passed",
            "source": {
                "store_root": str(store_root.resolve()),
                "manifest_sha256": active["manifest_sha256"],
            },
        },
    )
    _write(
        root / "matrix" / "matrix_status.json",
        {"schema": "open_ledger_backend_parity_matrix_v1", "status": "passed"},
    )
    performance_sources = {
        "matrix_status": root / "matrix" / "matrix_status.json",
        "csv_test_2025_performance": (
            root / "matrix" / "csv" / "test_2025" / "performance.json"
        ),
        "monthly_test_2025_performance": (
            root / "matrix" / "monthly" / "test_2025" / "performance.json"
        ),
        "csv_forward_2026_performance": (
            root / "matrix" / "csv" / "forward_2026" / "performance.json"
        ),
        "monthly_forward_2026_performance": (
            root / "matrix" / "monthly" / "forward_2026" / "performance.json"
        ),
        "incremental_benchmark": root / "incremental.json",
    }
    for key, path in performance_sources.items():
        if not path.exists():
            _write(path, {"source": key})
    hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in performance_sources.items()
    }
    gates = {
        key: True
        for key in (
            "parity_passed",
            "runtime_passed",
            "memory_passed",
            "process_io_recorded",
            "incremental_passed",
        )
    }
    _write(
        root / "acceptance.json",
        {
            "schema": "market_data_performance_acceptance_v1",
            "status": "passed",
            "gates": gates,
            "evidence_sha256": hashes,
            "evidence_root": str(root / "matrix"),
            "incremental": {"path": str(root / "incremental.json")},
        },
    )
    _write(
        root / "dual" / "dual_read_matrix_status.json",
        {"schema": "execution_market_dual_read_matrix_v1", "status": "passed"},
    )
    _write(
        root / "parity.json",
        {"schema": "open_ledger_backend_parity_matrix_v1", "status": "passed"},
    )
    _write(
        root / "calls.json",
        {"schema": "market_data_call_site_audit_v1", "status": "passed"},
    )
    for split in ("val_2024", "test_2025", "forward_2026"):
        _write(
            root / f"dual.{split}.json",
            {
                "schema": "execution_market_dual_read_v1",
                "status": "passed",
                "primary_backend": "monthly",
                "shadow_backend": "csv",
                "monthly_identity": {
                    "store_root": str(store_root.resolve()),
                    "cache_root": str(cache_root.resolve()),
                    "months": [{"month": "2026-07"}],
                },
            },
        )
    _write(root / "policy.json", _policy())


def _inspect(root):
    return inspect_nt6_market_backend_closure(
        root,
        incremental_evidence="incremental.json",
        clean_matrix_status="matrix/matrix_status.json",
        clean_acceptance="acceptance.json",
        dual_read_matrix_status="dual/dual_read_matrix_status.json",
        policy_path="policy.json",
    )


def test_closure_reports_first_missing_phase(tmp_path):
    _write(tmp_path / "policy.json", _policy())

    result = _inspect(tmp_path)

    assert result["status"] == "incomplete"
    assert result["next_phase"] == "incremental_benchmark"
    assert result["automatic_promotion"] is False


def test_closure_stops_at_manual_promotion_gate(tmp_path):
    _write_complete_evidence(tmp_path)

    result = _inspect(tmp_path)

    assert result["status"] == "ready_for_manual_promotion"
    assert result["next_phase"] == "manual_promotion"
    assert result["promotion_audit"]["status"] == "passed"
    assert result["recovery_drill"]["status"] == "incomplete"
    assert result["automatic_promotion"] is False


def test_closure_requires_and_recognizes_full_recovery_drill(tmp_path):
    _write_complete_evidence(tmp_path)
    initial = _inspect(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    policy = promote_policy(
        policy,
        initial["promotion_audit"],
        actor="tester",
        reason="initial promotion",
        changed_at="2026-07-31T00:00:00+00:00",
    )
    _write(policy_path, policy)
    promoted = _inspect(tmp_path)
    assert promoted["status"] == "recovery_drill_required"
    assert promoted["next_phase"] == "rollback_recovery_drill"

    policy = rollback_policy(
        policy,
        actor="tester",
        reason="recovery rollback",
        changed_at="2026-07-31T00:01:00+00:00",
    )
    _write(policy_path, policy)
    rolled_back = _inspect(tmp_path)
    policy = promote_policy(
        policy,
        rolled_back["promotion_audit"],
        actor="tester",
        reason="recovery restoration",
        changed_at="2026-07-31T00:02:00+00:00",
    )
    _write(policy_path, policy)

    completed = _inspect(tmp_path)

    assert completed["status"] == "completed"
    assert completed["next_phase"] == "complete"
    assert completed["recovery_drill"] == {
        "status": "passed",
        "passed": True,
        "sequence_matches": True,
        "promotion_identity_matches": True,
        "final_evidence_matches": True,
        "final_state_matches": True,
        "transition_count": 3,
    }
