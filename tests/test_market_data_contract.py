import json
from pathlib import Path

import pytest

import backtest.market_data_contract as market_contract
from backtest.market_data_contract import ExecutionMarketDataContract
from data.market_data_access_audit import audit_market_data_call_sites


ROOT = Path(__file__).resolve().parents[1]


def test_execution_market_data_contract_supports_explicit_legacy_rollback():
    contract = ExecutionMarketDataContract(backend="legacy")

    assert contract.backend == "legacy"
    assert contract.cli_args()[1] == "legacy"
    assert contract.manifest()["data_role"] == "legacy_compatibility"


def test_execution_market_data_contract_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown OHLC backend"):
        ExecutionMarketDataContract(backend="magic")


def test_execution_market_data_contract_builds_monthly_csv_dual_read():
    contract = ExecutionMarketDataContract(
        backend="monthly",
        shadow_backend="csv",
        shadow_report="reports/dual_read.json",
    )

    args = contract.cli_args()
    assert args[args.index("--ohlc-shadow-backend") + 1] == "csv"
    assert args[args.index("--ohlc-shadow-report") + 1] == "reports/dual_read.json"
    assert contract.manifest()["shadow_backend"] == "csv"


def test_execution_market_data_contract_rejects_legacy_dual_read():
    with pytest.raises(ValueError, match="one csv and one monthly"):
        ExecutionMarketDataContract(
            backend="legacy",
            shadow_backend="csv",
            shadow_report="dual.json",
        )


def test_configured_default_backend_reads_versioned_policy(tmp_path, monkeypatch):
    policy = {
        "schema": "execution_market_backend_policy_v2",
        "state": "monthly_active",
        "active_backend": "monthly",
        "candidate_backend": "monthly",
        "candidate_store_root": "data/custom_store",
        "candidate_monthly_cache_root": "cache/custom_monthly",
        "rollback_backend": "legacy",
        "shadow_backend": "csv",
        "required_dual_read_splits": ["val_2024", "test_2025", "forward_2026"],
        "evidence": {},
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setattr(market_contract, "BACKEND_POLICY_PATH", path)

    assert market_contract.configured_default_backend() == "monthly"
    contract = ExecutionMarketDataContract()
    assert contract.backend == "monthly"
    assert contract.manifest()["data_role"] == "monthly_authoritative"
    assert contract.market_daily_store_root == "data/custom_store"
    assert contract.monthly_cache_root == "cache/custom_monthly"


def test_market_data_call_site_policy_passes_for_repository():
    result = audit_market_data_call_sites(ROOT)

    assert result["status"] == "passed"
    assert result["formal_default_backend"] == "monthly"
    assert result["default_switch_stage"] == "MD9_completed"
    assert result["unregistered_legacy_imports"] == []
    assert all(row["status"] == "ok" for row in result["formal_contract_entrypoints"])
    assert all(row["status"] == "ok" for row in result["artifact_only_consumers"])


def test_market_data_call_site_policy_rejects_new_legacy_import(tmp_path):
    for directory in ("run", "backtest", "data", "experiments", "configs"):
        (tmp_path / directory).mkdir()
    policy = json.loads(
        (ROOT / "configs" / "market_data_call_sites.json").read_text(encoding="utf-8")
    )
    policy["formal_contract_entrypoints"] = []
    policy["artifact_only_consumers"] = []
    (tmp_path / "configs" / "market_data_call_sites.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    (tmp_path / "run" / "new_reader.py").write_text(
        "from backtest.open_ledger import load_ohlc_money\n", encoding="utf-8"
    )

    result = audit_market_data_call_sites(tmp_path)

    assert result["status"] == "failed"
    assert result["unregistered_legacy_imports"][0]["path"] == "run/new_reader.py"
