import json
from pathlib import Path

import pytest

from backtest.market_data_contract import ExecutionMarketDataContract
from data.market_data_access_audit import audit_market_data_call_sites


ROOT = Path(__file__).resolve().parents[1]


def test_execution_market_data_contract_is_explicit_and_legacy_by_default():
    contract = ExecutionMarketDataContract()

    assert contract.backend == "legacy"
    assert contract.cli_args()[1] == "legacy"
    assert contract.manifest()["data_role"] == "legacy_compatibility"


def test_execution_market_data_contract_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown OHLC backend"):
        ExecutionMarketDataContract(backend="magic")


def test_market_data_call_site_policy_passes_for_repository():
    result = audit_market_data_call_sites(ROOT)

    assert result["status"] == "passed"
    assert result["formal_default_backend"] == "legacy"
    assert result["default_switch_stage"] == "MD9"
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
