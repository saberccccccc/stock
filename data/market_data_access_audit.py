"""Static governance audit for execution-market data call sites."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping


WATCHED_IMPORTS = {
    "backtest.ohlc_matrix_cache": {
        "build_ohlc_matrix_cache",
        "discover_stock_csvs",
        "ensure_ohlc_matrix_cache",
        "load_ohlc_matrix_meta",
        "load_ohlc_money_from_matrix_cache",
        "load_ohlcv_fields_from_matrix_cache",
        "matrix_cache_is_current",
        "source_signature",
    },
    "backtest.open_ledger": {
        "load_ohlc_money",
    },
}
CONTRACT_MARKERS = (
    "ExecutionMarketDataContract",
    "add_execution_market_data_args",
    "contract_from_args",
)


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _legacy_imports(root: Path) -> list[dict[str, Any]]:
    rows = []
    for directory in ("run", "backtest", "data", "experiments"):
        for path in sorted((root / directory).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module not in WATCHED_IMPORTS:
                    continue
                for imported in node.names:
                    if imported.name in WATCHED_IMPORTS[node.module]:
                        rows.append(
                            {
                                "path": _relative(root, path),
                                "line": int(node.lineno),
                                "module": node.module,
                                "symbol": imported.name,
                            }
                        )
    return rows


def audit_market_data_call_sites(
    project_root: str | Path,
    *,
    policy_path: str | Path = "configs/market_data_call_sites.json",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    policy_file = Path(policy_path)
    if not policy_file.is_absolute():
        policy_file = root / policy_file
    policy: Mapping[str, Any] = json.loads(policy_file.read_text(encoding="utf-8-sig"))
    allowlist = dict(policy["legacy_internal_import_allowlist"])
    imports = _legacy_imports(root)
    unregistered = [row for row in imports if row["path"] not in allowlist]

    contract_rows = []
    missing_contract = []
    for relative in policy["formal_contract_entrypoints"]:
        path = root / relative
        source = path.read_text(encoding="utf-8-sig")
        markers = [marker for marker in CONTRACT_MARKERS if marker in source]
        row = {"path": relative, "markers": markers, "status": "ok" if markers else "missing"}
        contract_rows.append(row)
        if not markers:
            missing_contract.append(relative)

    artifact_rows = []
    for relative in policy["artifact_only_consumers"]:
        path = root / relative
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in WATCHED_IMPORTS:
                forbidden.extend(
                    item.name for item in node.names if item.name in WATCHED_IMPORTS[node.module]
                )
        artifact_rows.append(
            {
                "path": relative,
                "legacy_market_imports": sorted(forbidden),
                "status": "ok" if not forbidden else "failed",
            }
        )

    registered_paths = set(allowlist)
    observed_paths = {row["path"] for row in imports}
    stale_allowlist = sorted(registered_paths - observed_paths)
    status = (
        "passed"
        if not unregistered
        and not missing_contract
        and all(row["status"] == "ok" for row in artifact_rows)
        else "failed"
    )
    return {
        "schema": "market_data_call_site_audit_v1",
        "status": status,
        "policy_path": _relative(root, policy_file),
        "formal_default_backend": policy["policy"]["formal_default_backend"],
        "default_switch_stage": policy["policy"]["default_switch_stage"],
        "formal_contract_entrypoints": contract_rows,
        "artifact_only_consumers": artifact_rows,
        "legacy_internal_imports": imports,
        "unregistered_legacy_imports": unregistered,
        "stale_allowlist_paths": stale_allowlist,
    }
