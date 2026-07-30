"""Deterministic daily Shadow packaging around the project open-ledger."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from alpha.io import load_alpha_rows, resolve_row_scores
from backtest.market_data_contract import ExecutionMarketDataContract
from experiments.recording import canonical_json_hash, sha256_file, utc_now
from experiments.shadow_lifecycle import record_shadow_observation, validate_shadow_lifecycle


RUN_MANIFEST_NAME = "shadow_run_manifest.json"
DAILY_MANIFEST_NAME = "daily_manifest.json"


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": int(source.stat().st_size),
    }


def _iso_date(value: Any) -> str:
    return pd.Timestamp(value).normalize().strftime("%Y-%m-%d")


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _filter_date(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    if "date" not in frame.columns or frame.empty:
        return frame.iloc[0:0].copy()
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return frame.loc[dates == date].copy()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _score_stats(scores: list[float]) -> dict[str, float]:
    series = pd.Series([value for value in scores if math.isfinite(value)], dtype="float64")
    if series.empty:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _drift(row: Mapping[str, Any], previous: Mapping[str, Any] | None) -> dict[str, Any]:
    codes = list(row.get("codes", []))
    scores = [_finite(value) for value in resolve_row_scores(row)]
    current_top = codes[:30]
    previous_top = list(previous.get("codes", []))[:30] if previous else []
    overlap = len(set(current_top) & set(previous_top))
    union = len(set(current_top) | set(previous_top))
    return {
        "top30_overlap_count": overlap,
        "top30_overlap_fraction": overlap / max(len(current_top), 1) if previous else None,
        "top30_jaccard": overlap / union if previous and union else None,
        "universe_size": len(codes),
        "score": _score_stats(scores),
    }


def build_sweep_command(
    *,
    project_root: str | Path,
    alpha_path: str | Path,
    candidate_id: str,
    execution_dir: str | Path,
    data_dir: str | Path,
    start_date: str,
    end_date: str,
    max_data_date: str,
    portfolio_value: float,
    market_data: ExecutionMarketDataContract | None = None,
    python_executable: str | Path | None = None,
) -> list[str]:
    root = Path(project_root).resolve()
    python = str(python_executable or sys.executable)
    contract = market_data or ExecutionMarketDataContract()
    command = [
        python,
        str(root / "run" / "sweep_open_price_ledger_params.py"),
        "--alpha-specs",
        f"{candidate_id}={Path(alpha_path).resolve()}",
        "--output-dir",
        str(Path(execution_dir).resolve()),
        "--data-dir",
        str(Path(data_dir).resolve()),
        "--target-fracs",
        "0.006",
        "--hold-fracs",
        "0.10",
        "--rebalance-bands",
        "0.20",
        "--stresses",
        "normal",
        "--portfolio-values",
        str(float(portfolio_value)),
        "--max-new-names-list",
        "5",
        "--exit-hold-fracs",
        "0",
        "--switch-gap-fracs",
        "0",
        "--execution-constraint-mode",
        "realistic",
        "--start-date",
        _iso_date(start_date),
        "--end-date",
        _iso_date(end_date),
        "--max-data-date",
        _iso_date(max_data_date),
        "--save-path-details",
    ]
    command.extend(contract.cli_args())
    return command


def _load_ledger_artifacts(execution_dir: Path, candidate_id: str, portfolio_value: float):
    index_path = execution_dir / "path_artifact_index.csv"
    index = pd.read_csv(index_path)
    capital = pd.to_numeric(index["portfolio_value"], errors="coerce")
    matches = index.loc[
        (index["alpha_name"].astype(str) == candidate_id)
        & (index["stress"].astype(str) == "normal")
        & ((capital - float(portfolio_value)).abs() < 0.01)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one normal ledger path, found {len(matches)}")
    descriptor = matches.iloc[0].to_dict()
    frames = {
        name: _read_csv(descriptor[name])
        for name in ("equity_curve", "diagnostics", "positions", "orders", "rejections", "costs")
    }
    return index_path, descriptor, frames


def _execution_date_map(alpha_rows, diagnostics: pd.DataFrame) -> dict[str, str]:
    execution_dates = sorted({_iso_date(value) for value in diagnostics["date"].dropna()})
    result = {}
    used = set()
    for row in alpha_rows:
        signal_date = _iso_date(row["date"])
        candidates = [date for date in execution_dates if date > signal_date and date not in used]
        if not candidates:
            raise ValueError(f"no next-session ledger execution for signal_date={signal_date}")
        result[signal_date] = candidates[0]
        used.add(candidates[0])
    return result


def materialize_daily_packets(
    *,
    run_dir: str | Path,
    lifecycle_snapshot: Mapping[str, Any],
    alpha_path: str | Path,
    candidate_id: str,
    start_date: str,
    end_date: str,
    portfolio_value: float,
    execution_dir: str | Path,
) -> tuple[list[dict[str, Any]], str]:
    target = Path(run_dir).resolve()
    rows = [
        row
        for row in load_alpha_rows(alpha_path)
        if _iso_date(start_date) <= _iso_date(row["date"]) <= _iso_date(end_date)
    ]
    if not rows:
        raise ValueError("daily Shadow interval contains no alpha rows")
    _index_path, ledger_descriptor, frames = _load_ledger_artifacts(
        Path(execution_dir).resolve(), candidate_id, portfolio_value
    )
    execution_dates = _execution_date_map(rows, frames["diagnostics"])
    source_descriptor = _artifact(alpha_path)
    daily = []
    previous = None
    for row in rows:
        signal_date = _iso_date(row["date"])
        execution_date = execution_dates[signal_date]
        daily_dir = target / "daily" / signal_date
        daily_dir.mkdir(parents=True, exist_ok=True)
        codes = list(row.get("codes", []))
        scores = [_finite(value) for value in resolve_row_scores(row)]
        proposal = {
            "schema": "shadow_proposal_v1",
            "candidate_id": candidate_id,
            "signal_date": signal_date,
            "execution_date": execution_date,
            "selection_eligible": False,
            "source_alpha": source_descriptor,
            "ranked_candidates": [
                {"rank": rank, "code": code, "score": score}
                for rank, (code, score) in enumerate(zip(codes[:100], scores[:100]), start=1)
            ],
        }
        proposal_path = _write_json(daily_dir / "proposal.json", proposal)
        artifact_paths = {"proposal": proposal_path}
        daily_frames = {}
        for name, frame in frames.items():
            selected = _filter_date(frame, execution_date)
            output = daily_dir / f"{name}.csv"
            selected.to_csv(output, index=False)
            artifact_paths[name] = output
            daily_frames[name] = selected
        failures = []
        if daily_frames["diagnostics"].empty:
            failures.append("missing_execution_diagnostics")
        if daily_frames["equity_curve"].empty:
            failures.append("missing_equity_mark")
        if not codes:
            failures.append("empty_candidate_universe")
        diag = daily_frames["diagnostics"].iloc[-1].to_dict() if not daily_frames["diagnostics"].empty else {}
        equity = daily_frames["equity_curve"].iloc[-1].to_dict() if not daily_frames["equity_curve"].empty else {}
        drift = _drift(row, previous)
        drift.update(
            {
                "holdings": int(_finite(diag.get("holdings"))),
                "gross_weight": _finite(diag.get("gross_weight")),
                "turnover": _finite(diag.get("turnover")),
                "daily_return": _finite(equity.get("return")),
                "active_return": _finite(equity.get("active_return")),
                "equity_cny": _finite(equity.get("equity_cny")),
                "orders": int(len(daily_frames["orders"])),
                "rejections": int(len(daily_frames["rejections"])),
                "total_cost_cny": float(
                    pd.to_numeric(daily_frames["costs"].get("total_cost_cny", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
                ),
            }
        )
        drift_path = _write_json(daily_dir / "drift.json", drift)
        artifact_paths["drift"] = drift_path
        artifacts = {name: _artifact(path) for name, path in artifact_paths.items()}
        packet = {
            "schema": "daily_shadow_packet_v1",
            "lifecycle_id": lifecycle_snapshot["manifest"]["lifecycle_id"],
            "candidate_id": candidate_id,
            "signal_date": signal_date,
            "execution_date": execution_date,
            "portfolio_value": float(portfolio_value),
            "execution_contract": {
                "price": "next_session_open",
                "constraint_mode": "realistic",
                "selection_policy": "retention",
                "target_frac": 0.006,
                "hold_frac": 0.10,
                "rebalance_band": 0.20,
                "max_new_names": 5,
            },
            "checks": {
                "data_ready": not failures,
                "silent_failures": failures,
                "alpha_source_hash_verified": True,
                "lifecycle_artifacts_verified": True,
            },
            "artifacts": artifacts,
        }
        packet["packet_sha256"] = canonical_json_hash(packet)
        manifest_path = _write_json(daily_dir / DAILY_MANIFEST_NAME, packet)
        semantic_packet_sha256 = canonical_json_hash(
            {
                "candidate_id": candidate_id,
                "signal_date": signal_date,
                "execution_date": execution_date,
                "portfolio_value": float(portfolio_value),
                "execution_contract": packet["execution_contract"],
                "checks": packet["checks"],
                "artifact_sha256": {
                    name: descriptor["sha256"] for name, descriptor in artifacts.items()
                },
            }
        )
        daily.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "manifest": _artifact(manifest_path),
                "packet_sha256": packet["packet_sha256"],
                "semantic_packet_sha256": semantic_packet_sha256,
            }
        )
        previous = row
    if any(not json.loads(Path(item["manifest"]["path"]).read_text(encoding="utf-8"))["checks"]["data_ready"] for item in daily):
        raise RuntimeError("daily Shadow run contains silent failures")
    semantic = canonical_json_hash(
        {
            "candidate_id": candidate_id,
            "portfolio_value": float(portfolio_value),
            "alpha_sha256": source_descriptor["sha256"],
            "ledger_sweep_key": str(ledger_descriptor["sweep_key_sha256"]),
            "daily_packets": [item["semantic_packet_sha256"] for item in daily],
        }
    )
    return daily, semantic


def run_daily_shadow(
    *,
    project_root: str | Path,
    lifecycle_dir: str | Path,
    run_dir: str | Path,
    alpha_path: str | Path,
    candidate_id: str,
    data_dir: str | Path,
    start_date: str,
    end_date: str,
    max_data_date: str,
    portfolio_value: float,
    market_data: ExecutionMarketDataContract | None = None,
    mode: str = "historical_replay",
    actor: str = "",
    reason: str = "",
    python_executable: str | Path | None = None,
    execute: bool = True,
) -> Path:
    root = Path(project_root).resolve()
    target = Path(run_dir).resolve()
    if target.exists() and any(target.iterdir()) and execute:
        raise FileExistsError(f"daily Shadow run directory is not empty: {target}")
    snapshot = validate_shadow_lifecycle(lifecycle_dir)
    if snapshot["manifest"]["candidate_id"] != candidate_id:
        raise ValueError("daily Shadow candidate does not match lifecycle candidate")
    if mode not in {"historical_replay", "shadow_observation"}:
        raise ValueError(f"unsupported daily Shadow mode: {mode}")
    if mode == "shadow_observation" and snapshot["state"]["state"] != "shadow":
        raise ValueError("formal daily observations require lifecycle state=shadow")
    if mode == "shadow_observation" and (not actor.strip() or not reason.strip()):
        raise ValueError("formal daily observations require actor and reason")
    target.mkdir(parents=True, exist_ok=True)
    execution_dir = target / "execution"
    market_data = market_data or ExecutionMarketDataContract()
    command = build_sweep_command(
        project_root=root,
        alpha_path=alpha_path,
        candidate_id=candidate_id,
        execution_dir=execution_dir,
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        max_data_date=max_data_date,
        portfolio_value=portfolio_value,
        market_data=market_data,
        python_executable=python_executable,
    )
    if execute:
        result = subprocess.run(command, cwd=root)
        if result.returncode:
            raise RuntimeError(f"daily Shadow ledger failed with returncode={result.returncode}")
    daily, semantic = materialize_daily_packets(
        run_dir=target,
        lifecycle_snapshot=snapshot,
        alpha_path=alpha_path,
        candidate_id=candidate_id,
        start_date=start_date,
        end_date=end_date,
        portfolio_value=portfolio_value,
        execution_dir=execution_dir,
    )
    manifest = {
        "schema": "daily_shadow_run_v1",
        "created_at": utc_now(),
        "mode": mode,
        "lifecycle_dir": str(Path(lifecycle_dir).resolve()),
        "lifecycle_state_at_run": snapshot["state"]["state"],
        "candidate_id": candidate_id,
        "selection_eligible": False,
        "forward_used_for_selection": False,
        "start_date": _iso_date(start_date),
        "end_date": _iso_date(end_date),
        "max_data_date": _iso_date(max_data_date),
        "portfolio_value": float(portfolio_value),
        "alpha_source": _artifact(alpha_path),
        "data_dir": str(Path(data_dir).resolve()),
        "market_data": market_data.manifest(project_root=root),
        "command": command,
        "daily_count": len(daily),
        "daily": daily,
        "semantic_sha256": semantic,
        "recorded_to_lifecycle": mode == "shadow_observation",
    }
    manifest_path = _write_json(target / RUN_MANIFEST_NAME, manifest)
    if mode == "shadow_observation":
        for item in daily:
            daily_manifest = Path(item["manifest"]["path"])
            record_shadow_observation(
                lifecycle_dir,
                observation_date=item["signal_date"],
                artifacts={"daily_manifest": daily_manifest, "run_manifest": manifest_path},
                actor=actor,
                reason=reason,
            )
    return manifest_path


def validate_daily_shadow_run(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = json.loads((target / RUN_MANIFEST_NAME).read_text(encoding="utf-8-sig"))
    packets = []
    for item in manifest["daily"]:
        descriptor = item["manifest"]
        path = Path(descriptor["path"])
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"daily manifest changed: {path}")
        packet = json.loads(path.read_text(encoding="utf-8-sig"))
        expected = packet.pop("packet_sha256")
        if canonical_json_hash(packet) != expected:
            raise ValueError(f"daily packet hash is invalid: {path}")
        for artifact in packet["artifacts"].values():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"daily artifact changed: {artifact['path']}")
        packets.append(expected)
    if len(packets) != int(manifest["daily_count"]):
        raise ValueError("daily Shadow count is inconsistent")
    validate_shadow_lifecycle(manifest["lifecycle_dir"])
    return manifest


def replay_daily_shadow_run(
    source_run_dir: str | Path,
    replay_run_dir: str | Path,
    *,
    project_root: str | Path,
    python_executable: str | Path | None = None,
) -> Path:
    """Re-execute a historical package and require identical semantic output."""

    source = validate_daily_shadow_run(source_run_dir)
    replay_manifest_path = run_daily_shadow(
        project_root=project_root,
        lifecycle_dir=source["lifecycle_dir"],
        run_dir=replay_run_dir,
        alpha_path=source["alpha_source"]["path"],
        candidate_id=source["candidate_id"],
        data_dir=source["data_dir"],
        start_date=source["start_date"],
        end_date=source["end_date"],
        max_data_date=source["max_data_date"],
        portfolio_value=float(source["portfolio_value"]),
        market_data=ExecutionMarketDataContract(
            backend=source.get("market_data", {}).get("backend", "legacy"),
            market_daily_store_root=source.get("market_data", {}).get(
                "market_daily_store_root", "data/market_daily_candidate_v2"
            )
            or "data/market_daily_candidate_v2",
            monthly_cache_root=source.get("market_data", {}).get(
                "monthly_cache_root", "cache/ohlcv_monthly_v3_candidate"
            )
            or "cache/ohlcv_monthly_v3_candidate",
        ),
        mode="historical_replay",
        python_executable=python_executable,
    )
    replay = validate_daily_shadow_run(replay_run_dir)
    comparison = {
        "schema": "daily_shadow_replay_comparison_v1",
        "source_run": str(Path(source_run_dir).resolve()),
        "replay_run": str(Path(replay_run_dir).resolve()),
        "source_semantic_sha256": source["semantic_sha256"],
        "replay_semantic_sha256": replay["semantic_sha256"],
        "deterministic_match": source["semantic_sha256"] == replay["semantic_sha256"],
    }
    comparison_path = _write_json(Path(replay_run_dir).resolve() / "replay_comparison.json", comparison)
    if not comparison["deterministic_match"]:
        raise RuntimeError(f"daily Shadow deterministic replay mismatch: {comparison_path}")
    return replay_manifest_path
