"""Audit the completed Compact monthly OOF experiment before P4 strong rolling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import load_alpha_rows
from experiments.recording import load_events, sha256_file, validate_manifest_for_formal_use


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        default="reports/experiments/workflow_monthly_rolling_compact_4y6m1m_v1_20260716",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main(argv=None):
    args = parse_args(argv)
    experiment = _resolve(args.experiment_dir)
    output = _resolve(args.output)
    model_signal = experiment / "model_signal"
    formal = validate_manifest_for_formal_use(model_signal / "experiment_manifest.json")
    events = load_events(experiment)
    if not events or events[-1]["status"] != "completed":
        raise ValueError("workflow experiment has no terminal completed event")
    rolling = json.loads((model_signal / "rolling_manifest.json").read_text(encoding="utf-8-sig"))
    windows = rolling["windows"]
    if len(windows) != 24:
        raise ValueError(f"expected 24 Compact windows, found {len(windows)}")
    window_rows = 0
    owners = {}
    for window in windows:
        alpha = Path(window["alpha_path"])
        model = Path(window["model_path"])
        if sha256_file(alpha) != window["alpha_sha256"]:
            raise ValueError(f"window alpha hash mismatch: {window['name']}")
        if sha256_file(model) != window["model_sha256"]:
            raise ValueError(f"window model hash mismatch: {window['name']}")
        for row in load_alpha_rows(alpha):
            date = str(row["date"])
            if date in owners:
                raise ValueError(f"duplicate OOS owner for {date}")
            owners[date] = window["name"]
            window_rows += 1
    split_counts = {}
    for split, descriptor in rolling["split_alpha_paths"].items():
        rows = load_alpha_rows(descriptor["path"])
        split_counts[split] = len(rows)
        if len(rows) != int(descriptor["rows"]):
            raise ValueError(f"stitched split row count mismatch: {split}")
    if split_counts != {"val_2024": 242, "test_2025": 243}:
        raise ValueError(f"unexpected stitched split counts: {split_counts}")
    reports = pd.read_csv(experiment / "workflow_reports.csv")
    candidate = "monthly_rolling_compact_4y6m1m_v1"
    cells = reports.loc[reports["candidate_id"].astype(str) == candidate]
    expected_stress = {"normal", "lag1", "cost2x", "capacity_3pct"}
    if len(cells) != 16 or set(cells["stress"].astype(str)) != expected_stress:
        raise ValueError("Compact ledger does not contain the required 16 cells")
    decisions = pd.read_csv(experiment / "scorecard" / "registry_decisions.csv")
    decision = decisions.loc[decisions["candidate"].astype(str) == candidate].iloc[0].to_dict()
    provenance_exists = (experiment / "provenance" / "provenance_bundle.json").is_file()
    records_exists = (experiment / "records" / "bundle_manifest.json").is_file()
    result = {
        "schema": "p4_compact_oof_pre_audit_v1",
        "experiment": str(experiment),
        "formal_model_signal_manifest": formal,
        "workflow_terminal_status": events[-1]["status"],
        "windows": len(windows),
        "unique_oos_dates": len(owners),
        "window_alpha_rows": window_rows,
        "split_counts": split_counts,
        "compact_ledger_cells": len(cells),
        "decision": str(decision["decision"]),
        "mean_sharpe_delta": float(decision["mean_sharpe_delta"]),
        "mean_ann_delta": float(decision["mean_ann_delta"]),
        "p3_provenance_complete": provenance_exists,
        "standard_records_complete": records_exists,
        "rerun_decision": "do_not_rerun_rejected_compact_for_metadata_only",
        "eligible_for_promotion": False,
        "eligible_as_historical_engineering_evidence": True,
        "reason": (
            "OOF ownership, hashes, and ledger are complete, but the candidate is materially rejected; "
            "P3 runtime provenance cannot be honestly reconstructed after the fact."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
