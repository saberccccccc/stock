"""Complete missing realistic attribution cells for cond_pairrisk_volg001.

This is deliberately scoped to the registered formal baseline and challenger.
It fills only the missing 2024/2025 cost2x and capacity_3pct cells, then
rebuilds the aggregate CSV consumed by the registry attribution audit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_SIGNAL = ROOT / "reports/state_aware_policy_applied_20260704/multi_downside_e19_sa_p05_ledger_path_v3_t0001_nolookahead"
CANDIDATE_SIGNAL = ROOT / "reports/state_aware_policy_applied_20260704/multi_downside_e19_sa_p05_ledger_path_v3_nolookahead_cond_pairrisk_volg001"
LEDGER_ROOT = ROOT / "reports/state_aware_pairwise_ledger_path_dataset_20260704/multi_downside_e19_sa_p05"
BASE_NORMAL = ROOT / "reports/state_aware_policy_training_20260704/ledger_path_v3_t0001_nolookahead_open_ledger"
BASE_STRESS = ROOT / "reports/state_aware_policy_training_20260704/stress_ledger_path_v3_t0001_nolookahead"
CANDIDATE_BACKTEST = ROOT / "reports/sa_20260710_backtest/cond_pairrisk_volg001_realistic"
OUTPUT_ROOT = ROOT / "reports/cond_pairrisk_volg001_attribution_20260710"
ATTRIBUTION_SCRIPT = ROOT / "run/summarize_ledger_path_v3_attribution.py"


def path_for(base_root, split, stress, capital, stem):
    return base_root / split / stress / f"{stem}_pv{capital}_target006_hold100.csv"


def run_cell(split, stress, capital):
    output = OUTPUT_ROOT / split / stress / capital
    summary = output / "attribution_summary.json"
    if summary.exists():
        return False
    baseline_root = BASE_NORMAL if stress == "normal" else BASE_STRESS
    command = [
        sys.executable,
        str(ATTRIBUTION_SCRIPT),
        "--audit-csv", str(CANDIDATE_SIGNAL / split / "alpha_policy.pairwise_audit.csv"),
        "--ledger-dataset", str(LEDGER_ROOT / split / "pairwise_ledger_path_dataset.parquet"),
        "--baseline-diagnostics", str(path_for(baseline_root, split, stress, capital, "diagnostics")),
        "--candidate-diagnostics", str(path_for(CANDIDATE_BACKTEST, split, stress, capital, "diagnostics")),
        "--baseline-returns", str(path_for(baseline_root, split, stress, capital, "returns")),
        "--candidate-returns", str(path_for(CANDIDATE_BACKTEST, split, stress, capital, "returns")),
        "--output-dir", str(output),
        "--split-name", f"{split}_{stress}_{capital}",
    ]
    missing = [Path(command[index + 1]) for index, token in enumerate(command) if token.startswith("--") and index + 1 < len(command) and command[index + 1].endswith((".csv", ".parquet")) and not Path(command[index + 1]).exists()]
    if missing:
        raise FileNotFoundError(missing)
    subprocess.run(command, check=True)
    return True


def rebuild_aggregate():
    rows = []
    for summary_path in OUTPUT_ROOT.glob("*/*/*/attribution_summary.json"):
        split, scenario, capital = summary_path.relative_to(OUTPUT_ROOT).parts[:3]
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {"split": split, "scenario": scenario, "capital": capital}
        row.update({f"ret_{key}": value for key, value in payload["return_delta_summary"].items()})
        row.update({f"repl_{key}": value for key, value in payload["replacement_summary"].items()})
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["split", "scenario", "capital"])
    frame.to_csv(OUTPUT_ROOT / "attribution_aggregate_summary.csv", index=False, encoding="utf-8-sig")
    return len(frame)


def main():
    completed = 0
    for split in ("val_2024", "test_2025"):
        for stress in ("cost2x", "capacity_3pct"):
            for capital in ("0050w", "0100w"):
                completed += int(run_cell(split, stress, capital))
    rows = rebuild_aggregate()
    print({"new_cells": completed, "aggregate_rows": rows}, flush=True)


if __name__ == "__main__":
    main()
