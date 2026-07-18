"""Run a predeclared, resumable LightGBM tuning search on frozen research data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.recording import canonical_json_hash
from experiments.tuning import (
    append_trial_record,
    build_trial_config,
    latest_trial_records,
    load_tuning_spec,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_command(command, *, cwd, log_path):
    with Path(log_path).open("w", encoding="utf-8") as log:
        return subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)


def ledger_metrics(path: Path, split: str) -> dict[str, float | int]:
    frame = pd.read_csv(path)
    frame = frame[
        (frame["target_frac"].round(6) == 0.006)
        & (frame["hold_frac"].round(6) == 0.10)
        & (frame["rebalance_band"].round(6) == 0.20)
        & (frame["max_new_names"] == 5)
    ].copy()
    if len(frame) != 8:
        raise ValueError(f"{split} ledger must have 8 fixed-contract rows, got {len(frame)}")
    normal_1m = frame[(frame["stress"] == "normal") & (frame["portfolio_value"] == 1000000)].iloc[0]
    return {
        f"{split}_normal_1m_ann": float(normal_1m["ann"]),
        f"{split}_normal_1m_sharpe": float(normal_1m["sharpe"]),
        f"{split}_normal_1m_mdd": float(normal_1m["mdd"]),
        f"{split}_min_sharpe": float(frame["sharpe"].min()),
        f"{split}_mean_sharpe": float(frame["sharpe"].mean()),
        f"{split}_max_mdd": float(frame["mdd"].max()),
        f"{split}_mean_turnover": float(frame["avg_turnover"].mean()),
        f"{split}_mean_cost": float(frame["total_cost"].mean()),
    }


def main(argv=None):
    args = parse_args(argv)
    spec_path = Path(args.spec).resolve()
    spec = load_tuning_spec(spec_path)
    base_config_path = (ROOT / spec["base_config"]).resolve()
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trial_log = output_root / "trials.jsonl"
    latest = latest_trial_records(trial_log)
    search_manifest_path = output_root / "search_manifest.json"
    search_manifest = {
        "schema_version": 1,
        "spec_path": str(spec_path),
        "spec_sha256": canonical_json_hash(spec),
        "base_config_path": str(base_config_path),
        "base_config_sha256": canonical_json_hash(base_config),
        "spec": spec,
        "created_at": now(),
    }
    if not search_manifest_path.exists():
        search_manifest_path.write_text(json.dumps(search_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif not args.resume:
        raise FileExistsError(f"tuning output exists; use --resume: {output_root}")

    for trial in spec["trials"]:
        trial_id = trial["trial_id"]
        if args.resume and latest.get(trial_id, {}).get("status") == "completed":
            continue
        trial_dir = output_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        config = build_trial_config(base_config, trial)
        config_path = trial_dir / "trial_config.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"trial config changed after creation: {config_path}")
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        experiment_id = f"compact_tune_{trial_id}_20260715"
        experiment_dir = trial_dir / "experiment"
        append_trial_record(trial_log, {"at": now(), "trial_id": trial_id, "status": "running", "experiment_id": experiment_id, "overrides": trial["overrides"]})
        try:
            rolling_command = [
                args.python, "-u", "run/rolling_lgbm_alpha.py",
                "--config", str(config_path),
                "--output-dir", str(experiment_dir),
                "--experiment-id", experiment_id,
            ]
            if args.dry_run:
                print(json.dumps({"trial_id": trial_id, "rolling_command": rolling_command}, ensure_ascii=False))
                continue
            result = run_command(rolling_command, cwd=ROOT, log_path=trial_dir / "rolling.log")
            if result.returncode != 0:
                raise RuntimeError(f"rolling failed with return code {result.returncode}")
            for split, window in (("val_2024", "predict_2024"), ("test_2025", "predict_2025")):
                alpha_path = experiment_dir / "windows" / window / "alpha_raw.jsonl"
                ledger_dir = experiment_dir / "ledger" / split
                ledger_command = [
                    args.python, "-u", "run/evaluate_experiment_alpha.py",
                    "--experiment-dir", str(experiment_dir),
                    "--experiment-id", experiment_id,
                    "--alpha-path", str(alpha_path),
                    "--split", split,
                    "--output-dir", str(ledger_dir),
                ]
                result = run_command(ledger_command, cwd=ROOT, log_path=trial_dir / f"ledger_{split}.log")
                if result.returncode != 0:
                    raise RuntimeError(f"ledger {split} failed with return code {result.returncode}")
            metrics = {}
            for split in ("val_2024", "test_2025"):
                metrics.update(ledger_metrics(experiment_dir / "ledger" / split / "open_price_ledger_param_sweep_summary.csv", split))
            (trial_dir / "trial_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            append_trial_record(trial_log, {"at": now(), "trial_id": trial_id, "status": "completed", "experiment_id": experiment_id, "metrics": metrics})
        except Exception as exc:
            append_trial_record(trial_log, {"at": now(), "trial_id": trial_id, "status": "failed", "experiment_id": experiment_id, "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()
