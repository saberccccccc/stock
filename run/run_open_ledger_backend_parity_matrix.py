"""Run resumable CSV-vs-monthly open-ledger parity over the fixed 24 cells."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

from core.research_protocol import get_split_spec
from backtest.market_data_contract import configured_candidate_paths
from run.audit_open_ledger_backend_parity import ARTIFACT_COLUMNS, compare_sweep_roots


CANDIDATE_ID = "ledger_path_v3_t0001_nolookahead"
ALPHA_ROOT = (
    ROOT
    / "reports"
    / "state_aware_policy_applied_20260704"
    / "multi_downside_e19_sa_p05_ledger_path_v3_t0001_nolookahead"
)
SPLITS = ("val_2024", "test_2025", "forward_2026")
BACKENDS = ("csv", "monthly")
STRESSES = "normal,lag1,cost2x,capacity_3pct"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def available_memory_gib() -> float:
    return psutil.virtual_memory().available / (1024**3)


def build_sweep_command(args, *, split: str, backend: str, output_dir: Path) -> list[str]:
    spec = get_split_spec(split)
    start, end, max_date = spec.command_dates()
    alpha = (ALPHA_ROOT / split / "alpha_policy.jsonl").resolve()
    if not alpha.is_file():
        raise FileNotFoundError(alpha)
    data_dir = args.forward_data_dir if spec.is_forward else args.research_data_dir
    return [
        sys.executable,
        "run/sweep_open_price_ledger_params.py",
        "--alpha-specs",
        f"{CANDIDATE_ID}={alpha}",
        "--output-dir",
        str(output_dir.resolve()),
        "--data-dir",
        data_dir,
        "--target-fracs",
        "0.006",
        "--hold-fracs",
        "0.10",
        "--rebalance-bands",
        "0.20",
        "--stresses",
        STRESSES,
        "--portfolio-values",
        "500000,1000000",
        "--max-new-names-list",
        "5",
        "--exit-hold-fracs",
        "0",
        "--switch-gap-fracs",
        "0",
        "--execution-constraint-mode",
        "realistic",
        "--start-date",
        start,
        "--end-date",
        end,
        "--max-data-date",
        max_date,
        "--ohlc-backend",
        backend,
        "--market-daily-store-root",
        args.market_daily_store_root,
        "--ohlc-monthly-cache-dir",
        args.ohlc_monthly_cache_dir,
        "--execution-mask-cache-dir",
        str((output_dir / "execution_masks").resolve()),
        "--performance-report",
        str((output_dir / "performance.json").resolve()),
        "--save-path-details",
        "--resume",
    ]


def sweep_is_complete(root: Path) -> bool:
    summary_path = root / "open_price_ledger_param_sweep_summary.csv"
    index_path = root / "path_artifact_index.csv"
    if not summary_path.is_file() or not index_path.is_file():
        return False
    try:
        summary = pd.read_csv(summary_path)
        path_index = pd.read_csv(index_path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    required_summary = {"alpha_name", "stress", "portfolio_value"}
    required_index = {"sweep_key_sha256", *ARTIFACT_COLUMNS}
    if not required_summary.issubset(summary.columns):
        return False
    if not required_index.issubset(path_index.columns):
        return False
    expected = {
        (CANDIDATE_ID, stress, float(capital))
        for stress in STRESSES.split(",")
        for capital in (500000, 1000000)
    }
    actual = {
        (str(row.alpha_name), str(row.stress), float(row.portfolio_value))
        for row in summary.itertuples(index=False)
    }
    if actual != expected or len(summary) != len(expected):
        return False
    if len(path_index) != len(expected):
        return False
    if path_index["sweep_key_sha256"].astype(str).duplicated().any():
        return False
    for artifact in ARTIFACT_COLUMNS:
        if not all(Path(str(path)).is_file() for path in path_index[artifact]):
            return False
    return True


def parse_args(argv=None):
    configured_store, configured_cache = configured_candidate_paths()
    parser = argparse.ArgumentParser(
        description="Run fixed CSV/monthly 24-cell open-ledger parity."
    )
    parser.add_argument(
        "--output-root",
        default=(
            "reports/non_training_closure_20260719/"
            "nt6_market_data_pilot_20260730/md6_ledger_backend_parity"
        ),
    )
    parser.add_argument("--research-data-dir", default="data/raw")
    parser.add_argument("--forward-data-dir", default="data/forward_raw")
    parser.add_argument(
        "--market-daily-store-root",
        default=configured_store,
    )
    parser.add_argument(
        "--ohlc-monthly-cache-dir",
        default=configured_cache,
    )
    parser.add_argument("--min-free-memory-gib", type=float, default=3.0)
    parser.add_argument("--estimated-peak-memory-gib", type=float, default=0.75)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_root = Path(args.output_root).resolve()
    status_path = output_root / "matrix_status.json"
    status = {
        "schema": "open_ledger_backend_parity_matrix_v1",
        "candidate_id": CANDIDATE_ID,
        "python": sys.executable,
        "selection_splits": ["val_2024", "test_2025"],
        "observation_splits": ["forward_2026"],
        "stresses": STRESSES.split(","),
        "capitals": [500000, 1000000],
        "cell_count_per_backend": 24,
        "backends": list(BACKENDS),
        "runs": [],
        "comparisons": [],
        "updated_at": _utc_now(),
    }
    for split in SPLITS:
        for backend in BACKENDS:
            run_root = output_root / backend / split
            command = build_sweep_command(
                args,
                split=split,
                backend=backend,
                output_dir=run_root,
            )
            run_record = {
                "split": split,
                "backend": backend,
                "output_root": str(run_root),
                "command": command,
            }
            if sweep_is_complete(run_root):
                run_record["status"] = "already_complete"
            elif args.dry_run:
                run_record["status"] = "dry_run"
            else:
                free_gib = available_memory_gib()
                run_record["available_memory_gib_before"] = round(free_gib, 3)
                required_before = (
                    args.min_free_memory_gib + args.estimated_peak_memory_gib
                )
                if free_gib < required_before:
                    run_record["status"] = "blocked_low_memory"
                    run_record["required_memory_gib_before"] = required_before
                    run_record["reserved_memory_gib"] = args.min_free_memory_gib
                    run_record["estimated_peak_memory_gib"] = (
                        args.estimated_peak_memory_gib
                    )
                    status["runs"].append(run_record)
                    status["status"] = "blocked_low_memory"
                    status["updated_at"] = _utc_now()
                    _atomic_json(status_path, status)
                    print(
                        f"blocked: free_memory={free_gib:.2f} GiB "
                        f"required_before={required_before:.2f} GiB "
                        f"reserve={args.min_free_memory_gib:.2f} GiB "
                        f"next={backend}/{split}",
                        flush=True,
                    )
                    return 3
                result = subprocess.run(command, cwd=ROOT)
                run_record["returncode"] = result.returncode
                run_record["status"] = (
                    "completed" if result.returncode == 0 else "failed"
                )
                if result.returncode != 0:
                    status["runs"].append(run_record)
                    status["status"] = "failed"
                    status["updated_at"] = _utc_now()
                    _atomic_json(status_path, status)
                    return result.returncode
            status["runs"].append(run_record)
            status["updated_at"] = _utc_now()
            _atomic_json(status_path, status)

        if not args.dry_run:
            comparison = compare_sweep_roots(
                output_root / "csv" / split,
                output_root / "monthly" / split,
            )
            comparison_path = output_root / "parity" / f"{split}.json"
            _atomic_json(comparison_path, comparison)
            status["comparisons"].append(
                {
                    "split": split,
                    "status": comparison["status"],
                    "report": str(comparison_path),
                }
            )
            if comparison["status"] != "passed":
                status["status"] = "failed_parity"
                status["updated_at"] = _utc_now()
                _atomic_json(status_path, status)
                return 2

    status["status"] = "dry_run" if args.dry_run else "passed"
    status["updated_at"] = _utc_now()
    _atomic_json(status_path, status)
    print(f"status={status['status']} manifest={status_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
