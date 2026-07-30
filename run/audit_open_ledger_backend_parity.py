"""Compare two open-ledger sweep roots at summary and path-artifact level."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)


ARTIFACT_COLUMNS = (
    "equity_curve",
    "diagnostics",
    "positions",
    "orders",
    "rejections",
    "costs",
)
LINEAGE_COLUMNS = {
    "ohlc_backend",
    "market_daily_store_root",
    "ohlc_monthly_cache_dir",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _frame_difference(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_exact=True,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as exc:
        return str(exc)
    return None


def compare_sweep_roots(oracle_root: str | Path, candidate_root: str | Path) -> dict:
    oracle_root = Path(oracle_root).resolve()
    candidate_root = Path(candidate_root).resolve()
    summary_name = "open_price_ledger_param_sweep_summary.csv"
    index_name = "path_artifact_index.csv"
    oracle_summary = _read_csv(oracle_root / summary_name)
    candidate_summary = _read_csv(candidate_root / summary_name)
    common_columns = [
        column
        for column in oracle_summary.columns
        if column in candidate_summary.columns and column not in LINEAGE_COLUMNS
    ]
    missing_oracle_columns = sorted(
        set(candidate_summary.columns) - set(oracle_summary.columns) - LINEAGE_COLUMNS
    )
    missing_candidate_columns = sorted(
        set(oracle_summary.columns) - set(candidate_summary.columns) - LINEAGE_COLUMNS
    )
    alignment_columns = [
        column
        for column in ("alpha_name", "stress", "portfolio_value")
        if column in common_columns
    ]
    if alignment_columns:
        if oracle_summary.duplicated(alignment_columns).any():
            raise ValueError(
                f"oracle summary has duplicate alignment keys: {alignment_columns}"
            )
        if candidate_summary.duplicated(alignment_columns).any():
            raise ValueError(
                f"candidate summary has duplicate alignment keys: {alignment_columns}"
            )
    oracle_common = oracle_summary[common_columns]
    candidate_common = candidate_summary[common_columns]
    if alignment_columns:
        oracle_common = oracle_common.sort_values(
            alignment_columns, kind="stable"
        )
        candidate_common = candidate_common.sort_values(
            alignment_columns, kind="stable"
        )
    oracle_common = oracle_common.reset_index(drop=True)
    candidate_common = candidate_common.reset_index(drop=True)
    summary_difference = _frame_difference(oracle_common, candidate_common)

    oracle_index = _read_csv(oracle_root / index_name).set_index(
        "sweep_key_sha256", drop=False
    )
    candidate_index = _read_csv(candidate_root / index_name).set_index(
        "sweep_key_sha256", drop=False
    )
    oracle_keys = set(oracle_index.index.astype(str))
    candidate_keys = set(candidate_index.index.astype(str))
    shared_keys = sorted(oracle_keys & candidate_keys)
    artifact_results = []
    for key in shared_keys:
        left_row = oracle_index.loc[key]
        right_row = candidate_index.loc[key]
        if isinstance(left_row, pd.DataFrame) or isinstance(right_row, pd.DataFrame):
            raise ValueError(f"duplicate sweep key in path artifact index: {key}")
        for artifact in ARTIFACT_COLUMNS:
            left_path = Path(str(left_row[artifact])).resolve()
            right_path = Path(str(right_row[artifact])).resolve()
            left = _read_csv(left_path)
            right = _read_csv(right_path)
            difference = _frame_difference(left, right)
            oracle_sha256 = _sha256(left_path)
            candidate_sha256 = _sha256(right_path)
            artifact_results.append(
                {
                    "sweep_key_sha256": key,
                    "artifact": artifact,
                    "oracle_path": str(left_path),
                    "candidate_path": str(right_path),
                    "oracle_sha256": oracle_sha256,
                    "candidate_sha256": candidate_sha256,
                    "row_count": len(left),
                    "exact_frame_equal": difference is None,
                    "byte_equal": oracle_sha256 == candidate_sha256,
                    "difference": difference,
                }
            )
    failures = [
        item for item in artifact_results if not item["exact_frame_equal"]
    ]
    passed = (
        summary_difference is None
        and not missing_oracle_columns
        and not missing_candidate_columns
        and oracle_keys == candidate_keys
        and not failures
    )
    return {
        "schema": "open_ledger_backend_parity_v1",
        "status": "passed" if passed else "failed",
        "oracle_root": str(oracle_root),
        "candidate_root": str(candidate_root),
        "summary": {
            "oracle_rows": len(oracle_summary),
            "candidate_rows": len(candidate_summary),
            "compared_columns": common_columns,
            "missing_oracle_columns": missing_oracle_columns,
            "missing_candidate_columns": missing_candidate_columns,
            "exact_equal": summary_difference is None,
            "difference": summary_difference,
        },
        "path_index": {
            "oracle_keys": sorted(oracle_keys),
            "candidate_keys": sorted(candidate_keys),
            "shared_key_count": len(shared_keys),
            "keys_equal": oracle_keys == candidate_keys,
        },
        "artifacts": {
            "comparison_count": len(artifact_results),
            "exact_frame_equal_count": sum(
                item["exact_frame_equal"] for item in artifact_results
            ),
            "byte_equal_count": sum(item["byte_equal"] for item in artifact_results),
            "failures": failures,
            "results": artifact_results,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit exact behavior parity between two open-ledger sweeps."
    )
    parser.add_argument("--oracle-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = compare_sweep_roots(args.oracle_root, args.candidate_root)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"status={report['status']} "
        f"summary_equal={report['summary']['exact_equal']} "
        f"artifact_failures={len(report['artifacts']['failures'])} "
        f"report={output.resolve()}",
        flush=True,
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
