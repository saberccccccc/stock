"""Build a read-only Phase D provider-contract audit from local project data."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.providers import (
    DataView,
    ExecutionConstraintProvider,
    ExternalMarketPITProvider,
    FundamentalPITProvider,
    OhlcvMatrixProvider,
    V14MemmapProvider,
)
from core.research_protocol import FORWARD_DATA_DIR, RESEARCH_DATA_DIR, SPLIT_SPECS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--split",
        choices=("selection_2024_2025", *SPLIT_SPECS),
        default="selection_2024_2025",
    )
    parser.add_argument("--matrix-cache", default="cache/open_ledger_ohlc_matrix")
    parser.add_argument("--v14-meta", required=True)
    parser.add_argument("--fundamentals", default="cache/fundamental_features_akshare.parquet")
    parser.add_argument("--global-features", default="data/global/global_overnight_features.parquet")
    parser.add_argument("--global-summary", default="data/global/global_overnight_features_summary.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def build_data_view(split, data_dir=None):
    if split == "selection_2024_2025":
        evaluation_start, evaluation_end = "2024-01-01", "2025-12-31"
        max_data_date = "2025-12-31"
        dataset_role = "research"
    else:
        spec = SPLIT_SPECS[split]
        evaluation_start = str(spec.start.date())
        evaluation_end = str(spec.end.date())
        max_data_date = str(spec.max_data_date.date())
        dataset_role = "forward" if spec.is_forward else "research"
    physical_root = data_dir or (FORWARD_DATA_DIR if dataset_role == "forward" else RESEARCH_DATA_DIR)
    view = DataView.create(
        name=split,
        physical_root=_resolve(physical_root),
        feature_warmup_start="2010-01-04",
        feature_warmup_end=str((pd.Timestamp(evaluation_start) - pd.Timedelta(days=1)).date()),
        task_start="2010-01-04",
        task_end=max_data_date,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        max_data_date=max_data_date,
    )
    return view, dataset_role


def capture_provider(name, factory, blockers):
    try:
        manifest = factory()
    except Exception as exc:
        blocker = f"provider_{name}_unavailable"
        blockers.append(blocker)
        return {
            "status": "unavailable",
            "blocker": blocker,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    if name == "ohlcv" and not manifest["cache"]["matches_data_view"]:
        blockers.append("ohlcv_cache_identity_mismatch")
    return manifest


def main(argv=None):
    args = parse_args(argv)
    view, dataset_role = build_data_view(args.split, args.data_dir)
    meta_path = _resolve(args.v14_meta)
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    blockers = []
    providers = {
        "ohlcv": capture_provider("ohlcv", lambda: OhlcvMatrixProvider(
            data_view=view,
            cache_dir=_resolve(args.matrix_cache),
        ).manifest(ensure_cache=False), blockers),
        "v14": capture_provider("v14", lambda: V14MemmapProvider(
            meta=meta,
            meta_path=meta_path,
            data_view=view,
        ).manifest(), blockers),
        "fundamentals": capture_provider("fundamentals", lambda: FundamentalPITProvider(
            source_path=_resolve(args.fundamentals),
            data_view=view,
        ).manifest(), blockers),
        "external_markets": capture_provider("external_markets", lambda: ExternalMarketPITProvider(
            feature_path=_resolve(args.global_features),
            summary_path=_resolve(args.global_summary),
            data_view=view,
        ).manifest(), blockers),
        "execution_constraints": capture_provider("execution_constraints", lambda: ExecutionConstraintProvider(
            data_view=view,
            matrix_cache_dir=_resolve(args.matrix_cache),
            dataset_role=dataset_role,
        ).manifest(), blockers),
    }
    execution = providers["execution_constraints"]
    execution_gaps = execution.get("coverage", {}).get("gaps", [])
    blockers.extend(gap for gap in execution_gaps if gap not in blockers)
    report = {
        "schema_version": 1,
        "split": args.split,
        "dataset_role": dataset_role,
        "status": "complete_with_declared_external_gaps" if blockers else "complete",
        "provider_count": len(providers),
        "providers": providers,
        "blockers": blockers,
        "note": "Read-only audit: cache mismatches and unavailable providers are reported, never rebuilt or fabricated.",
    }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({"output": str(output), "status": report["status"], "blockers": blockers}, ensure_ascii=False))


if __name__ == "__main__":
    main()
