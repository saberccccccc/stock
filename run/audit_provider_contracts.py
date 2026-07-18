"""Build a read-only Phase D provider-contract audit from local project data."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
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


def main(argv=None):
    args = parse_args(argv)
    data_dir = _resolve(args.data_dir)
    view = DataView.create(
        name="selection_2024_2025",
        physical_root=data_dir,
        feature_warmup_start="2010-01-04",
        feature_warmup_end="2023-12-31",
        task_start="2010-01-04",
        task_end="2025-12-31",
        evaluation_start="2024-01-01",
        evaluation_end="2025-12-31",
        max_data_date="2025-12-31",
    )
    meta_path = _resolve(args.v14_meta)
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    providers = {
        "ohlcv": OhlcvMatrixProvider(
            data_view=view,
            cache_dir=_resolve(args.matrix_cache),
        ).manifest(),
        "v14": V14MemmapProvider(
            meta=meta,
            meta_path=meta_path,
            data_view=view,
        ).manifest(),
        "fundamentals": FundamentalPITProvider(
            source_path=_resolve(args.fundamentals),
            data_view=view,
        ).manifest(),
        "external_markets": ExternalMarketPITProvider(
            feature_path=_resolve(args.global_features),
            summary_path=_resolve(args.global_summary),
            data_view=view,
        ).manifest(),
        "execution_constraints": ExecutionConstraintProvider(
            data_view=view,
            matrix_cache_dir=_resolve(args.matrix_cache),
            dataset_role="research",
        ).manifest(),
    }
    execution_gaps = providers["execution_constraints"]["coverage"]["gaps"]
    report = {
        "schema_version": 1,
        "status": "complete_with_declared_external_gaps" if execution_gaps else "complete",
        "provider_count": len(providers),
        "providers": providers,
        "blockers": execution_gaps,
        "note": "Historical ST remains an external data limitation; provider interfaces do not fabricate coverage.",
    }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({"output": str(output), "status": report["status"], "blockers": execution_gaps}, ensure_ascii=False))


if __name__ == "__main__":
    main()
