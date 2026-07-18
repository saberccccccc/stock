"""Audit project-native Alpha158-inspired baseline specifications."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.factor_baselines import audit_factor_baselines


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transform-contract",
        default="reports/qlib_research_framework_20260712/v14_transform_contract.json",
    )
    parser.add_argument(
        "--output",
        default="reports/qlib_research_framework_20260712/factor_baseline_audit.json",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    contract = json.loads((ROOT / args.transform_contract).read_text(encoding="utf-8"))
    features = contract["cache_metadata"]["feature_columns"]
    baselines = audit_factor_baselines(features)
    report = {
        "schema_version": 1,
        "source_contract": str(Path(args.transform_contract)),
        "label_contract": "oo_lag1; task-level label-tail purge remains required",
        "baselines": baselines,
        "status": "ready_for_training" if all(item["ready"] for item in baselines.values()) else "blocked",
        "not_in_scope": ["Qlib provider data", "Qlib close-return labels", "automatic inclusion in v14"],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
