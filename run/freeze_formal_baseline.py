"""Freeze the Registry baseline, canonical evidence and artifact inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text in sys.path:
    sys.path.remove(root_text)
sys.path.insert(0, root_text)

from experiments.baseline_artifacts import (
    apply_baseline_lineage,
    build_baseline_freeze,
    write_baseline_freeze,
    write_json_artifact,
)
from experiments.recording import validate_manifest_for_formal_use


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default="ledger_path_v3_t0001_nolookahead")
    parser.add_argument("--baselines", default="registry/baselines.yaml")
    parser.add_argument("--candidates", default="registry/candidates.csv")
    parser.add_argument("--reports", default="registry/reports.csv")
    parser.add_argument("--decision-rules", default="registry/decision_rules.json")
    parser.add_argument(
        "--canonical-workflow-root",
        default="reports/experiments/workflow_formal_baseline_ledger_replay_v4_20260716",
    )
    parser.add_argument("--output-dir", default="reports/non_training_closure_20260719/nt1_baseline_freeze")
    parser.add_argument("--registry-contract-output", default="registry/baseline_contract.json")
    parser.add_argument("--registry-lineage-output", default="registry/evidence_lineage.json")
    parser.add_argument("--apply-registry-lineage", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    baseline_config = yaml.safe_load((ROOT / args.baselines).read_text(encoding="utf-8"))
    decision_rules = json.loads((ROOT / args.decision_rules).read_text(encoding="utf-8"))
    candidates = pd.read_csv(ROOT / args.candidates)
    reports_path = ROOT / args.reports
    reports = pd.read_csv(reports_path)
    matches = candidates.loc[candidates["candidate_id"].astype(str) == args.candidate_id]
    if len(matches) != 1:
        raise ValueError(f"candidate_id must resolve exactly once: {args.candidate_id!r}")

    workflow_root = (ROOT / args.canonical_workflow_root).resolve()
    ledger_root = workflow_root / "ledger" / "formal_baseline_ledger_replay_v1"
    ledger_manifest = ledger_root / "experiment_manifest.json"
    validate_manifest_for_formal_use(ledger_manifest)
    canonical_manifest = ledger_manifest.relative_to(ROOT).as_posix()
    canonical_ledger_root = ledger_root.relative_to(ROOT).as_posix()

    updated_reports = apply_baseline_lineage(
        reports,
        candidate_id=args.candidate_id,
        canonical_ledger_root=canonical_ledger_root,
        canonical_manifest=canonical_manifest,
    )
    if args.apply_registry_lineage:
        for path in updated_reports.loc[
            (updated_reports["candidate_id"].astype(str) == args.candidate_id)
            & updated_reports["canonical_evidence"].map(lambda value: str(value).lower() == "true"),
            "path",
        ]:
            if not (ROOT / str(path)).is_file():
                raise FileNotFoundError(ROOT / str(path))
        updated_reports.to_csv(reports_path, index=False, encoding="utf-8")
        reports = updated_reports

    contract, inventory, lineage = build_baseline_freeze(
        ROOT,
        candidate=matches.iloc[0].to_dict(),
        baseline_config=baseline_config,
        decision_rules=decision_rules,
        canonical_workflow_root=workflow_root,
        reports=updated_reports,
    )
    written = write_baseline_freeze(
        ROOT / args.output_dir,
        contract=contract,
        inventory=inventory,
        lineage=lineage,
    )
    written["registry_contract"] = write_json_artifact(ROOT / args.registry_contract_output, contract)
    written["registry_lineage"] = write_json_artifact(ROOT / args.registry_lineage_output, lineage)
    print(json.dumps({key: str(path) for key, path in written.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
