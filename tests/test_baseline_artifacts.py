import pandas as pd

from experiments.baseline_artifacts import (
    apply_baseline_lineage,
    canonical_report_rows,
    evidence_is_canonical,
)


def test_legacy_rows_default_to_canonical_for_backward_compatibility():
    assert evidence_is_canonical(None)
    assert evidence_is_canonical("")
    assert evidence_is_canonical("true")
    assert not evidence_is_canonical("false")
    assert not evidence_is_canonical("superseded")


def test_apply_baseline_lineage_retains_history_and_repoints_formal_rows():
    reports = pd.DataFrame(
        [
            {
                "candidate_id": "base",
                "evidence_class": "legacy_registered",
                "split": "val_2024",
                "stress": "normal",
                "capital": 500000,
                "path": "legacy.csv",
                "experiment_manifest": "",
            },
            {
                "candidate_id": "base",
                "evidence_class": "formal_experiment",
                "split": "val_2024",
                "stress": "normal",
                "capital": 500000,
                "path": "old/formal.csv",
                "experiment_manifest": "old/manifest.json",
            },
            {
                "candidate_id": "base",
                "evidence_class": "legacy_registered",
                "split": "forward_2026",
                "stress": "normal",
                "capital": 500000,
                "path": "forward.csv",
                "experiment_manifest": "",
            },
            {
                "candidate_id": "other",
                "evidence_class": "legacy_registered",
                "split": "val_2024",
                "stress": "normal",
                "capital": 500000,
                "path": "other.csv",
                "experiment_manifest": "",
            },
        ]
    )

    result = apply_baseline_lineage(
        reports,
        candidate_id="base",
        canonical_ledger_root="reports/canonical",
        canonical_manifest="reports/canonical/experiment_manifest.json",
    )

    assert result.loc[0, "canonical_evidence"] == "false"
    assert result.loc[0, "superseded_by"] == "reports/canonical/experiment_manifest.json"
    assert result.loc[1, "canonical_evidence"] == "true"
    assert result.loc[1, "path"] == "reports/canonical/val_2024/base/normal/open_ledger_summary.csv"
    assert result.loc[1, "experiment_manifest"] == "reports/canonical/experiment_manifest.json"
    assert result.loc[2, "canonical_evidence"] == "true"
    assert evidence_is_canonical(result.loc[3, "canonical_evidence"])
    assert len(canonical_report_rows(result.loc[result["candidate_id"] == "base"])) == 2
