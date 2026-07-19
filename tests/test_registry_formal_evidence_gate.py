import pandas as pd
import pytest

import run.scorecard_from_registry as scorecard


def _candidates():
    return pd.DataFrame([{"candidate_id": "demo", "status": "research"}])


def _reports(evidence_class, experiment_manifest=""):
    return pd.DataFrame(
        [
            {
                "candidate_id": "demo",
                "path": "summary.csv",
                "split": "val_2024",
                "stress": "normal",
                "capital": 500000,
                "execution_mode": "realistic",
                "selection_eligible": True,
                "is_forward": False,
                "evidence_class": evidence_class,
                "experiment_manifest": experiment_manifest,
            }
        ]
    )


def _summary():
    return pd.DataFrame(
        [
            {
                "portfolio_value": 500000,
                "execution_constraint_mode": "realistic",
                "signal_start": "2024-01-02",
                "signal_end": "2024-12-31",
                "backtest_start": "2024-01-03",
                "backtest_end": "2024-12-31",
                "ann": 10.0,
                "sharpe": 1.0,
            }
        ]
    )


def test_legacy_registered_evidence_remains_readable_but_explicit(monkeypatch):
    monkeypatch.setattr(scorecard, "load_summary", lambda path: _summary())

    result = scorecard.build_long(_reports("legacy_registered"), _candidates(), "realistic", False)

    assert len(result) == 1


def test_explicitly_superseded_evidence_is_not_loaded(monkeypatch):
    monkeypatch.setattr(scorecard, "load_summary", lambda path: _summary())
    reports = _reports("legacy_registered")
    reports["canonical_evidence"] = "false"

    result = scorecard.build_long(reports, _candidates(), "realistic", False)

    assert result.empty


def test_formal_evidence_requires_manifest(monkeypatch):
    monkeypatch.setattr(scorecard, "load_summary", lambda path: _summary())

    with pytest.raises(ValueError, match="missing experiment_manifest"):
        scorecard.build_long(_reports("formal_experiment"), _candidates(), "realistic", False)


def test_formal_evidence_revalidates_manifest(monkeypatch):
    called = []
    monkeypatch.setattr(scorecard, "load_summary", lambda path: _summary())
    monkeypatch.setattr(
        scorecard,
        "validate_manifest_for_formal_use",
        lambda path: called.append(path) or {"complete": True},
    )

    result = scorecard.build_long(
        _reports("formal_experiment", "reports/experiments/demo/experiment_manifest.json"),
        _candidates(),
        "realistic",
        False,
    )

    assert len(result) == 1
    assert called == [scorecard.ROOT / "reports/experiments/demo/experiment_manifest.json"]


def test_unknown_evidence_class_is_rejected(monkeypatch):
    monkeypatch.setattr(scorecard, "load_summary", lambda path: _summary())

    with pytest.raises(ValueError, match="unsupported or missing evidence_class"):
        scorecard.build_long(_reports(""), _candidates(), "realistic", False)


def test_scorecard_candidate_filter_rejects_unknown_id(tmp_path, monkeypatch):
    monkeypatch.setattr(scorecard, "ROOT", tmp_path)
    _reports("legacy_registered").to_csv(tmp_path / "reports.csv", index=False)
    _candidates().to_csv(tmp_path / "candidates.csv", index=False)
    (tmp_path / "rules.json").write_text("{}", encoding="utf-8")

    with pytest.raises(KeyError, match="unknown candidate IDs"):
        scorecard.main(
            [
                "--reports-csv",
                "reports.csv",
                "--candidates-csv",
                "candidates.csv",
                "--decision-rules",
                "rules.json",
                "--candidate-id",
                "missing",
            ]
        )
