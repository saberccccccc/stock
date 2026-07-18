import pandas as pd

from run.attribute_pairwise_policy_changes import main


def test_attribute_pairwise_policy_changes_outputs_event_summaries(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    label = tmp_path / "label.parquet"
    output_dir = tmp_path / "out"

    pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "policy_applied": [1, 1],
            "best_score": [0.01, 0.02],
            "threshold": [0.0001, 0.0001],
            "baseline_code": ["000001.SZ", "000002.SZ"],
            "chosen_code": ["000003.SZ", "000004.SZ"],
            "chosen_original_position": [10, 11],
            "baseline_position": [31, 32],
        }
    ).to_csv(baseline, index=False)

    pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "policy_applied": [1, 1],
            "best_score": [0.01, 0.02],
            "threshold": [0.0001, 0.0001],
            "baseline_code": ["000001.SZ", "000002.SZ"],
            "chosen_code": ["000003.SZ", "000005.SZ"],
            "chosen_original_position": [10, 12],
            "baseline_position": [31, 32],
            "risk_guard_active": [0, 1],
            "risk_guard_penalty": [0.0, 0.001],
        }
    ).to_csv(candidate, index=False)

    pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "code": ["000003.SZ", "000005.SZ"],
            "baseline_code": ["000001.SZ", "000002.SZ"],
            "candidate_position": [10, 12],
            "ledger_path_utility": [0.001, -0.002],
            "ledger_weighted_raw_edge": [0.0012, -0.0018],
            "pair_path_raw_edge": [0.05, -0.08],
            "pair_risk_delta": [-0.1, 0.2],
            "pair_downside_delta": [-0.02, 0.03],
            "pair_quick_fade_delta": [0.0, 0.01],
            "diff_ret_20d": [0.03, -0.04],
            "diff_specific_vol_60d": [-0.03, 0.06],
            "diag_active_drawdown_trailing_return": [0.02, -0.01],
            "diag_global_risk_pressure": [0.01, 0.06],
            "cand_ret_20d": [0.05, -0.02],
            "cand_specific_vol_60d": [0.13, 0.14],
        }
    ).to_parquet(label, index=False)

    main(
        [
            "--baseline-audit",
            str(baseline),
            "--candidate-audit",
            str(candidate),
            "--label-dataset",
            str(label),
            "--candidate-name",
            "candidate_x",
            "--split",
            "test",
            "--output-dir",
            str(output_dir),
        ]
    )

    events = pd.read_csv(output_dir / "policy_change_events.csv")
    assert events["decision_changed"].tolist() == [0, 1]
    assert events["label_matched"].tolist() == [1, 1]
    assert (output_dir / "by_decision_changed.csv").exists()
    assert (output_dir / "policy_change_attribution.md").exists()
