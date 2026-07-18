from pathlib import Path

import pandas as pd
import pytest

from run.summarize_apm_scorecard import (
    aggregate_candidate_selection,
    load_scorecard_rows,
    main,
    parse_input_spec,
    rank_scorecard,
    split_selection_observation,
)


def test_parse_input_spec_keeps_windows_drive_colon():
    spec = parse_input_spec("v9:test:normal:C:/tmp/open_ledger_summary.csv")

    assert spec["candidate"] == "v9"
    assert spec["split"] == "test"
    assert spec["scenario"] == "normal"
    assert spec["path"] == Path("C:/tmp/open_ledger_summary.csv")


def _coverage():
    return {
        "signal_start": "2026-01-05",
        "signal_end": "2026-06-30",
        "backtest_start": "2026-01-06",
        "backtest_end": "2026-06-30",
    }


def test_apm_scorecard_ranks_by_information_ratio(tmp_path):
    summary = tmp_path / "open_ledger_summary.csv"
    pd.DataFrame(
        [
            {
                "portfolio_value": 500_000,
                **_coverage(),
                "target_frac": 0.006,
                "hold_frac": 0.10,
                "n_return_days": 100,
                "ann": 50.0,
                "sharpe": 1.5,
                "mdd": 0.12,
                "active_ann": 40.0,
                "information_ratio": 1.2,
                "beta_to_benchmark": 0.8,
                "avg_executed_turnover": 0.3,
                "total_cost": 0.05,
            },
            {
                "portfolio_value": 1_000_000,
                **_coverage(),
                "target_frac": 0.006,
                "hold_frac": 0.10,
                "n_return_days": 100,
                "ann": 55.0,
                "sharpe": 1.4,
                "mdd": 0.10,
                "active_ann": 35.0,
                "information_ratio": 1.6,
                "beta_to_benchmark": 0.7,
                "avg_executed_turnover": 0.25,
                "total_cost": 0.04,
            },
        ]
    ).to_csv(summary, index=False)

    frame = load_scorecard_rows(
        [
            {
                "candidate": "candidate_a",
                "split": "val",
                "scenario": "normal",
                "path": summary,
            }
        ]
    )
    ranked = rank_scorecard(frame)

    assert ranked.iloc[0]["portfolio_value"] == 1_000_000
    assert ranked.iloc[0]["apm_rank"] == 1
    assert ranked.iloc[0]["signal_end"] == "2026-06-30"


def test_apm_scorecard_main_writes_outputs(tmp_path):
    summary = tmp_path / "open_ledger_summary.csv"
    pd.DataFrame(
        [
            {
                "portfolio_value": 500_000,
                **_coverage(),
                "ann": 20.0,
                "sharpe": 1.0,
                "mdd": 0.08,
                "active_ann": 15.0,
                "information_ratio": 0.9,
            }
        ]
    ).to_csv(summary, index=False)
    output = tmp_path / "out"

    main([
        "--input",
        f"v9:val:normal:{summary}",
        "--output-dir",
        str(output),
    ])

    assert (output / "apm_scorecard_long.csv").exists()
    assert (output / "apm_scorecard_selection_input.csv").exists()
    assert (output / "apm_scorecard_observation.csv").exists()
    assert (output / "apm_scorecard_ranked.csv").exists()
    assert (output / "apm_candidate_selection_summary.csv").exists()
    assert (output / "apm_candidate_selection_summary.md").exists()
    assert (output / "apm_scorecard.md").exists()


def test_apm_scorecard_rejects_missing_coverage_columns(tmp_path):
    summary = tmp_path / "open_ledger_summary.csv"
    pd.DataFrame([{"portfolio_value": 500_000, "ann": 20.0}]).to_csv(summary, index=False)

    with pytest.raises(ValueError, match="missing required coverage columns"):
        load_scorecard_rows(
            [
                {
                    "candidate": "candidate_a",
                    "split": "val",
                    "scenario": "normal",
                    "path": summary,
                }
            ]
        )


def test_forward_split_is_observation_not_selection(tmp_path):
    val_summary = tmp_path / "val_summary.csv"
    forward_summary = tmp_path / "forward_summary.csv"
    pd.DataFrame(
        [
            {
                "portfolio_value": 500_000,
                **_coverage(),
                "ann": 20.0,
                "sharpe": 1.0,
                "mdd": 0.08,
                "active_ann": 15.0,
                "information_ratio": 0.9,
            }
        ]
    ).to_csv(val_summary, index=False)
    pd.DataFrame(
        [
            {
                "portfolio_value": 500_000,
                **_coverage(),
                "ann": 200.0,
                "sharpe": 9.0,
                "mdd": 0.01,
                "active_ann": 180.0,
                "information_ratio": 8.0,
            }
        ]
    ).to_csv(forward_summary, index=False)

    frame = load_scorecard_rows(
        [
            {
                "candidate": "candidate_a",
                "split": "val_2024",
                "scenario": "normal",
                "path": val_summary,
            },
            {
                "candidate": "candidate_a",
                "split": "forward_2026",
                "scenario": "normal",
                "path": forward_summary,
            },
        ]
    )

    selection, observation = split_selection_observation(frame)
    ranked = rank_scorecard(selection)

    assert ranked["split"].tolist() == ["val_2024"]
    assert observation["split"].tolist() == ["forward_2026"]


def test_candidate_selection_summary_aggregates_selection_rows():
    frame = pd.DataFrame(
        [
            {
                "candidate": "a",
                "split": "val_2024",
                "scenario": "normal",
                "portfolio_value": 500_000,
                "information_ratio": 1.0,
                "active_ann": 20.0,
                "ann": 30.0,
                "sharpe": 1.2,
                "mdd": 0.10,
                "avg_executed_turnover": 0.30,
                "total_cost": 0.05,
            },
            {
                "candidate": "a",
                "split": "test_2025",
                "scenario": "lag1",
                "portfolio_value": 1_000_000,
                "information_ratio": 1.2,
                "active_ann": 25.0,
                "ann": 35.0,
                "sharpe": 1.4,
                "mdd": 0.12,
                "avg_executed_turnover": 0.32,
                "total_cost": 0.06,
            },
            {
                "candidate": "b",
                "split": "val_2024",
                "scenario": "normal",
                "portfolio_value": 500_000,
                "information_ratio": 0.8,
                "active_ann": 30.0,
                "ann": 40.0,
                "sharpe": 1.8,
                "mdd": 0.08,
                "avg_executed_turnover": 0.20,
                "total_cost": 0.03,
            },
        ]
    )

    summary = aggregate_candidate_selection(frame)

    assert summary.iloc[0]["candidate"] == "a"
    assert summary.iloc[0]["selection_rows"] == 2
    assert summary.iloc[0]["capital_count"] == 2
    assert summary.iloc[0]["mean_information_ratio"] == pytest.approx(1.1)
    assert summary.iloc[0]["worst_mdd"] == pytest.approx(0.12)
