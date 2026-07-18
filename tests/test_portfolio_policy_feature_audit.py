import pandas as pd
import pytest

from run.audit_portfolio_policy_features import (
    audit_event_file,
    parse_event_spec,
    summarize_audit,
)


def test_parse_event_spec_keeps_windows_drive_colon():
    spec = parse_event_spec("demo:val_2024:normal:C:/tmp/replacement_events.csv")

    assert spec["candidate"] == "demo"
    assert spec["split"] == "val_2024"
    assert spec["scenario"] == "normal"
    assert spec["path"].drive == "C:"
    assert spec["path"].name == "replacement_events.csv"


def test_feature_audit_marks_missing_required_columns(tmp_path):
    path = tmp_path / "replacement_events.csv"
    pd.DataFrame(
        [
            {
                "diag_gross_weight": 0.8,
                "diag_market_mult": 1.0,
                "label_available": True,
                "ledger_path_utility": 0.01,
            }
        ]
    ).to_csv(path, index=False)
    spec = {
        "candidate": "demo",
        "split": "val_2024",
        "scenario": "normal",
        "path": path,
    }

    rows = audit_event_file(spec)
    summary = summarize_audit(rows)
    detail = pd.DataFrame(rows)

    assert summary.iloc[0]["status"] == "incomplete"
    assert summary.iloc[0]["missing_columns"] > 0
    label_rows = detail[detail["group"] == "label_or_future_evidence"]
    assert set(label_rows["column"]) >= {"label_available", "ledger_path_utility"}
    assert label_rows.set_index("column").loc["label_available", "status"] == "present"


def test_feature_audit_complete_when_required_columns_present(tmp_path):
    path = tmp_path / "replacement_events.csv"
    row = {
        "diag_gross_weight": 0.8,
        "diag_market_mult": 1.0,
        "diag_portfolio_beta_60d": 0.9,
        "diag_portfolio_beta_per_gross_60d": 1.1,
        "diag_portfolio_specific_vol_60d": 0.06,
        "diag_turnover": 0.3,
        "diag_executed_turnover": 0.3,
        "diag_active_drawdown_trailing_return": -0.02,
        "diag_global_risk_pressure": 0.4,
        "cand_global_us_hk_pressure": 0.3,
        "cand_global_defensive_pressure": 0.2,
        "cand_global_hk_risk_pressure": 0.1,
        "diff_candidate_industry_top_share": -0.02,
        "diff_top_industry_share": -0.01,
        "diff_top_industry_hhi": -0.001,
        "cand_top_industry_share": 0.2,
        "base_top_industry_share": 0.22,
        "diff_candidate_rank_pct": 0.01,
        "diff_ret_1d": 0.0,
        "diff_ret_5d": 0.01,
        "diff_ret_20d": -0.02,
        "diff_vol_20d": 0.01,
        "diff_vol_60d": 0.01,
        "diff_drawdown_20d": 0.0,
        "diff_beta_60d": -0.02,
        "diff_specific_vol_60d": 0.001,
        "diff_money_ma20": 1000.0,
        "diff_was_held": 0.0,
        "diff_holding_age": -2.0,
        "ledger_cost": 0.001,
        "pair_risk_delta": -0.01,
        "pair_downside_delta": -0.01,
        "pair_quick_fade_delta": 0.0,
        "pair_rank_delta": 0.02,
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    spec = {
        "candidate": "demo",
        "split": "val_2024",
        "scenario": "normal",
        "path": path,
    }

    summary = summarize_audit(audit_event_file(spec))

    assert summary.iloc[0]["status"] == "complete"
    assert summary.iloc[0]["missing_columns"] == 0
    assert summary.iloc[0]["max_missing_rate"] == pytest.approx(0.0)


def test_feature_audit_reads_parquet(tmp_path):
    path = tmp_path / "replacement_events.parquet"
    pd.DataFrame(
        [
            {
                "diag_gross_weight": 0.8,
                "diag_market_mult": 1.0,
                "label_available": False,
            }
        ]
    ).to_parquet(path, index=False)
    spec = {
        "candidate": "demo",
        "split": "val_2024",
        "scenario": "normal",
        "path": path,
    }

    rows = audit_event_file(spec)

    assert rows
    assert any(row["column"] == "diag_gross_weight" for row in rows)
