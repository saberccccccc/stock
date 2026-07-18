import pandas as pd

from run.audit_apm_completeness import (
    audit_attribution_dir,
    parse_summary_spec,
    run_audit,
)


def test_parse_summary_spec_keeps_windows_drive_colon():
    spec = parse_summary_spec("v9:test_2025_20260518:normal:C:/tmp/open_ledger_summary.csv")

    assert spec["candidate"] == "v9"
    assert spec["split"] == "test_2025_20260518"
    assert spec["scenario"] == "normal"
    assert str(spec["path"]).replace("\\", "/") == "C:/tmp/open_ledger_summary.csv"


def test_audit_marks_legacy_v9_and_missing_coverage(tmp_path):
    summary = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "ann": 10.0,
                "sharpe": 1.0,
                "mdd": 0.1,
                "benchmark_ann": 2.0,
                "active_ann": 8.0,
                "tracking_error": 0.2,
                "information_ratio": 0.8,
                "beta_to_benchmark": 0.3,
                "avg_executed_turnover": 0.2,
                "avg_unfilled_turnover": 0.0,
                "total_cost": 0.01,
                "blocked_buy": 0,
                "blocked_sell": 0,
                "adv_blocked": 0,
                "avg_portfolio_beta_60d": 1.0,
                "avg_portfolio_beta_per_gross_60d": 1.2,
                "avg_portfolio_specific_vol_60d": 0.1,
            }
        ]
    ).to_csv(summary, index=False)

    rows = run_audit(
        "official_v9",
        [
            {
                "candidate": "official_v9",
                "split": "test_2025_20260518",
                "scenario": "normal",
                "path": summary,
            }
        ],
        [],
    )
    by_item = {row["item"]: row for row in rows}

    assert by_item["protocol:trust_status"]["status"] == "weak"
    assert by_item["coverage:required_splits"]["status"] == "incomplete"
    assert by_item["coverage:required_stresses"]["status"] == "incomplete"
    assert by_item[
        "summary:official_v9:test_2025_20260518:normal:risk_columns"
    ]["status"] == "complete"


def test_audit_requires_each_split_to_have_all_stresses(tmp_path):
    summary = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "ann": 10.0,
                "sharpe": 1.0,
                "mdd": 0.1,
                "benchmark_ann": 2.0,
                "active_ann": 8.0,
                "tracking_error": 0.2,
                "information_ratio": 0.8,
                "beta_to_benchmark": 0.3,
                "avg_executed_turnover": 0.2,
                "avg_unfilled_turnover": 0.0,
                "total_cost": 0.01,
                "blocked_buy": 0,
                "blocked_sell": 0,
                "adv_blocked": 0,
                "avg_portfolio_beta_60d": 1.0,
                "avg_portfolio_beta_per_gross_60d": 1.2,
                "avg_portfolio_specific_vol_60d": 0.1,
            }
        ]
    ).to_csv(summary, index=False)

    specs = [
        {"candidate": "x", "split": "validation_2024", "scenario": "normal", "path": summary},
        {"candidate": "x", "split": "test_2025_20260518", "scenario": "lag1", "path": summary},
        {"candidate": "x", "split": "forward_shadow", "scenario": "cost2x", "path": summary},
        {"candidate": "x", "split": "forward_shadow", "scenario": "capacity_3pct", "path": summary},
    ]
    rows = run_audit("candidate_x", specs, [], trusted_protocol=True)
    by_item = {row["item"]: row for row in rows}

    assert by_item["coverage:required_splits"]["status"] == "complete"
    assert by_item["coverage:required_stresses"]["status"] == "complete"
    assert by_item["coverage:split_stresses:validation_2024"]["status"] == "incomplete"
    assert "lag1" in by_item["coverage:split_stresses:validation_2024"]["note"]
    assert by_item["coverage:split_stresses:test_2025_20260518"]["status"] == "incomplete"
    assert by_item["coverage:split_stresses:forward_shadow"]["status"] == "incomplete"


def test_audit_attribution_dir_requires_slice_summary(tmp_path):
    for name in (
        "apm_attribution_summary.csv",
        "apm_industry_exposure.csv",
        "apm_style_exposure.csv",
        "apm_attribution_report.md",
    ):
        (tmp_path / name).write_text("x\n", encoding="utf-8")

    rows = audit_attribution_dir(tmp_path)
    by_item = {row["item"]: row for row in rows}

    assert by_item[f"attribution:{tmp_path.name}:apm_slice_summary.csv"]["status"] == "missing"
