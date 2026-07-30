from types import SimpleNamespace

import pandas as pd
import pytest

from run.sweep_open_price_ledger_params import (
    append_summary_rows,
    apply_stress_overrides,
    completed_keys_from_summary,
    filter_alpha_rows,
    parse_args,
    parse_int_list,
    parse_stress_names,
    retention_param_grid,
    sweep_key,
    write_path_artifacts,
)


def test_append_summary_rows_preserves_prior_chunks(tmp_path):
    path = tmp_path / "summary.csv"
    append_summary_rows(path, [{"alpha_name": "a", "ann": 1.0}])
    append_summary_rows(path, [{"alpha_name": "b", "ann": 2.0}])

    frame = pd.read_csv(path)

    assert frame.to_dict("records") == [
        {"alpha_name": "a", "ann": 1.0},
        {"alpha_name": "b", "ann": 2.0},
    ]


def test_filter_alpha_rows_uses_inclusive_date_bounds():
    rows = [
        {"date": pd.Timestamp("2024-01-02")},
        {"date": pd.Timestamp("2024-06-28")},
        {"date": pd.Timestamp("2024-07-01")},
    ]
    filtered = filter_alpha_rows(rows, "2024-01-03", "2024-06-28")
    assert [row["date"] for row in filtered] == [pd.Timestamp("2024-06-28")]


def test_parse_args_does_not_truncate_market_data_by_default(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_open_price_ledger_params.py",
            "--alpha-specs",
            "alpha=reports/example.jsonl",
            "--output-dir",
            "reports/example_sweep",
        ],
    )

    args = parse_args()

    assert args.max_data_date is None


def test_sweep_key_is_stable_for_optional_state_pressure_column():
    common = ("alpha", "normal", 0.2, 0.0, None, 1.0, 0.0, 1.0, 500000, 5, None, 0.0, 0.006, 0.1)
    assert sweep_key(*common, state_aware_selection_pressure_col=None) != sweep_key(
        *common,
        state_aware_selection_pressure_col="global_defensive_pressure",
    )


def test_parse_args_accepts_performance_report(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_open_price_ledger_params.py",
            "--alpha-specs",
            "alpha=reports/example.jsonl",
            "--output-dir",
            "reports/example_sweep",
            "--performance-report",
            "timing.json",
        ],
    )

    args = parse_args()

    assert args.performance_report == "timing.json"


def test_parse_args_accepts_monthly_execution_backend(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_open_price_ledger_params.py",
            "--alpha-specs",
            "alpha=reports/example.jsonl",
            "--output-dir",
            "reports/example_sweep",
            "--ohlc-backend",
            "monthly",
            "--market-daily-store-root",
            "data/store",
            "--ohlc-monthly-cache-dir",
            "cache/monthly",
        ],
    )

    args = parse_args()

    assert args.ohlc_backend == "monthly"
    assert args.market_daily_store_root == "data/store"
    assert args.ohlc_monthly_cache_dir == "cache/monthly"


def test_parse_args_accepts_path_details(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_open_price_ledger_params.py",
            "--alpha-specs", "alpha=reports/example.jsonl",
            "--output-dir", "reports/example_sweep",
            "--save-path-details",
        ],
    )
    assert parse_args().save_path_details is True


def test_write_path_artifacts_separates_orders_rejections_and_costs(tmp_path):
    returns = pd.DataFrame([{"date": "2025-01-03", "return": 0.01, "equity_cny": 505000.0}])
    diagnostics = pd.DataFrame([{"date": "2025-01-03", "blocked_buy": 1}])
    orders = [
        {
            "date": "2025-01-03", "code": "A", "side": "buy", "status": "filled",
            "executed_shares": 100.0, "executed_value_cny": 1000.0,
            "commission_cny": 5.0, "stamp_tax_cny": 0.0,
            "slippage_cny": 0.5, "total_cost_cny": 5.5,
        },
        {
            "date": "2025-01-03", "code": "B", "side": "buy", "status": "rejected",
            "reason": "limit_up_open", "executed_shares": 0.0,
        },
    ]
    positions = [{"date": "2025-01-03", "code": "A", "shares": 100.0}]

    paths = write_path_artifacts(tmp_path, "cell", returns, diagnostics, orders, positions)

    assert set(paths) == {"equity_curve", "diagnostics", "positions", "orders", "rejections", "costs"}
    assert pd.read_csv(paths["orders"]).shape[0] == 2
    assert pd.read_csv(paths["rejections"])["code"].tolist() == ["B"]
    assert pd.read_csv(paths["costs"])["total_cost_cny"].tolist() == [5.5]


def test_write_path_artifacts_keeps_headers_for_empty_execution(tmp_path):
    paths = write_path_artifacts(
        tmp_path,
        "empty",
        pd.DataFrame(columns=["date", "return", "equity_cny"]),
        pd.DataFrame(columns=["date"]),
        [],
        [],
    )

    assert list(pd.read_csv(paths["orders"]).columns)[:3] == ["date", "code", "side"]
    assert list(pd.read_csv(paths["positions"]).columns)[:3] == ["date", "code", "shares"]
    assert "reason" in pd.read_csv(paths["rejections"]).columns


def test_write_path_artifacts_bounds_long_windows_paths(tmp_path):
    nested = tmp_path / ("workflow_" + "x" * 50) / ("ledger_" + "y" * 35) / "paths"
    paths = write_path_artifacts(
        nested,
        "candidate_" + "z" * 120,
        pd.DataFrame(columns=["date", "return", "equity_cny"]),
        pd.DataFrame(columns=["date"]),
        [],
        [],
    )

    assert all(path.is_file() for path in paths.values())
    assert max(len(str(path.resolve())) for path in paths.values()) <= 240


def test_parse_args_exposes_active_drawdown_throttle(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_open_price_ledger_params.py",
            "--alpha-specs",
            "alpha=reports/example.jsonl",
            "--output-dir",
            "reports/example_sweep",
            "--active-drawdown-throttle-lookback",
            "15",
            "--active-drawdown-throttle-trigger",
            "-0.03",
            "--active-drawdown-throttle-scale",
            "0.95",
            "--active-drawdown-throttle-cooldown",
            "5",
        ],
    )

    args = parse_args()

    assert args.active_drawdown_throttle_lookback == 15
    assert args.active_drawdown_throttle_trigger == pytest.approx(-0.03)
    assert args.active_drawdown_throttle_scale == pytest.approx(0.95)
    assert args.active_drawdown_throttle_cooldown == 5


def test_parse_args_exposes_topk_dropout_policy(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_open_price_ledger_params.py",
            "--alpha-specs", "alpha=reports/example.jsonl",
            "--output-dir", "reports/example_sweep",
            "--selection-policy", "topk_dropout",
            "--top-k", "30",
            "--n-drop", "3",
        ],
    )
    args = parse_args()
    assert args.selection_policy == "topk_dropout"
    assert args.top_k == 30
    assert args.n_drop == 3


def test_parse_args_exposes_state_aware_selection(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_open_price_ledger_params.py",
            "--alpha-specs", "alpha=reports/example.jsonl",
            "--output-dir", "reports/example_sweep",
            "--global-risk-features", "data/global/global_overnight_features.parquet",
            "--state-aware-selection-mode", "risk_rank",
            "--state-aware-selection-pressure-col", "global_defensive_pressure",
            "--state-aware-selection-rank-penalty", "0.10",
        ],
    )
    args = parse_args()
    assert args.state_aware_selection_mode == "risk_rank"
    assert args.state_aware_selection_pressure_col == "global_defensive_pressure"
    assert args.state_aware_selection_rank_penalty == pytest.approx(0.10)


def test_parse_args_exposes_risk_suppression_threshold(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_open_price_ledger_params.py",
            "--alpha-specs", "alpha=reports/example.jsonl",
            "--output-dir", "reports/example_sweep",
            "--state-aware-selection-mode", "risk_suppress",
            "--state-aware-selection-risk-delta-threshold", "0.15",
        ],
    )
    args = parse_args()
    assert args.state_aware_selection_mode == "risk_suppress"
    assert args.state_aware_selection_risk_delta_threshold == pytest.approx(0.15)


def test_sweep_key_distinguishes_selection_policies():
    common = ("alpha", "normal", 0.2, 0.0, None, 1.0, 0.0, 1.0, 500000, 5, None, 0.0, 0.006, 0.1)
    assert sweep_key(*common, selection_policy="retention") != sweep_key(
        *common, selection_policy="topk_dropout", top_k=30, n_drop=3
    )


def test_sweep_key_distinguishes_state_aware_selection():
    common = ("alpha", "normal", 0.2, 0.0, None, 1.0, 0.0, 1.0, 500000, 5, None, 0.0, 0.006, 0.1)
    assert sweep_key(*common) != sweep_key(
        *common,
        state_aware_selection_mode="risk_rank",
        state_aware_selection_pressure_col="global_defensive_pressure",
        state_aware_selection_rank_penalty=0.10,
    )


def test_sweep_key_distinguishes_risk_suppression_threshold():
    common = ("alpha", "normal", 0.2, 0.0, None, 1.0, 0.0, 1.0, 500000, 5, None, 0.0, 0.006, 0.1)
    assert sweep_key(*common, state_aware_selection_mode="risk_suppress", state_aware_selection_risk_delta_threshold=0.10) != sweep_key(
        *common,
        state_aware_selection_mode="risk_suppress",
        state_aware_selection_risk_delta_threshold=0.15,
    )


def test_parse_stress_names_preserves_requested_order():
    assert parse_stress_names("normal,lag1,cost2x") == [
        "normal",
        "lag1",
        "cost2x",
    ]


def test_parse_stress_names_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown stress"):
        parse_stress_names("normal,missing")


def test_apply_stress_overrides_does_not_mutate_base_args():
    base = SimpleNamespace(
        execution_lag=0,
        commission_rate=0.0001,
        stamp_tax_rate=0.0005,
        slippage_rate=0.0005,
        adv_participation_cap=0.05,
    )
    stressed = apply_stress_overrides(base, "lag1")

    assert stressed.execution_lag == 1
    assert base.execution_lag == 0


def test_retention_param_grid_deduplicates_unlimited_replacement():
    assert parse_int_list("0,1,3") == [0, 1, 3]
    grid = retention_param_grid([0, 1], [0.0, 0.08], [0.0, 0.002])

    assert grid[0] == (0, None, 0.0)
    assert grid.count((0, None, 0.0)) == 1
    assert (1, None, 0.002) in grid
    assert (1, 0.08, 0.0) in grid
    assert len(grid) == 5


def test_sweep_resume_key_includes_active_drawdown_throttle():
    base = sweep_key(
        "alpha",
        "normal",
        0.2,
        0.0,
        None,
        1.0,
        0.0,
        1.0,
        500000,
        5,
        None,
        0.0,
        0.006,
        0.10,
    )
    throttled = sweep_key(
        "alpha",
        "normal",
        0.2,
        0.0,
        None,
        1.0,
        0.0,
        1.0,
        500000,
        5,
        None,
        0.0,
        0.006,
        0.10,
        active_drawdown_throttle_lookback=15,
        active_drawdown_throttle_trigger=-0.03,
        active_drawdown_throttle_scale=0.95,
        active_drawdown_throttle_cooldown=5,
    )

    assert base != throttled


def test_completed_keys_from_summary_restores_active_drawdown_throttle():
    frame = pd.DataFrame(
        [
            {
                "alpha_name": "alpha",
                "stress": "normal",
                "rebalance_band": 0.2,
                "portfolio_value": 500000,
                "max_new_names": 5,
                "target_frac": 0.006,
                "hold_frac": 0.10,
                "active_drawdown_throttle_lookback": 15,
                "active_drawdown_throttle_trigger": -0.03,
                "active_drawdown_throttle_scale": 0.95,
                "active_drawdown_throttle_cooldown": 5,
            }
        ]
    )

    expected = sweep_key(
        "alpha",
        "normal",
        0.2,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        500000,
        5,
        0.0,
        0.0,
        0.006,
        0.10,
        active_drawdown_throttle_lookback=15,
        active_drawdown_throttle_trigger=-0.03,
        active_drawdown_throttle_scale=0.95,
        active_drawdown_throttle_cooldown=5,
    )

    assert completed_keys_from_summary(frame) == {expected}
