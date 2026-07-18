import json

import numpy as np

from alpha.io import write_alpha_rows
from backtest.open_ledger import (
    build_desired_target,
    limit_new_names,
    load_alpha_rows,
    parse_float_list,
    weights_from_selected,
)
from run.backtest_retention_open_ledger import (
    collect_alpha_inputs,
    filter_alpha_rows_by_date,
    parse_args,
)


def _required_args():
    return [
        "--alpha-jsonl",
        "alpha.jsonl",
        "--output-dir",
        "out",
    ]


def test_open_ledger_legacy_defaults_are_unchanged_without_preset():
    args = parse_args(_required_args())

    assert args.preset is None
    assert args.stress is None
    assert args.target_fracs == "0.006"
    assert args.hold_fracs == "0.10"
    assert args.max_new_names == 0
    assert not hasattr(args, "applied_preset")


def test_open_ledger_parser_accepts_alpha_manifest():
    args = parse_args([
        "--alpha-manifest",
        "manifest.json",
        "--output-dir",
        "out",
    ])

    assert args.alpha_manifest == "manifest.json"
    assert args.alpha_jsonl is None


def test_open_ledger_official_preset_is_explicit_opt_in():
    args = parse_args(_required_args() + ["--preset", "official_open_price_share_ledger"])

    assert args.applied_preset == "official_open_price_share_ledger"
    assert args.target_fracs == "0.006"
    assert args.hold_fracs == "0.1"
    assert args.portfolio_values == "500000,1000000"
    assert args.max_new_names == 5
    assert args.min_adv_cny == 3_000_000.0


def test_open_ledger_preset_keeps_explicit_overrides():
    args = parse_args(
        _required_args()
        + [
            "--preset",
            "official_open_price_share_ledger",
            "--max-new-names",
            "3",
            "--min-adv-cny",
            "20000000",
        ]
    )

    assert args.applied_preset == "official_open_price_share_ledger"
    assert args.max_new_names == 3
    assert args.min_adv_cny == 20_000_000.0
    assert args.rebalance_band == 0.20


def test_open_ledger_preset_keeps_industry_budget_override():
    args = parse_args(
        _required_args()
        + [
            "--preset",
            "official_open_price_share_ledger",
            "--industry-csv",
            "data/stock_industry.csv",
            "--max-industry-weight",
            "0.25",
        ]
    )

    assert args.applied_preset == "official_open_price_share_ledger"
    assert args.industry_csv == "data/stock_industry.csv"
    assert args.max_industry_weight == 0.25


def test_open_ledger_stress_without_preset_uses_official_base():
    args = parse_args(_required_args() + ["--stress", "cost2x"])

    assert args.applied_preset == "official_open_price_share_ledger_cost2x"
    assert args.max_new_names == 5
    assert args.commission_rate == 0.0002
    assert args.stamp_tax_rate == 0.001
    assert args.slippage_rate == 0.001


def test_open_ledger_lag1_stress_keeps_explicit_execution_lag_override():
    args = parse_args(
        _required_args()
        + [
            "--preset",
            "official_open_price_share_ledger",
            "--stress",
            "lag1",
            "--execution-lag",
            "2",
        ]
    )

    assert args.applied_preset == "official_open_price_share_ledger_lag1"
    assert args.execution_lag == 2


def test_open_ledger_date_filter_keeps_inclusive_range():
    rows = [
        {"date": "2025-12-31", "codes": ["A"], "alpha": [1.0]},
        {"date": "2026-01-05", "codes": ["A"], "alpha": [1.0]},
        {"date": "2026-05-19", "codes": ["A"], "alpha": [1.0]},
    ]

    filtered = filter_alpha_rows_by_date(rows, "2026-01-01", "2026-05-18")

    assert [row["date"] for row in filtered] == ["2026-01-05"]


def test_collect_alpha_inputs_resolves_manifest_per_portfolio_value(tmp_path):
    small_alpha = tmp_path / "small.jsonl"
    large_alpha = tmp_path / "large.jsonl"
    write_alpha_rows(small_alpha, [{"date": "2025-01-02", "codes": ["A"], "alpha": [1.0]}])
    write_alpha_rows(large_alpha, [{"date": "2025-01-03", "codes": ["B"], "alpha": [1.0]}])
    manifest_path = tmp_path / "alpha_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "default_alpha_jsonl": "small.jsonl",
                "rules": [
                    {
                        "name": "large_only",
                        "min_portfolio_value": 1000000,
                        "alpha_jsonl": "large.jsonl",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--alpha-manifest",
            str(manifest_path),
            "--output-dir",
            "out",
            "--portfolio-values",
            "500000,1000000",
        ]
    )
    portfolio_values, resolved_sources, alpha_rows_by_path, alpha_stats_by_path = collect_alpha_inputs(args)

    assert portfolio_values == [500000.0, 1000000.0]
    assert resolved_sources[500000.0].rule_name == "default"
    assert resolved_sources[1000000.0].rule_name == "large_only"
    assert len(alpha_rows_by_path) == 2
    assert alpha_stats_by_path[resolved_sources[500000.0].alpha_jsonl]["first_signal"] == "2025-01-02"
    assert alpha_stats_by_path[resolved_sources[1000000.0].alpha_jsonl]["first_signal"] == "2025-01-03"


def test_open_ledger_parser_has_no_protocol_check_knob():
    args = parse_args(_required_args())

    assert not hasattr(args, "protocol_check")


def test_open_ledger_parser_accepts_arbitrary_range_and_cache_knobs():
    args = parse_args(
        _required_args()
        + [
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-06-18",
            "--max-data-date",
            "2026-06-18",
            "--ohlc-cache-dir",
            "cache/custom_ohlc",
            "--ohlc-matrix-cache-dir",
            "cache/custom_matrix",
            "--load-lookback-days",
            "120",
        ]
    )

    assert args.start_date == "2026-01-01"
    assert args.end_date == "2026-06-18"
    assert args.max_data_date == "2026-06-18"
    assert args.ohlc_cache_dir == "cache/custom_ohlc"
    assert args.ohlc_matrix_cache_dir == "cache/custom_matrix"
    assert args.load_lookback_days == 120
    assert args.no_ohlc_cache is False
    assert args.no_ohlc_matrix_cache is False
    assert args.rebuild_ohlc_matrix_cache is False
    assert args.allow_forward is False
    assert args.execution_constraint_mode == "proxy"
    assert args.block_intraday_limit_touch is False
    assert args.limit_price_tolerance == 1e-4
    assert args.min_buy_listing_days == 60
    assert args.no_limit_first_trading_days == 5


def test_open_ledger_parser_accepts_conditional_active_throttle():
    args = parse_args(
        _required_args()
        + [
            "--active-drawdown-throttle-lookback",
            "15",
            "--active-drawdown-throttle-trigger",
            "-0.03",
            "--active-drawdown-throttle-scale",
            "0.85",
            "--active-drawdown-throttle-cooldown",
            "5",
            "--active-drawdown-throttle-condition",
            "crowding_momentum",
            "--active-drawdown-throttle-min-top-industry-weight",
            "0.40",
            "--active-drawdown-throttle-min-industry-hhi",
            "0.25",
            "--active-drawdown-throttle-max-momentum20",
            "0.0",
            "--active-drawdown-throttle-min-volatility60",
            "0.45",
        ]
    )

    assert args.active_drawdown_throttle_condition == "crowding_momentum"
    assert args.active_drawdown_throttle_scale == 0.85
    assert args.active_drawdown_throttle_min_top_industry_weight == 0.40
    assert args.active_drawdown_throttle_min_industry_hhi == 0.25
    assert args.active_drawdown_throttle_max_momentum20 == 0.0
    assert args.active_drawdown_throttle_min_volatility60 == 0.45


def test_open_ledger_parser_accepts_continuous_risk_overlays():
    args = parse_args(
        _required_args()
        + [
            "--active-drawdown-throttle-mode",
            "continuous",
            "--active-drawdown-throttle-continuous-width",
            "0.06",
            "--global-risk-overlay-mode",
            "defensive_pressure_continuous",
            "--global-risk-pressure-width",
            "0.05",
        ]
    )

    assert args.active_drawdown_throttle_mode == "continuous"
    assert args.active_drawdown_throttle_continuous_width == 0.06
    assert args.global_risk_overlay_mode == "defensive_pressure_continuous"
    assert args.global_risk_pressure_width == 0.05


def test_open_ledger_parser_accepts_state_aware_selection():
    args = parse_args(
        _required_args()
        + [
            "--state-aware-selection-mode",
            "risk_rank",
            "--state-aware-selection-pressure-col",
            "global_us_hk_pressure",
            "--state-aware-selection-pressure-threshold",
            "0.035",
            "--state-aware-selection-pressure-width",
            "0.055",
            "--state-aware-selection-rank-penalty",
            "0.05",
            "--state-aware-selection-momentum-weight",
            "0.5",
        ]
    )

    assert args.state_aware_selection_mode == "risk_rank"
    assert args.state_aware_selection_pressure_col == "global_us_hk_pressure"
    assert args.state_aware_selection_pressure_threshold == 0.035
    assert args.state_aware_selection_pressure_width == 0.055
    assert args.state_aware_selection_rank_penalty == 0.05
    assert args.state_aware_selection_momentum_weight == 0.5


def test_open_ledger_alpha_loader_accepts_utf8_bom(tmp_path):
    path = tmp_path / "alpha.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf"
        + json.dumps(
            {
                "date": "2025-01-02",
                "codes": ["000001.SZ"],
                "alpha": [1.0],
                "n_stocks": 1,
            }
        ).encode("utf-8")
    )

    rows = load_alpha_rows(path)

    assert len(rows) == 1
    assert rows[0]["date"].strftime("%Y-%m-%d") == "2025-01-02"
    assert rows[0]["codes"] == ["000001.SZ"]


def test_parse_float_list_ignores_empty_items():
    assert parse_float_list("0.006, 0.01,,") == [0.006, 0.01]


def test_build_desired_target_keeps_current_names_inside_hold_bucket():
    row = {"codes": ["A", "B", "C", "D", "E"]}

    selected, kept, target_n, hold_n, rank_map = build_desired_target(
        row,
        current_codes=["D", "B", "X"],
        target_frac=0.4,
        hold_frac=0.8,
    )

    assert target_n == 2
    assert hold_n == 4
    assert kept == ["D", "B"]
    assert selected == ["D", "B"]
    assert rank_map["A"] == 0


def test_build_desired_target_fills_from_alpha_order():
    row = {"codes": ["A", "B", "C", "D", "E"]}

    selected, kept, target_n, _, _ = build_desired_target(
        row,
        current_codes=["E"],
        target_frac=0.4,
        hold_frac=0.4,
    )

    assert target_n == 2
    assert kept == []
    assert selected == ["A", "B"]


def test_weights_from_selected_treats_max_weight_as_hard_cap():
    weights = weights_from_selected(
        selected=["A", "B", "MISSING"],
        code2idx={"A": 0, "B": 1, "C": 2},
        n_codes=3,
        gross_weight=0.7,
        max_weight=0.2,
    )

    assert np.allclose(weights, np.array([0.2, 0.2, 0.0]))
    assert weights.sum() < 0.7


def test_weights_from_selected_reaches_gross_when_cap_allows():
    weights = weights_from_selected(
        selected=["A", "B", "C"],
        code2idx={"A": 0, "B": 1, "C": 2},
        n_codes=3,
        gross_weight=0.6,
        max_weight=0.5,
    )

    assert np.allclose(weights, np.array([0.2, 0.2, 0.2]))


def test_weights_from_selected_returns_zero_for_empty_selection():
    weights = weights_from_selected([], {"A": 0}, 1, 0.7, 0.2)

    assert np.array_equal(weights, np.zeros(1))


def test_limit_new_names_noops_without_current_positions():
    selected = ["A", "B", "C"]
    row = {"codes": ["A", "B", "C", "D"]}

    assert limit_new_names(selected, [], row, 1, [], 3) is selected


def test_limit_new_names_retains_old_names_and_caps_new_entries():
    row = {"codes": ["N1", "OLD1", "N2", "OLD2", "N3", "OLD3"]}
    selected = ["N1", "OLD1", "N2", "OLD2"]

    limited = limit_new_names(
        selected=selected,
        kept=["OLD1", "OLD2"],
        row=row,
        max_new_names=1,
        current_selected=["OLD1", "OLD2", "OLD3"],
        target_n=4,
    )

    assert limited == ["OLD1", "OLD2", "OLD3", "N1"]


def test_limit_new_names_does_not_force_replacement_when_desired_is_unchanged():
    current = ["OLD1", "OLD2", "OLD3"]
    row = {"codes": ["OLD1", "OLD2", "OLD3", "N1"]}

    limited = limit_new_names(
        selected=current.copy(),
        kept=current.copy(),
        row=row,
        max_new_names=2,
        current_selected=current,
        target_n=3,
        mode="at_most",
    )

    assert limited == current


def test_limit_new_names_uses_exit_hold_fraction():
    row = {"codes": ["N1", "OLD1", "N2", "OLD2", "OLD3", "N3"]}

    limited = limit_new_names(
        selected=["N1", "OLD1", "N2"],
        kept=["OLD1"],
        row=row,
        max_new_names=1,
        current_selected=["OLD1", "OLD2", "OLD3"],
        target_n=3,
        exit_hold_frac=0.5,
    )

    assert limited == ["OLD1", "N1"]
    assert len(set(limited) - {"OLD1", "OLD2", "OLD3"}) == 1


def test_limit_new_names_switch_gap_prevents_weak_replacement():
    row = {"codes": ["N1", "OLD1", "N2", "OLD2", "N3", "OLD3"]}

    limited = limit_new_names(
        selected=["N1", "OLD1", "N2"],
        kept=["OLD1"],
        row=row,
        max_new_names=1,
        current_selected=["OLD1", "OLD2", "OLD3"],
        target_n=3,
        switch_gap_frac=1.0,
    )

    assert limited == ["OLD1", "OLD2", "OLD3"]


def test_limit_new_names_handles_too_few_exit_eligible_old_names():
    row = {"codes": ["N1", "OLD1", "N2", "OLD2", "OLD3", "N3"]}

    limited = limit_new_names(
        selected=["N1", "OLD1", "N2"],
        kept=["OLD1"],
        row=row,
        max_new_names=2,
        current_selected=["OLD1", "OLD2", "OLD3"],
        target_n=3,
        exit_hold_frac=0.5,
        switch_gap_frac=0.1,
        mode="at_most",
    )

    assert limited == ["OLD1", "N1", "N2"]
