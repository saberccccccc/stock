import json

import numpy as np

from backtest.open_ledger import (
    build_desired_target,
    limit_new_names,
    load_alpha_rows,
    parse_float_list,
    weights_from_selected,
)
from run.backtest_retention_open_ledger import parse_args


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


def test_weights_from_selected_respects_max_weight_and_gross():
    weights = weights_from_selected(
        selected=["A", "B", "MISSING"],
        code2idx={"A": 0, "B": 1, "C": 2},
        n_codes=3,
        gross_weight=0.7,
        max_weight=0.2,
    )

    assert np.allclose(weights, np.array([0.35, 0.35, 0.0]))


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

    assert limited == ["OLD1", "N1", "N2"]


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
