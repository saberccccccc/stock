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
