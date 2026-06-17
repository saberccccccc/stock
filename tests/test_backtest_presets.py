import pytest

from backtest.presets import (
    OFFICIAL_OPEN_PRICE_SHARE_LEDGER,
    RESEARCH_OPEN_TO_OPEN_WIDE_BOOK,
    get_preset,
)
from backtest.stress import get_stress, iter_stress_presets


def test_official_open_ledger_preset_matches_frozen_plan():
    preset = OFFICIAL_OPEN_PRICE_SHARE_LEDGER

    assert preset.target_fracs == (0.006,)
    assert preset.hold_fracs == (0.10,)
    assert preset.portfolio_values == (500_000.0, 1_000_000.0)
    assert preset.market_timing_mode == "legacy"
    assert preset.min_adv_cny == 3_000_000.0
    assert preset.adv_participation_cap == 0.05
    assert preset.limit_threshold == 0.095
    assert preset.rebalance_band == 0.20
    assert preset.max_new_names == 5


def test_research_open_to_open_wide_book_is_separate_family():
    preset = RESEARCH_OPEN_TO_OPEN_WIDE_BOOK

    assert preset.target_fracs == (0.03, 0.035)
    assert preset.hold_fracs == (0.50, 0.60)
    assert preset.portfolio_values == (1_000_000.0,)
    assert preset.min_adv_cny == 20_000_000.0


def test_cli_args_are_legacy_script_compatible_strings():
    args = OFFICIAL_OPEN_PRICE_SHARE_LEDGER.cli_args()

    assert args["target_fracs"] == "0.006"
    assert args["hold_fracs"] == "0.1"
    assert args["portfolio_values"] == "500000,1000000"
    assert args["market_timing_mode"] == "legacy"


def test_stress_specs_apply_expected_overrides():
    base = OFFICIAL_OPEN_PRICE_SHARE_LEDGER

    lag1 = get_stress("lag1").apply(base)
    cost2x = get_stress("cost2x").apply(base)
    capacity = get_stress("capacity_3pct").apply(base)

    assert lag1.execution_lag == 1
    assert cost2x.commission_rate == base.commission_rate * 2
    assert cost2x.stamp_tax_rate == base.stamp_tax_rate * 2
    assert cost2x.slippage_rate == base.slippage_rate * 2
    assert capacity.adv_participation_cap == 0.03


def test_iter_stress_presets_keeps_normal_first():
    presets = list(iter_stress_presets(OFFICIAL_OPEN_PRICE_SHARE_LEDGER))

    assert [preset.name for preset in presets] == [
        "official_open_price_share_ledger_normal",
        "official_open_price_share_ledger_lag1",
        "official_open_price_share_ledger_cost2x",
        "official_open_price_share_ledger_capacity_3pct",
    ]


def test_unknown_preset_and_stress_raise_clear_errors():
    with pytest.raises(KeyError, match="Unknown preset"):
        get_preset("missing")
    with pytest.raises(KeyError, match="Unknown stress"):
        get_stress("missing")
