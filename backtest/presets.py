"""Named backtest parameter presets.

These presets document the execution families used in reports.  They are not
applied implicitly by legacy CLI entrypoints; callers must opt in.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class OpenLedgerPreset:
    name: str
    target_fracs: tuple[float, ...] = (0.006,)
    hold_fracs: tuple[float, ...] = (0.10,)
    portfolio_values: tuple[float, ...] = (500_000.0, 1_000_000.0)
    max_weight: float = 0.05
    market_timing_mode: str = "legacy"
    market_min_mult: float = 0.20
    market_max_mult: float = 1.00
    legacy_bear_mult: float = 0.70
    legacy_crash_mult: float = 0.30
    commission_rate: float = 0.0001
    stamp_tax_rate: float = 0.0005
    slippage_rate: float = 0.0005
    adv_window: int = 20
    adv_participation_cap: float = 0.05
    min_adv_cny: float = 3_000_000.0
    money_scale: float = 1000.0
    limit_threshold: float = 0.095
    lot_size: int = 100
    min_commission_cny: float = 5.0
    rebalance_band: float = 0.20
    max_new_names: int = 5
    execution_lag: int = 0
    notes: str = ""

    def with_overrides(self, name=None, **overrides):
        return replace(self, name=name or self.name, **overrides)

    def cli_args(self):
        """Return CLI-style arguments for explicit wrapper use."""
        return {
            "target_fracs": ",".join(f"{value:g}" for value in self.target_fracs),
            "hold_fracs": ",".join(f"{value:g}" for value in self.hold_fracs),
            "portfolio_values": ",".join(f"{int(value):d}" for value in self.portfolio_values),
            "max_weight": self.max_weight,
            "market_timing_mode": self.market_timing_mode,
            "market_min_mult": self.market_min_mult,
            "market_max_mult": self.market_max_mult,
            "legacy_bear_mult": self.legacy_bear_mult,
            "legacy_crash_mult": self.legacy_crash_mult,
            "commission_rate": self.commission_rate,
            "stamp_tax_rate": self.stamp_tax_rate,
            "slippage_rate": self.slippage_rate,
            "adv_window": self.adv_window,
            "adv_participation_cap": self.adv_participation_cap,
            "min_adv_cny": self.min_adv_cny,
            "money_scale": self.money_scale,
            "limit_threshold": self.limit_threshold,
            "lot_size": self.lot_size,
            "min_commission_cny": self.min_commission_cny,
            "rebalance_band": self.rebalance_band,
            "max_new_names": self.max_new_names,
            "execution_lag": self.execution_lag,
        }


OFFICIAL_OPEN_PRICE_SHARE_LEDGER = OpenLedgerPreset(
    name="official_open_price_share_ledger",
    notes="V9 avgw3 + maxret095 + open-price share-ledger official baseline.",
)

LEGACY_CLOSE_BASED_TOP30 = OpenLedgerPreset(
    name="legacy_close_based_top30",
    max_new_names=0,
    notes="Continuity family for older close/constrained Top30 diagnostics.",
)

RESEARCH_OPEN_TO_OPEN_WIDE_BOOK = OpenLedgerPreset(
    name="research_open_to_open_wide_book",
    target_fracs=(0.03, 0.035),
    hold_fracs=(0.50, 0.60),
    portfolio_values=(1_000_000.0,),
    min_adv_cny=20_000_000.0,
    max_new_names=0,
    notes="Old V9 wide-book open-to-open research framework; keep separate from official ledger.",
)

PRESETS = {
    preset.name: preset
    for preset in (
        OFFICIAL_OPEN_PRICE_SHARE_LEDGER,
        LEGACY_CLOSE_BASED_TOP30,
        RESEARCH_OPEN_TO_OPEN_WIDE_BOOK,
    )
}


def option_to_dest(option):
    return option.lstrip("-").replace("-", "_")


def explicit_cli_dests(argv):
    """Return argparse-style dest names explicitly present in argv."""
    dests = set()
    for token in argv:
        if not str(token).startswith("--"):
            continue
        option = str(token).split("=", 1)[0]
        dests.add(option_to_dest(option))
    return dests


def apply_preset_to_namespace(namespace, preset, explicit_dests=()):
    """Apply preset values to an argparse namespace without overriding explicit CLI flags."""
    explicit = set(explicit_dests)
    for key, value in preset.cli_args().items():
        if key in explicit:
            continue
        setattr(namespace, key, value)
    setattr(namespace, "applied_preset", preset.name)
    return namespace


def get_preset(name):
    try:
        return PRESETS[name]
    except KeyError as exc:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown preset {name!r}. Known presets: {known}") from exc
