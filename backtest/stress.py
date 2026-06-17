"""Stress variants for named backtest presets."""

from dataclasses import dataclass

from backtest.presets import OFFICIAL_OPEN_PRICE_SHARE_LEDGER, OpenLedgerPreset


@dataclass(frozen=True)
class StressSpec:
    name: str
    description: str
    overrides: dict

    def apply(self, preset: OpenLedgerPreset):
        return preset.with_overrides(
            name=f"{preset.name}_{self.name}",
            **self.overrides,
        )


NORMAL = StressSpec("normal", "No stress override.", {})
LAG1 = StressSpec("lag1", "Execute one additional trading day later.", {"execution_lag": 1})
COST2X = StressSpec(
    "cost2x",
    "Double commission, stamp tax, and slippage assumptions.",
    {
        "commission_rate": OFFICIAL_OPEN_PRICE_SHARE_LEDGER.commission_rate * 2,
        "stamp_tax_rate": OFFICIAL_OPEN_PRICE_SHARE_LEDGER.stamp_tax_rate * 2,
        "slippage_rate": OFFICIAL_OPEN_PRICE_SHARE_LEDGER.slippage_rate * 2,
    },
)
CAPACITY_3PCT = StressSpec(
    "capacity_3pct",
    "Tighten ADV participation cap from 5% to 3%.",
    {"adv_participation_cap": 0.03},
)

STRESSES = {
    spec.name: spec
    for spec in (
        NORMAL,
        LAG1,
        COST2X,
        CAPACITY_3PCT,
    )
}


def get_stress(name):
    try:
        return STRESSES[name]
    except KeyError as exc:
        known = ", ".join(sorted(STRESSES))
        raise KeyError(f"Unknown stress {name!r}. Known stresses: {known}") from exc


def iter_stress_presets(preset, stress_names=("normal", "lag1", "cost2x", "capacity_3pct")):
    for name in stress_names:
        yield get_stress(name).apply(preset)
