import importlib
from pathlib import Path

import pytest

from backtest.market_state import load_index_states


OVERLAY_MODULES = (
    "run.make_breadth_triggered_market_alpha",
    "run.make_breadth_triggered_target_alpha",
    "run.make_state_triggered_target_alpha",
    "run.switch_alpha_by_market_state",
)


@pytest.mark.parametrize("module_name", OVERLAY_MODULES)
def test_market_overlay_imports_without_changing_cwd(module_name):
    cwd = Path.cwd()
    importlib.import_module(module_name)
    assert Path.cwd() == cwd


def test_market_state_accepts_short_positive_window(tmp_path):
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "date,close\n2024-01-01,100\n2024-01-02,99\n",
        encoding="utf-8",
    )

    states = load_index_states(index_path, ma_window=2)

    assert list(states["market_state"]) == ["normal", "bear"]


def test_market_state_rejects_nonpositive_window(tmp_path):
    index_path = tmp_path / "index.csv"
    index_path.write_text("date,close\n2024-01-01,100\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive"):
        load_index_states(index_path, ma_window=0)
