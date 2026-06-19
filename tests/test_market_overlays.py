from types import SimpleNamespace

import pandas as pd
import pytest

from alpha.market_overlays import (
    attach_breadth_market_multiplier,
    compute_breadth,
    rolling_breadth_map,
    shrink_target_row,
)
from run.make_breadth_triggered_target_alpha import transform_row


def test_shrink_target_row_demotes_names_between_risk_and_base_bucket():
    row = {"date": "2024-01-02", "codes": list("ABCDEFGHIJ"), "alpha": list(range(10))}

    output, details = shrink_target_row(row, True, 0.4, 0.2)

    assert output["codes"] == list("ABEFGHIJCD")
    assert details["base_n"] == 4
    assert details["risk_n"] == 2
    assert details["effective_target_frac"] == 0.2


def test_breadth_target_wrapper_preserves_metadata_contract():
    args = SimpleNamespace(
        base_target_frac=0.4,
        risk_target_frac=0.2,
        breadth_window=3,
        breadth_below=0.35,
    )
    row = {"codes": list("ABCDEFGHIJ"), "alpha": list(range(10))}

    output = transform_row(row, True, args, 0.3)

    assert output["breadth_target_transform"]["triggered"] is True
    assert output["breadth_target_transform"]["breadth_value"] == 0.3


def test_compute_and_roll_breadth(tmp_path):
    for code, closes in {"A.SZ": [10, 11], "B.SH": [10, 9]}.items():
        pd.DataFrame(
            {"trade_date": ["2024-01-01", "2024-01-02"], "close": closes}
        ).to_csv(tmp_path / f"{code}.csv", index=False)

    breadth = compute_breadth(tmp_path, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
    values = rolling_breadth_map(breadth, 1)

    assert values[pd.Timestamp("2024-01-02")] == pytest.approx(0.5)


def test_attach_market_multiplier_keeps_ranking():
    row = {"date": "2024-01-02", "codes": ["A", "B"], "alpha": [1.0, 0.0]}

    output = attach_breadth_market_multiplier(row, True, 3, 0.35, 0.3, 0.85)

    assert output["codes"] == row["codes"]
    assert output["breadth_market_transform"]["effective_market_mult"] == 0.85
