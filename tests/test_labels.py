import numpy as np
import pytest

from data.labels import build_forward_return_labels, label_end_offset


def test_forward_return_label_formulas_and_lag1_identity():
    opens = np.asarray([10, 11, 12, 13, 14], dtype=np.float32)
    closes = np.asarray([10.5, 11.5, 12.5, 13.5, 14.5], dtype=np.float32)

    labels = build_forward_return_labels(opens, closes, max_horizon=2)

    assert labels["cc"][0, 0] == pytest.approx(11.5 / 10.5 - 1)
    assert labels["oc"][0, 0] == pytest.approx(11.5 / 11.0 - 1)
    assert labels["oo"][0, 0] == pytest.approx(12.0 / 11.0 - 1)
    assert labels["oo"][1, 1] == pytest.approx(14.0 / 12.0 - 1)
    assert np.isnan(labels["oo"][-2:, 1]).all()

    # The lag-1 OO family is an exact date-shifted view, not a fourth file.
    assert labels["oo"][1, 0] == pytest.approx(13.0 / 12.0 - 1)


def test_missing_execution_price_does_not_skip_to_a_later_date():
    opens = np.asarray([10, np.nan, 12, 13], dtype=np.float32)
    closes = np.asarray([10.5, np.nan, 12.5, 13.5], dtype=np.float32)

    labels = build_forward_return_labels(opens, closes, max_horizon=1)

    assert np.isnan(labels["oc"][0, 0])
    assert np.isnan(labels["oo"][0, 0])
    assert np.isnan(labels["cc"][0, 0])
    assert labels["oo"][1, 0] == pytest.approx(13.0 / 12.0 - 1)


@pytest.mark.parametrize(
    ("family", "horizon_index", "expected"),
    [("cc", 6, 7), ("oc", 6, 7), ("oo", 6, 8), ("oo_lag1", 6, 9)],
)
def test_label_end_offset(family, horizon_index, expected):
    assert label_end_offset(family, horizon_index) == expected
