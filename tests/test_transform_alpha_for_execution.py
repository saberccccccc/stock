import pandas as pd

from run.transform_alpha_for_execution import transform_rows


def _row(date, codes):
    return {"date": pd.Timestamp(date), "codes": codes, "alpha": [], "n_stocks": len(codes)}


def test_high_signal_return_is_demoted():
    rows = [_row("2024-01-02", ["A", "B", "C"])]
    returns = {pd.Timestamp("2024-01-02"): {"A": 0.10, "B": 0.01, "C": -0.01}}
    output = transform_rows(rows, returns, max_signal_return=0.095)
    assert output[0]["codes"][-1] == "A"
    assert output[0]["execution_transform"]["demoted_count"] == 1


def test_stalled_stock_is_demoted_and_overlap_counted_once():
    rows = [_row("2024-01-02", ["A", "B", "C"])]
    date = pd.Timestamp("2024-01-02")
    returns = {date: {"A": 0.10}}
    stalls = {date: {"A", "B"}}
    output = transform_rows(
        rows,
        returns,
        max_signal_return=0.095,
        stall_signals=stalls,
        stall_config={"surge_return": 0.10},
    )

    assert output[0]["codes"] == ["C", "A", "B"]
    transform = output[0]["execution_transform"]
    assert transform["demoted_count"] == 2
    assert transform["chase_demoted_count"] == 1
    assert transform["stall_demoted_count"] == 2


def test_stability_uses_only_prior_rows():
    rows = [
        _row("2024-01-02", ["A", "B", "C"]),
        _row("2024-01-03", ["C", "B", "A"]),
        _row("2024-01-04", ["C", "B", "A"]),
    ]
    output = transform_rows(rows, {}, stability_window=2, current_weight=0.5)
    assert output[0]["codes"] == ["A", "B", "C"]
    assert output[1]["codes"] == ["A", "B", "C"]
    assert output[2]["codes"][0] == "C"


def test_ties_are_deterministic():
    rows = [_row("2024-01-02", ["B", "A"])]
    output = transform_rows(rows, {}, stability_window=0, current_weight=1.0)
    assert output[0]["codes"] == ["B", "A"]
