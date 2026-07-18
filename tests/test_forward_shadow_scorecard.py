import pandas as pd

from run.validate_forward_shadow_scorecard import validate_forward_frame


REQUIRED_FIELDS = [
    "signal_start",
    "signal_end",
    "backtest_start",
    "backtest_end",
    "model_alpha_contribution",
    "execution_rejection_contribution",
    "cost_contribution",
    "portfolio_constraint_contribution",
]


def _valid_frame():
    return pd.DataFrame(
        [
            {
                "signal_start": "2025-12-31",
                "signal_end": "2026-06-30",
                "backtest_start": "2026-01-02",
                "backtest_end": "2026-06-30",
                "model_alpha_contribution": 0.01,
                "execution_rejection_contribution": 0.0,
                "cost_contribution": -0.001,
                "portfolio_constraint_contribution": -0.002,
                "split": "forward_2026",
                "is_forward": True,
                "selection_eligible": False,
                "date": "2026-01-02",
            }
        ]
    )


def test_forward_scorecard_accepts_t_close_signal_and_t1_open_execution():
    checks = validate_forward_frame(
        _valid_frame(),
        path="forward.csv",
        required_fields=REQUIRED_FIELDS,
        research_end="2025-12-31",
        forward_start="2026-01-01",
    )

    assert all(check["status"] == "complete" for check in checks)


def test_forward_scorecard_rejects_pre_boundary_execution():
    frame = _valid_frame()
    frame.loc[0, "backtest_start"] = "2025-12-31"

    checks = validate_forward_frame(
        frame,
        path="forward.csv",
        required_fields=REQUIRED_FIELDS,
        research_end="2025-12-31",
        forward_start="2026-01-01",
    )

    boundary = next(check for check in checks if check["name"].startswith("forward_date_boundary"))
    assert boundary["status"] == "error"


def test_forward_scorecard_requires_attribution_fields():
    frame = _valid_frame().drop(columns=["cost_contribution"])

    checks = validate_forward_frame(
        frame,
        path="forward.csv",
        required_fields=REQUIRED_FIELDS,
        research_end="2025-12-31",
        forward_start="2026-01-01",
    )

    assert checks[0]["status"] == "error"
    assert "cost_contribution" in checks[0]["detail"]


def test_forward_scorecard_rejects_selection_eligible_forward_row():
    frame = _valid_frame()
    frame.loc[0, "selection_eligible"] = True

    checks = validate_forward_frame(
        frame,
        path="forward.csv",
        required_fields=REQUIRED_FIELDS,
        research_end="2025-12-31",
        forward_start="2026-01-01",
    )

    role = next(check for check in checks if check["name"].startswith("forward_selection_role"))
    assert role["status"] == "error"
