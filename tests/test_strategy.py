import pytest

from backtest.strategy import retention_target, select_target_policy, topk_dropout_target


def test_retention_target_keeps_eligible_current_names_then_refills_by_rank():
    proposal = retention_target(
        ["A", "B", "C", "D", "E"], ["C", "D"], target_frac=0.4, hold_frac=0.8
    )
    assert proposal.target_n == 2
    assert proposal.retained == ("C", "D")
    assert proposal.selected == ("C", "D")
    assert proposal.rationale == "rank_retention"


def test_topk_dropout_limits_new_names_without_execution_logic():
    proposal = topk_dropout_target(["A", "B", "C", "D"], ["C", "D"], top_k=2, n_drop=1)
    assert proposal.selected == ("C", "A")
    assert proposal.retained == ("C",)
    assert proposal.rationale == "topk_dropout"


def test_policy_dispatcher_derives_topk_from_target_fraction():
    proposal = select_target_policy(
        ["A", "B", "C", "D"], ["D"], policy="topk_dropout", target_frac=0.5, hold_frac=0.8, n_drop=1
    )
    assert proposal.selected == ("D", "A")
    assert proposal.target_n == 2


def test_topk_dropout_can_open_initial_positions_when_drop_is_zero():
    proposal = topk_dropout_target(["A", "B", "C"], [], top_k=2, n_drop=0)
    assert proposal.selected == ("A", "B")
    assert proposal.rationale == "topk_dropout_initial_build"


def test_policy_dispatcher_rejects_unknown_policy():
    with pytest.raises(ValueError, match="unknown selection policy"):
        select_target_policy([], [], policy="unknown", target_frac=0.1, hold_frac=0.1)
