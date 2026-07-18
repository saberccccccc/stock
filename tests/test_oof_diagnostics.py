import json

from run.audit_oof_lineage import _rank_corr, _top_overlap


def test_rank_correlation_and_top_overlap_are_cross_sectional():
    left = {"codes": ["a", "b", "c", "d"]}
    right = {"codes": ["a", "b", "d", "c"]}

    assert _rank_corr(left, right) > 0.79
    assert _top_overlap(left, right, 2) == 1.0
