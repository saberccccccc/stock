import numpy as np

from run.build_reranker_dataset import (
    percentile_rank_desc,
    relevance_from_rank,
)


def test_percentile_rank_desc_orders_largest_first():
    values = np.asarray([2.0, 5.0, 1.0, 3.0])
    ranks = percentile_rank_desc(values)
    assert ranks[1] == 0.0
    assert ranks[2] == 1.0


def test_relevance_cutoffs_are_graded():
    rank_pct = np.asarray([0.0, 0.10, 0.20, 0.40, 0.80])
    relevance = relevance_from_rank(rank_pct)
    assert relevance.tolist() == [4, 3, 2, 1, 0]
