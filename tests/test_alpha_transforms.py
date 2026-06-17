from alpha.transforms import (
    conditional_negative_filter_row,
    edge_rerank_row,
    negative_filter_row,
)


def _row(date, codes, **extra):
    row = {
        "date": date,
        "codes": codes,
        "alpha": list(range(len(codes), 0, -1)),
        "n_stocks": len(codes),
    }
    row.update(extra)
    return row


def test_edge_rerank_only_reorders_requested_window():
    base = _row("2024-01-02", ["A", "B", "C", "D", "E"])
    rerank = _row("2024-01-02", ["E", "D", "C", "B", "A"])

    out, changed = edge_rerank_row(base, rerank, 1, 4, "full.jsonl")

    assert changed is True
    assert out["codes"] == ["A", "D", "C", "B", "E"]
    assert out["edge_reranker"]["start_rank"] == 1
    assert out["edge_reranker"]["end_rank"] == 4


def test_negative_filter_moves_worst_window_names_to_tail():
    base = _row("2024-01-02", ["A", "B", "C", "D", "E"])
    rerank = _row("2024-01-02", ["A", "E", "D", "C", "B"])

    out, dropped = negative_filter_row(base, rerank, 1, 4, 2, "full.jsonl")

    assert dropped == 2
    assert out["codes"] == ["A", "D", "E", "B", "C"]
    assert out["negative_filter"]["drop_n"] == 2


def test_negative_filter_noop_when_drop_n_zero():
    base = _row("2024-01-02", ["A", "B", "C"])
    rerank = _row("2024-01-02", ["C", "B", "A"])

    out, dropped = negative_filter_row(base, rerank, 0, 2, 0, "full.jsonl")

    assert dropped == 0
    assert out is base


def test_conditional_negative_filter_skips_when_trigger_false():
    base = _row(
        "2024-01-02",
        ["A", "B", "C"],
        breadth_market_transform={"triggered": False},
    )
    rerank = _row("2024-01-02", ["C", "B", "A"])

    out, triggered, dropped = conditional_negative_filter_row(
        base,
        rerank,
        "breadth_market_transform.triggered",
        0,
        2,
        1,
        "full.jsonl",
    )

    assert triggered is False
    assert dropped == 0
    assert out["codes"] == ["A", "B", "C"]


def test_conditional_negative_filter_applies_when_trigger_true():
    base = _row(
        "2024-01-02",
        ["A", "B", "C", "D"],
        breadth_market_transform={"triggered": True},
    )
    rerank = _row("2024-01-02", ["D", "C", "B", "A"])

    out, triggered, dropped = conditional_negative_filter_row(
        base,
        rerank,
        "breadth_market_transform.triggered",
        0,
        3,
        1,
        "full.jsonl",
    )

    assert triggered is True
    assert dropped == 1
    assert out["codes"] == ["B", "C", "D", "A"]
    assert "conditional_negative_filter" in out
    assert "negative_filter" not in out
