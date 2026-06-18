import json

import pandas as pd

from experiments.checkpoint_selection import (
    SelectionRule,
    load_epoch_metrics,
    score_checkpoints,
    select_checkpoints,
)


def _write_metrics(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _record(epoch, alpha, raw_topret, raw_topstable):
    return {
        "epoch": epoch,
        "train_loss": -0.1 * epoch,
        "learning_rate": 0.0001,
        "selection_metric": "alpha",
        "selection_score": alpha,
        "train_components": {"global_ic": -alpha},
        "val_metrics": {
            "alpha": alpha,
            "rawtopret_h5_top0p6": raw_topret,
            "rawtopstable_h5_top0p6": raw_topstable,
        },
        "checkpoint": f"epoch_{epoch:03d}.pt",
    }


def test_load_epoch_metrics_flattens_records(tmp_path):
    metrics = tmp_path / "epoch_metrics.jsonl"
    _write_metrics(metrics, [_record(1, 0.08, 0.01, 0.12)])

    frame = load_epoch_metrics(metrics)

    assert frame.loc[0, "epoch"] == 1
    assert frame.loc[0, "alpha"] == 0.08
    assert frame.loc[0, "loss_global_ic"] == -0.08
    assert frame.loc[0, "checkpoint"] == "epoch_001.pt"


def test_score_checkpoints_prioritizes_gate_then_top_metrics():
    frame = pd.DataFrame(
        [
            {"epoch": 1, "alpha": 0.10, "rawtopret_h5_top0p6": 0.01, "rawtopstable_h5_top0p6": 0.10},
            {"epoch": 2, "alpha": 0.06, "rawtopret_h5_top0p6": 0.03, "rawtopstable_h5_top0p6": 0.30},
            {"epoch": 3, "alpha": 0.09, "rawtopret_h5_top0p6": 0.02, "rawtopstable_h5_top0p6": 0.20},
        ]
    )

    ranked = score_checkpoints(frame)

    assert ranked["epoch"].tolist() == [3, 1, 2]
    assert bool(ranked.loc[0, "passed_gates"]) is True
    assert bool(ranked.loc[2, "passed_gates"]) is False
    assert "alpha min 0.07" in ranked.loc[2, "failed_gates"]


def test_select_checkpoints_accepts_custom_rule(tmp_path):
    metrics = tmp_path / "epoch_metrics.jsonl"
    _write_metrics(
        metrics,
        [
            _record(1, 0.08, 0.01, 0.12),
            _record(2, 0.09, 0.02, 0.10),
        ],
    )
    rule = SelectionRule(
        gates={"alpha": ("min", 0.07)},
        rank_metrics=(("rawtopret_h5_top0p6", "max", 1.0),),
    )

    ranked = select_checkpoints(metrics, rule)

    assert ranked["epoch"].tolist() == [2, 1]
