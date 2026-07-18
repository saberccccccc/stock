import json

import pandas as pd

from run.train_topk_portfolio_policy_lgbm import (
    feature_columns,
    group_sizes_by_date,
    main,
    relevance_labels,
)


def test_topk_feature_columns_exclude_labels_and_selected_codes():
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "proposal": ["baseline"],
            "portfolio_utility": [0.1],
            "baseline_utility": [0.0],
            "utility_delta_vs_baseline": [0.1],
            "selected_codes": ["A;B"],
            "mean_beta_60d": [1.1],
            "cfg_rank_weight": [1.0],
        }
    )

    cols = feature_columns(frame)

    assert "portfolio_utility" not in cols
    assert "baseline_utility" not in cols
    assert "utility_delta_vs_baseline" not in cols
    assert "selected_codes" not in cols
    assert "mean_beta_60d" in cols
    assert "cfg_rank_weight" in cols


def test_relevance_labels_are_date_local_for_proposals():
    frame = pd.DataFrame(
        {
            "date": ["d1", "d1", "d1", "d2", "d2"],
            "utility_delta_vs_baseline": [-1.0, 0.0, 1.0, 10.0, 11.0],
        }
    )

    labels = relevance_labels(frame, "utility_delta_vs_baseline", bins=5)

    assert labels[:3].tolist() == [1, 3, 4]
    assert labels[3:].tolist() == [2, 4]
    assert group_sizes_by_date(frame) == [3, 2]


def test_main_accepts_custom_split_names(tmp_path):
    rows = []
    for date in ["2024-01-01", "2024-01-02", "2024-01-03"]:
        rows.append({"date": date, "proposal": "baseline", "utility_delta_vs_baseline": 0.0, "cfg_rank_weight": 0.0})
        rows.append({"date": date, "proposal": "risk_mild", "utility_delta_vs_baseline": 0.1, "cfg_rank_weight": 1.0})
    frame = pd.DataFrame(rows)
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    frame.to_parquet(train_path)
    frame.to_parquet(test_path)
    out_dir = tmp_path / "out"

    main(
        [
            "--train-dataset",
            str(train_path),
            "--test-dataset",
            str(test_path),
            "--output-dir",
            str(out_dir),
            "--train-name",
            "oof_2018_2023",
            "--test-name",
            "val_2024",
            "--selection-name",
            "val_2024",
            "--num-boost-round",
            "2",
            "--min-data-in-leaf",
            "1",
        ]
    )

    meta = json.loads((out_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert meta["train_name"] == "oof_2018_2023"
    assert meta["test_name"] == "val_2024"
    assert meta["selection_names"] == ["val_2024"]
    assert {row["split"] for row in meta["summaries"]} == {"oof_2018_2023", "val_2024"}
    markdown = (out_dir / "topk_portfolio_policy_summary.md").read_text(encoding="utf-8")
    assert "Selection splits: val_2024" in markdown
