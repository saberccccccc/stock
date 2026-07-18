import json

import pandas as pd

from alpha.io import iter_alpha_rows, write_alpha_rows
from run.build_realistic_topk_ledger_dataset_v2 import main


def test_exports_proposal_alpha_preserving_source_universe(tmp_path):
    topk = pd.DataFrame(
        [
            {
                "split": "val_2024",
                "date": "2024-01-02",
                "proposal": "baseline",
                "selected_codes": "B;A",
            },
            {
                "split": "val_2024",
                "date": "2024-01-02",
                "proposal": "risk_mild",
                "selected_codes": "C;B;MISSING",
            },
        ]
    )
    topk_path = tmp_path / "topk.parquet"
    topk.to_parquet(topk_path)
    source_path = tmp_path / "source.jsonl"
    write_alpha_rows(
        source_path,
        [
            {
                "date": "2024-01-02",
                "codes": ["A", "B", "C", "D"],
                "alpha": [0.4, 0.3, 0.2, 0.1],
            }
        ],
    )
    out_dir = tmp_path / "out"

    main(
        [
            "--topk-dataset",
            str(topk_path),
            "--source-alpha-jsonl",
            str(source_path),
            "--output-dir",
            str(out_dir),
            "--split-name",
            "val_2024",
            "--max-data-date",
            "2024-12-31",
            "--proposal",
            "risk_mild",
        ]
    )

    rows = list(iter_alpha_rows(out_dir / "proposal_alpha" / "risk_mild.jsonl"))
    assert len(rows) == 1
    assert rows[0]["codes"] == ["C", "B", "A", "D"]
    assert len(rows[0]["codes"]) == rows[0]["n_stocks"] == 4
    assert len(rows[0]["alpha"]) == 4
    assert rows[0]["selected_count"] == 3
    assert rows[0]["selected_in_source"] == 2

    manifest = json.loads((out_dir / "realistic_topk_ledger_dataset_v2_manifest.json").read_text())
    assert manifest["split_name"] == "val_2024"
    assert manifest["max_data_date"] == "2024-12-31"
    assert manifest["proposal_summaries"][0]["signal_start"] == "2024-01-02"
    assert manifest["command_count"] == 4
    commands = json.loads((out_dir / "backtest_commands.json").read_text())
    assert commands[0]["command"][0:2] == ["python", "run/backtest_retention_open_ledger.py"]
    assert "--execution-constraint-mode" in commands[0]["command"]
    assert "realistic" in commands[0]["command"]
