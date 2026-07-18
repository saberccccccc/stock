import json

import pandas as pd
import pytest

from run.assemble_realistic_topk_ledger_results import main


def _write_summary(path, ann, sharpe, mdd, portfolio_value=500000):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "target_frac": 0.006,
                "hold_frac": 0.10,
                "ann": ann,
                "sharpe": sharpe,
                "mdd": mdd,
                "avg_executed_turnover": 0.20,
                "total_cost": 0.01,
                "blocked_buy": 1,
                "adv_blocked": 0,
                "new_stock_buy_blocked": 0,
                "signal_start": "2024-01-02",
                "signal_end": "2024-01-05",
                "backtest_start": "2024-01-03",
                "backtest_end": "2024-01-09",
                "portfolio_value": portfolio_value,
            }
        ]
    ).to_csv(path, index=False)


def test_assemble_realistic_results_adds_baseline_delta(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "realistic_topk_ledger_dataset_v2_manifest.json").write_text(
        json.dumps({"split_name": "val_2024"}),
        encoding="utf-8",
    )
    _write_summary(root / "open_ledger" / "baseline" / "normal" / "open_ledger_summary.csv", 10.0, 1.0, 0.10)
    _write_summary(root / "open_ledger" / "risk_mild" / "normal" / "open_ledger_summary.csv", 12.0, 1.2, 0.08)
    out = tmp_path / "out"

    main(["--root-dir", str(root), "--split-name", "val_2024", "--output-dir", str(out)])

    frame = pd.read_parquet(out / "realistic_topk_ledger_results.parquet")
    baseline_delta = frame.loc[frame["proposal"].eq("baseline"), "ledger_utility_delta_vs_baseline"].iloc[0]
    risk_delta = frame.loc[frame["proposal"].eq("risk_mild"), "ledger_utility_delta_vs_baseline"].iloc[0]
    assert baseline_delta == pytest.approx(0.0)
    assert risk_delta > 0
    meta = json.loads((out / "assemble_manifest.json").read_text(encoding="utf-8"))
    assert meta["split_name"] == "val_2024"
    assert meta["required_date_columns"] == ["signal_start", "signal_end", "backtest_start", "backtest_end"]


def test_assemble_rejects_summary_without_required_dates(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "realistic_topk_ledger_dataset_v2_manifest.json").write_text(
        json.dumps({"split_name": "val_2024"}),
        encoding="utf-8",
    )
    path = root / "open_ledger" / "baseline" / "normal" / "open_ledger_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "target_frac": 0.006,
                "hold_frac": 0.10,
                "ann": 10.0,
                "sharpe": 1.0,
                "mdd": 0.10,
                "portfolio_value": 500000,
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        main(["--root-dir", str(root), "--split-name", "val_2024", "--output-dir", str(tmp_path / "out")])
