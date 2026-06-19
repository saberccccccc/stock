import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from run.diagnose_negative_filter import PriceCache, iter_aligned_rows
from run.summarize_open_ledger_candidates import load_rows


TOOL_MODULES = (
    "run.compare_open_ledger_diagnostics",
    "run.diagnose_negative_filter",
    "run.summarize_candidate_stability",
    "run.summarize_open_ledger_candidates",
    "run.sweep_open_ledger_params",
    "run.sweep_open_price_ledger_params",
)


@pytest.mark.parametrize("module_name", TOOL_MODULES)
def test_open_ledger_tool_imports_without_changing_cwd(module_name):
    cwd = Path.cwd()
    importlib.import_module(module_name)
    assert Path.cwd() == cwd


def test_candidate_summary_prefers_explicit_split(tmp_path):
    path = tmp_path / "sweep.csv"
    pd.DataFrame(
        [{
            "split": "test",
            "n_return_days": 100,
            "portfolio_value": 500_000,
            "ann": 10.0,
            "sharpe": 1.0,
            "mdd": 0.1,
        }]
    ).to_csv(path, index=False)

    assert load_rows(path)[0]["split"] == "test"


def test_aligned_rows_reject_different_file_lengths(tmp_path):
    base = tmp_path / "base.jsonl"
    rerank = tmp_path / "rerank.jsonl"
    base.write_text(
        "\n".join(
            json.dumps({"date": date, "codes": []})
            for date in ("2024-01-02", "2024-01-03")
        ),
        encoding="utf-8",
    )
    rerank.write_text(json.dumps({"date": "2024-01-02", "codes": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="different row counts"):
        list(iter_aligned_rows(base, rerank))


def test_price_cache_hides_forward_rows(tmp_path):
    pd.DataFrame(
        {
            "trade_date": ["2026-05-18", "2026-05-19"],
            "open": [10.0, 11.0],
            "high": [10.0, 11.0],
            "low": [10.0, 11.0],
            "close": [10.0, 11.0],
            "money": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    ).to_csv(tmp_path / "000001.SZ.csv", index=False)

    cached = PriceCache(tmp_path, maxsize=1).get("000001.SZ")

    assert cached["trade_date"].max() == pd.Timestamp("2026-05-18")
