import pandas as pd
import pytest

from run.summarize_apm_attribution import (
    industry_weights,
    main,
    normalize_ts_code,
    parse_holdings,
    summarize_attribution,
)


def test_normalize_ts_code_supports_baostock_and_tushare_formats():
    assert normalize_ts_code("sh.600000") == "600000.SH"
    assert normalize_ts_code("000001.SZ") == "000001.SZ"
    assert normalize_ts_code("000001") == "000001.SZ"


def test_parse_holdings_and_industry_weights():
    holdings = parse_holdings("sh.600000=0.2;000001.SZ=0.1;BAD=x")
    industry_map = {"600000.SH": "Bank", "000001.SZ": "Bank"}

    assert holdings == {"600000.SH": 0.2, "000001.SZ": 0.1}
    assert industry_weights(holdings, industry_map)["Bank"] == pytest.approx(0.3)


def test_summarize_attribution_minimal(tmp_path):
    pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=80),
            "close": [10 + i * 0.1 for i in range(80)],
            "money": [1000 + i for i in range(80)],
        }
    ).to_csv(tmp_path / "000001.SZ.csv", index=False)
    returns = pd.DataFrame(
        {
            "date": ["2025-03-20", "2025-03-21"],
            "return": [0.01, 0.02],
            "benchmark_return": [0.005, 0.006],
            "active_return": [0.005, 0.014],
        }
    )
    diag = pd.DataFrame(
        {
            "date": ["2025-03-20", "2025-03-21"],
            "holdings": ["000001.SZ=0.5", "000001.SZ=0.6"],
            "cost": [0.001, 0.002],
            "commission": [0.0002, 0.0003],
            "stamp_tax": [0.0004, 0.0005],
            "slippage": [0.0004, 0.0012],
            "market_mult": [1.0, 0.7],
            "gross_weight": [0.5, 0.6],
            "portfolio_beta_60d": [0.8, 0.9],
            "portfolio_beta_per_gross_60d": [1.6, 1.5],
            "portfolio_specific_vol_60d": [0.1, 0.2],
        }
    )

    summary, industry, style, slices = summarize_attribution(
        returns,
        diag,
        {"000001.SZ": "Bank"},
        tmp_path,
    )

    assert summary["days"] == 2
    assert summary["total_cost"] == 0.003
    assert summary["avg_industry_count"] == 1.0
    assert industry["top_industry"].tolist() == ["Bank", "Bank"]
    assert not style.empty
    assert set(slices["slice"]) >= {"market_state", "industry_concentration"}
    assert "defensive" in set(slices["value"])


def test_apm_attribution_main_writes_outputs(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=80),
            "close": [10 + i * 0.1 for i in range(80)],
            "money": [1000 + i for i in range(80)],
        }
    ).to_csv(data_dir / "000001.SZ.csv", index=False)
    returns_csv = tmp_path / "returns.csv"
    diagnostics_csv = tmp_path / "diagnostics.csv"
    industry_csv = tmp_path / "industry.csv"
    pd.DataFrame(
        {
            "date": ["2025-03-20"],
            "return": [0.01],
            "benchmark_return": [0.005],
            "active_return": [0.005],
        }
    ).to_csv(returns_csv, index=False)
    pd.DataFrame(
        {
            "date": ["2025-03-20"],
            "holdings": ["000001.SZ=0.5"],
            "cost": [0.001],
        }
    ).to_csv(diagnostics_csv, index=False)
    pd.DataFrame(
        {
            "code": ["sz.000001"],
            "industry": ["Bank"],
        }
    ).to_csv(industry_csv, index=False)
    output = tmp_path / "out"

    main([
        "--returns-csv",
        str(returns_csv),
        "--diagnostics-csv",
        str(diagnostics_csv),
        "--industry-csv",
        str(industry_csv),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output),
    ])

    assert (output / "apm_attribution_summary.csv").exists()
    assert (output / "apm_industry_exposure.csv").exists()
    assert (output / "apm_style_exposure.csv").exists()
    assert (output / "apm_slice_summary.csv").exists()
    assert (output / "apm_attribution_report.md").exists()
