import json
from types import SimpleNamespace

import pandas as pd
import pytest

import backtest.open_ledger as ledger


def _frames(value=1.0):
    index = pd.DatetimeIndex(["2026-01-05"], name="trade_date")
    return {
        field: pd.DataFrame([[value]], index=index, columns=["000001.SZ"])
        for field in ("open", "high", "low", "close", "volume", "money")
    }


def _args(report):
    return SimpleNamespace(
        ohlc_backend="monthly",
        ohlc_shadow_backend="csv",
        ohlc_shadow_report=str(report),
    )


def test_dual_read_writes_exact_pass_report(tmp_path, monkeypatch):
    primary = _frames()
    shadow = _frames()

    def fake_load(args, codes, *, start_date, end_date):
        return primary if args.ohlc_backend == "monthly" else shadow

    monkeypatch.setattr(ledger, "_load_execution_market_frames_once", fake_load)
    report_path = tmp_path / "dual.json"

    result = ledger.load_execution_market_frames(
        _args(report_path),
        ["000001.SZ"],
        start_date="2026-01-05",
        end_date="2026-01-05",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result is primary
    assert report["status"] == "passed"
    assert report["fields"]["open"]["missing"] == 0


def test_dual_read_fails_closed_and_records_difference(tmp_path, monkeypatch):
    primary = _frames()
    shadow = _frames()
    shadow["close"].iloc[0, 0] = 2.0

    def fake_load(args, codes, *, start_date, end_date):
        return primary if args.ohlc_backend == "monthly" else shadow

    monkeypatch.setattr(ledger, "_load_execution_market_frames_once", fake_load)
    report_path = tmp_path / "dual.json"

    with pytest.raises(ValueError, match="dual-read parity failed"):
        ledger.load_execution_market_frames(
            _args(report_path),
            ["000001.SZ"],
            start_date="2026-01-05",
            end_date="2026-01-05",
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "AssertionError" in report["error"]
