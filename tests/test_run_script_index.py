import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_PATH = ROOT / "ARCHITECTURE.md"


def test_current_architecture_documents_official_entrypoints():
    architecture_text = ARCHITECTURE_PATH.read_text(encoding="utf-8")

    for script in (
        "run/train.py",
        "run/generate_ledger_path_v3_signal.py",
        "run/official_backtest_from_registry.py",
        "run/freeze_formal_baseline.py",
        "run/sweep_open_price_ledger_params.py",
        "run/attribution_from_registry.py",
        "run/scorecard_from_registry.py",
    ):
        assert f"`{script}`" in architecture_text


def test_documented_primary_entrypoints_render_help():
    entrypoints = (
        ROOT / "run" / "train.py",
        ROOT / "run" / "generate_ledger_path_v3_signal.py",
        ROOT / "run" / "official_backtest_from_registry.py",
        ROOT / "run" / "freeze_formal_baseline.py",
        ROOT / "run" / "sweep_open_price_ledger_params.py",
        ROOT / "run" / "attribution_from_registry.py",
        ROOT / "run" / "scorecard_from_registry.py",
    )

    for entrypoint in entrypoints:
        result = subprocess.run(
            [sys.executable, str(entrypoint), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
