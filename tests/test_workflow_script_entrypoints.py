import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_workflow_child_scripts_bootstrap_project_imports():
    for script in (
        "run/official_backtest_from_registry.py",
        "run/scorecard_from_registry.py",
        "run/materialize_workflow_candidates.py",
        "run/rolling_strong_staged_pilot.py",
        "run/materialize_strong_rolling_manifest.py",
        "run/materialize_frozen_predictions.py",
        "run/materialize_workflow_records.py",
        "run/manage_shadow_lifecycle.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"
