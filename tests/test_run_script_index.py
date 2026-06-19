import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "run"
INDEX_PATH = ROOT / "reports" / "codebase_cleanup_20260618" / "run_script_index.md"


def test_every_run_python_file_is_classified():
    index_text = INDEX_PATH.read_text(encoding="utf-8")
    run_scripts = sorted(RUN_DIR.glob("*.py"))

    missing = [
        script.relative_to(ROOT).as_posix()
        for script in run_scripts
        if f"`{script.relative_to(ROOT).as_posix()}`" not in index_text
    ]

    assert len(run_scripts) == 93
    assert missing == []


def test_documented_primary_entrypoints_render_help():
    entrypoints = (
        ROOT / "data" / "update.py",
        ROOT / "run" / "backtest.py",
        ROOT / "run" / "daily_top10.py",
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
