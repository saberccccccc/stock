from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests"
INDEX_PATH = ROOT / "reports" / "codebase_cleanup_20260618" / "test_suite_index.md"


def test_every_python_test_is_classified():
    index_text = INDEX_PATH.read_text(encoding="utf-8")
    test_files = sorted(TEST_DIR.glob("test_*.py"))

    missing = [
        test_file.relative_to(ROOT).as_posix()
        for test_file in test_files
        if f"`{test_file.relative_to(ROOT).as_posix()}`" not in index_text
    ]

    assert missing == []
