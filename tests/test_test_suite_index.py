import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests"
RULES_PATH = ROOT / "PROJECT_RULES.md"


def test_current_rules_require_proportional_regression_coverage():
    rules_text = RULES_PATH.read_text(encoding="utf-8")

    assert "focused then proportional regression tests" in rules_text
    assert "Backtest changes test timing, costs, lots, ADV, limits, and no-lookahead" in rules_text


def test_every_python_test_file_defines_collectable_tests():
    missing = []
    for test_file in sorted(TEST_DIR.glob("test_*.py")):
        tree = ast.parse(test_file.read_text(encoding="utf-8-sig"))
        has_test = any(
            (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            )
            or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
            for node in tree.body
        )
        if not has_test:
            missing.append(test_file.relative_to(ROOT).as_posix())

    assert missing == []
