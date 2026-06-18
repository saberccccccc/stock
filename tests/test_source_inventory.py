import csv

from experiments.source_inventory import (
    classify_top_level,
    inventory_markdown,
    scan_top_level,
    write_inventory_csv,
)


def test_classify_top_level_known_groups():
    assert classify_top_level("core", "dir") == "source_or_docs"
    assert classify_top_level("checkpoints_loss_ablation_A0", "dir") == "checkpoint_or_model"
    assert classify_top_level("backtest_results_exp_base_val", "dir") == "experiment_output"
    assert classify_top_level("loss_ablation_queue.pid", "file") == "runtime_log_or_pid"
    assert classify_top_level("archive", "dir") == "archive_or_cache"
    assert classify_top_level("cache", "dir") == "archive_or_cache"


def test_scan_top_level_and_write_csv(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "cache").mkdir()
    (tmp_path / "loss_ablation_queue.pid").write_text("123", encoding="utf-8")

    items = scan_top_level(tmp_path)
    output = tmp_path / "inventory.csv"
    write_inventory_csv(items, output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    classes = {row["name"]: row["class"] for row in rows}
    assert classes["core"] == "source_or_docs"
    assert classes["cache"] == "archive_or_cache"
    assert classes["loss_ablation_queue.pid"] == "runtime_log_or_pid"


def test_inventory_markdown_contains_summary(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "backtest_results_demo").mkdir()

    markdown = inventory_markdown(scan_top_level(tmp_path))

    assert "# Source Inventory" in markdown
    assert "| source_or_docs, dir | 1 |" in markdown
    assert "| experiment_output, dir | 1 |" in markdown
