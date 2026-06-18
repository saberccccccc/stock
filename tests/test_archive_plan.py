import csv

from experiments.archive_plan import (
    build_archive_plan,
    is_protected,
    plan_inventory_row,
    write_archive_plan_csv,
)


def test_is_protected_covers_active_paths():
    assert is_protected("forward_results")
    assert is_protected("v9_avgw3_open_ledger_20260617")
    assert is_protected("checkpoints_exp_topfocus_w005_topic")
    assert not is_protected("backtest_results_exp_base_avgw3_val")


def test_plan_inventory_row_routes_archive_candidates():
    row = plan_inventory_row(
        {
            "name": "loss_ablation_queue.pid",
            "kind": "file",
            "class": "runtime_log_or_pid",
        }
    )

    assert row.action == "archive_candidate"
    assert row.target == "archive/logs_202606"


def test_plan_inventory_row_protects_official_paths():
    row = plan_inventory_row(
        {
            "name": "v9_avgw3_open_ledger_20260617",
            "kind": "dir",
            "class": "experiment_output",
        }
    )

    assert row.action == "protect"
    assert row.target == ""


def test_build_and_write_archive_plan(tmp_path):
    inventory = tmp_path / "inventory.csv"
    with inventory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "kind", "class", "length", "last_write"])
        writer.writeheader()
        writer.writerow({"name": "core", "kind": "dir", "class": "source_or_docs"})
        writer.writerow({"name": "errors.log", "kind": "file", "class": "runtime_log_or_pid"})

    rows = build_archive_plan(inventory)
    output = tmp_path / "archive_plan.csv"
    write_archive_plan_csv(rows, output)

    loaded = list(csv.DictReader(output.open(encoding="utf-8")))
    assert [row["action"] for row in loaded] == ["protect", "archive_candidate"]
