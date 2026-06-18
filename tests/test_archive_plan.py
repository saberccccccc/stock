import csv

from experiments.archive_plan import (
    build_archive_plan,
    build_archive_moves,
    execute_archive_moves,
    is_protected,
    load_archive_plan,
    plan_inventory_row,
    write_archive_plan_csv,
)


def test_is_protected_covers_active_paths():
    assert is_protected("forward_results")
    assert is_protected("archive")
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


def test_build_archive_moves_only_uses_archive_candidates(tmp_path):
    rows = [
        plan_inventory_row({"name": "core", "kind": "dir", "class": "source_or_docs"}),
        plan_inventory_row({"name": "errors.log", "kind": "file", "class": "runtime_log_or_pid"}),
        plan_inventory_row({"name": ".vscode", "kind": "dir", "class": "misc"}),
    ]
    (tmp_path / "errors.log").write_text("oops", encoding="utf-8")

    moves = build_archive_moves(rows, root=tmp_path)

    assert len(moves) == 1
    assert moves[0].source == tmp_path / "errors.log"
    assert moves[0].target == tmp_path / "archive" / "logs_202606" / "errors.log"
    assert moves[0].item_class == "runtime_log_or_pid"


def test_build_archive_moves_filters_class_and_target(tmp_path):
    rows = [
        plan_inventory_row({"name": "errors.log", "kind": "file", "class": "runtime_log_or_pid"}),
        plan_inventory_row({"name": ".pytest_cache", "kind": "dir", "class": "archive_or_cache"}),
    ]
    (tmp_path / "errors.log").write_text("oops", encoding="utf-8")
    (tmp_path / ".pytest_cache").mkdir()

    moves = build_archive_moves(rows, root=tmp_path, item_class="runtime_log_or_pid")
    assert [move.name for move in moves] == ["errors.log"]

    moves = build_archive_moves(rows, root=tmp_path, target="archive/cache_202606")
    assert [move.name for move in moves] == [".pytest_cache"]


def test_build_archive_moves_filters_name_prefix_and_glob(tmp_path):
    rows = [
        plan_inventory_row(
            {
                "name": "backtest_results_exp_base_avgw3_val",
                "kind": "dir",
                "class": "experiment_output",
            }
        ),
        plan_inventory_row(
            {
                "name": "backtest_results_test_plan_v9_avgw3_val",
                "kind": "dir",
                "class": "experiment_output",
            }
        ),
        plan_inventory_row(
            {
                "name": "candidate_model_validation_20260614",
                "kind": "dir",
                "class": "experiment_output",
            }
        ),
        plan_inventory_row(
            {
                "name": "backtest_results_summary_20260528.txt",
                "kind": "file",
                "class": "experiment_output",
            }
        ),
    ]
    for row in rows:
        path = tmp_path / row.name
        if row.kind == "dir":
            path.mkdir()
        else:
            path.write_text("summary", encoding="utf-8")

    prefix_moves = build_archive_moves(
        rows,
        root=tmp_path,
        item_class="experiment_output",
        name_prefix="backtest_results_exp_",
    )
    assert [move.name for move in prefix_moves] == ["backtest_results_exp_base_avgw3_val"]

    glob_moves = build_archive_moves(
        rows,
        root=tmp_path,
        item_class="experiment_output",
        name_glob="backtest_results_summary_*.txt",
    )
    assert [move.name for move in glob_moves] == ["backtest_results_summary_20260528.txt"]


def test_execute_archive_moves_moves_file_in_tmpdir(tmp_path):
    rows = [
        plan_inventory_row({"name": "errors.log", "kind": "file", "class": "runtime_log_or_pid"}),
    ]
    source = tmp_path / "errors.log"
    source.write_text("oops", encoding="utf-8")

    moves = build_archive_moves(rows, root=tmp_path)
    execute_archive_moves(moves)

    assert not source.exists()
    assert (tmp_path / "archive" / "logs_202606" / "errors.log").read_text(encoding="utf-8") == "oops"


def test_load_archive_plan_roundtrip(tmp_path):
    rows = [
        plan_inventory_row({"name": "errors.log", "kind": "file", "class": "runtime_log_or_pid"}),
    ]
    output = tmp_path / "archive_plan.csv"
    write_archive_plan_csv(rows, output)

    loaded = load_archive_plan(output)

    assert loaded == rows
