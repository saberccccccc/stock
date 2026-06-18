import csv

from experiments.archive_plan import write_archive_plan_csv
from experiments.checkpoint_reference_audit import (
    build_checkpoint_reference_audit,
    checkpoint_group,
    iter_reference_files,
    write_checkpoint_audit_csv,
)
from experiments.source_inventory import classify_top_level
from experiments.archive_plan import plan_inventory_row


def test_checkpoint_group_known_patterns():
    assert checkpoint_group("checkpoints_loss_ablation_A4") == "loss_ablation_a_series"
    assert checkpoint_group("checkpoints_loss_ablation_LAG005") == "downside_lag_topfocus_series"
    assert checkpoint_group("checkpoints_reranker_oof_F1_train2017_val2018") == "reranker_oof"
    assert checkpoint_group("models_multi_v9_tech_macro") == "legacy_v9_model"


def test_iter_reference_files_ignores_archive_and_data(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "note.md").write_text("x", encoding="utf-8")
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "old.md").write_text("x", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "raw.txt").write_text("x", encoding="utf-8")

    refs = [path.relative_to(tmp_path).as_posix() for path in iter_reference_files(tmp_path)]

    assert refs == ["reports/note.md"]


def test_build_checkpoint_reference_audit_detects_text_refs(tmp_path):
    rows = [
        plan_inventory_row(
            {
                "name": "checkpoints_loss_ablation_A4",
                "kind": "dir",
                "class": classify_top_level("checkpoints_loss_ablation_A4", "dir"),
            }
        ),
        plan_inventory_row(
            {
                "name": "checkpoints_reranker_oof_F1_train2017_val2018",
                "kind": "dir",
                "class": classify_top_level("checkpoints_reranker_oof_F1_train2017_val2018", "dir"),
            }
        ),
    ]
    archive_plan = tmp_path / "archive_plan.csv"
    write_archive_plan_csv(rows, archive_plan)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "decision.md").write_text(
        "use checkpoints_loss_ablation_A4 for comparison", encoding="utf-8"
    )

    audit = build_checkpoint_reference_audit(archive_plan, root=tmp_path)
    by_name = {row.name: row for row in audit}

    assert by_name["checkpoints_loss_ablation_A4"].reference_count == 1
    assert by_name["checkpoints_loss_ablation_A4"].decision == "hold"
    assert (
        by_name["checkpoints_reranker_oof_F1_train2017_val2018"].decision
        == "archive_after_matching_artifact_ledger"
    )


def test_write_checkpoint_audit_csv(tmp_path):
    rows = [
        plan_inventory_row(
            {
                "name": "switch_value_models_demo",
                "kind": "dir",
                "class": classify_top_level("switch_value_models_demo", "dir"),
            }
        )
    ]
    archive_plan = tmp_path / "archive_plan.csv"
    write_archive_plan_csv(rows, archive_plan)
    audit = build_checkpoint_reference_audit(archive_plan, root=tmp_path)
    output = tmp_path / "audit.csv"

    write_checkpoint_audit_csv(audit, output)

    loaded = list(csv.DictReader(output.open(encoding="utf-8")))
    assert loaded[0]["name"] == "switch_value_models_demo"
    assert loaded[0]["group"] == "switch_value_model"
