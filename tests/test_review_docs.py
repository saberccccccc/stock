import csv

from experiments.review_docs import (
    build_review_doc_index,
    load_review_markdown_names,
    review_doc_markdown,
    write_review_doc_csv,
)


def _write_archive_plan(path):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "kind", "class", "action", "target", "reason"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "README.md",
                "kind": "file",
                "class": "source_or_docs",
                "action": "review",
            }
        )
        writer.writerow(
            {
                "name": "LOSS_ABLATION_PLAN.md",
                "kind": "file",
                "class": "source_or_docs",
                "action": "review",
            }
        )
        writer.writerow(
            {
                "name": "errors.log",
                "kind": "file",
                "class": "runtime_log_or_pid",
                "action": "archive_candidate",
            }
        )


def test_load_review_markdown_names_filters_archive_candidates(tmp_path):
    archive_plan = tmp_path / "archive_plan.csv"
    _write_archive_plan(archive_plan)

    names = load_review_markdown_names(archive_plan)

    assert names == ["README.md", "LOSS_ABLATION_PLAN.md"]


def test_build_review_doc_index_uses_known_metadata(tmp_path):
    archive_plan = tmp_path / "archive_plan.csv"
    _write_archive_plan(archive_plan)
    (tmp_path / "README.md").write_text(
        "# Project\n\n```bash\n# Not the title\n```\n\n## Quick start\n",
        encoding="utf-8",
    )
    (tmp_path / "LOSS_ABLATION_PLAN.md").write_text(
        "# Loss Plan\n\n## Objective\n",
        encoding="utf-8",
    )

    docs = build_review_doc_index(tmp_path, archive_plan)
    by_name = {doc.name: doc for doc in docs}

    assert by_name["README.md"].cleanup_decision == "keep"
    assert by_name["README.md"].title == "Project"
    assert by_name["LOSS_ABLATION_PLAN.md"].role == "training_plan"
    assert by_name["LOSS_ABLATION_PLAN.md"].consolidation_target.endswith(
        "training_research_index.md"
    )


def test_write_review_doc_outputs(tmp_path):
    archive_plan = tmp_path / "archive_plan.csv"
    _write_archive_plan(archive_plan)
    (tmp_path / "README.md").write_text("# Project\n\n## Quick start\n", encoding="utf-8")
    (tmp_path / "LOSS_ABLATION_PLAN.md").write_text(
        "# Loss Plan\n\n## Objective\n",
        encoding="utf-8",
    )
    docs = build_review_doc_index(tmp_path, archive_plan)

    output = tmp_path / "review_docs.csv"
    write_review_doc_csv(docs, output)
    markdown = review_doc_markdown(docs)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 2
    assert "# Review Document Index" in markdown
    assert "LOSS_ABLATION_PLAN.md" in markdown
