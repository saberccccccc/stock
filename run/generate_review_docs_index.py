"""Generate an index for root-level documents that remain in manual review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.review_docs import (  # noqa: E402
    build_review_doc_index,
    write_review_doc_csv,
    write_review_doc_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cleanup review document index")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--archive-plan",
        default="reports/codebase_cleanup_20260618/archive_plan.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/codebase_cleanup_20260618",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    docs = build_review_doc_index(args.root, args.archive_plan)
    csv_path = output_dir / "review_docs_index.csv"
    md_path = output_dir / "review_docs_index.md"
    write_review_doc_csv(docs, csv_path)
    write_review_doc_markdown(docs, md_path)
    print(f"wrote {csv_path} rows={len(docs)}", flush=True)
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
