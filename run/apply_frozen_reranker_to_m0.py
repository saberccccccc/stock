"""Apply a frozen V3 or V4 reranker to a retrained M0 Alpha stream."""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research_protocol import assert_alpha_rows_within_research
from run.validate_reranker_2024 import load_alpha_rows, write_alpha
from run.validate_reranker_v3_2024 import build_rows as build_v3_rows
from run.validate_reranker_v4_2024 import build_rows as build_v4_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("v3", "v4"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-alpha", required=True)
    parser.add_argument("--output-alpha", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--target-frac", type=float, default=0.006)
    parser.add_argument("--hold-frac", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = pd.read_parquet(args.dataset)
    with Path(args.model).open("rb") as handle:
        payload = pickle.load(handle)
    missing = sorted(set(payload["feature_columns"]) - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing reranker features: {missing}")
    base_rows = load_alpha_rows(args.base_alpha)
    assert_alpha_rows_within_research(base_rows, context="new M0 frozen reranker")
    builder = build_v3_rows if args.variant == "v3" else build_v4_rows
    rows, audit = builder(
        dataset,
        payload,
        base_rows,
        target_frac=args.target_frac,
        hold_frac=args.hold_frac,
    )
    output_alpha = Path(args.output_alpha)
    audit_output = Path(args.audit_output)
    output_alpha.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    write_alpha(output_alpha, rows)
    audit.to_csv(audit_output, index=False)
    changed = float(audit["changed_fills"].mean()) if len(audit) else 0.0
    print(
        f"variant={args.variant} rows={len(rows)} mean_changed_fills={changed:.3f} "
        f"output={output_alpha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
