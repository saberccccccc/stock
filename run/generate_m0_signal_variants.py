"""Generate raw and smoothed M0 alpha variants with one model inference pass."""

import argparse
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alpha.persistence import rank_alpha_rows, rolling_average_alpha_scores
from core.research_protocol import RESEARCH_END_DATE, assert_alpha_rows_within_research
from run.backtest_v9_retention import compute_v9_alpha_scores, load_v9_samples_and_predictor
from run.validate_candidate_models import apply_chase_filter, generation_args, write_alpha


def parse_variant_specs(value):
    specs = []
    for token in (part.strip().lower() for part in value.split(",")):
        if not token:
            continue
        if token == "raw":
            specs.append(("raw", "none", 1))
            continue
        if token.startswith("avg") and token[3:].isdigit() and int(token[3:]) >= 2:
            specs.append((token, "average", int(token[3:])))
            continue
        raise ValueError(f"unknown signal variant: {token}")
    if not specs:
        raise ValueError("at least one signal variant is required")
    return specs


def pending_variant_specs(output_root, name, specs):
    """Return variants without a completed filtered signal file."""
    pending = []
    for label, mode, window in specs:
        filtered_path = Path(output_root) / name / label / "alpha_maxret095.jsonl"
        if filtered_path.exists() and filtered_path.stat().st_size > 0:
            print(f"SKIP completed variant: {label}", flush=True)
            continue
        pending.append((label, mode, window))
    return pending


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--variants", default="raw,avg2,avg3,avg5")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=160)
    parser.add_argument("--score-source", default="alpha", choices=["alpha","raw_alpha","horizon0","horizon1","horizon2","horizon3"], help="Source of scores: alpha (default, tanh(z-score)), raw_alpha, or horizon0-3")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    specs = parse_variant_specs(args.variants)
    pending_specs = pending_variant_specs(output_root, args.name, specs)
    if not pending_specs:
        print("All requested variants are already complete.", flush=True)
        return

    load_args = generation_args(
        Path(args.checkpoint), output_root / args.name, args.device, args.progress_every
    )
    load_args.predictor_mode = "none"
    load_args.window = 1
    load_args.split = args.split
    if args.start_date is not None:
        load_args.start_date = args.start_date
    elif args.split == "test":
        load_args.start_date = "2025-01-01"
    if args.end_date is not None:
        load_args.end_date = args.end_date
    elif args.split == "test":
        load_args.end_date = str(RESEARCH_END_DATE.date())
    _, samples, raw_predictor = load_v9_samples_and_predictor(load_args)

    if args.score_source != "alpha":
        raw_predictor.base.score_source = args.score_source
    print("Running one shared model-inference pass for all requested variants...", flush=True)
    score_rows = compute_v9_alpha_scores(samples, raw_predictor, args.progress_every)
    for label, mode, window in pending_specs:
        variant_dir = output_root / args.name / label
        raw_path = variant_dir / "alpha_raw.jsonl"
        filtered_path = variant_dir / "alpha_maxret095.jsonl"
        rows = (
            rank_alpha_rows(score_rows)
            if mode == "none"
            else rank_alpha_rows(rolling_average_alpha_scores(score_rows, window))
        )
        assert_alpha_rows_within_research(
            rows, context=f"M0 {args.split} signal variant {label}"
        )
        write_alpha(raw_path, rows)
        del rows
        gc.collect()
        apply_chase_filter(raw_path, filtered_path, args.data_dir)
        print(f"completed variant: {label} -> {filtered_path}", flush=True)


if __name__ == "__main__":
    main()
