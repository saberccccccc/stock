"""Apply the saved open reranker to forward alpha rows.

This is the forward-only counterpart of train_open_reranker_current_v9.py.
It does not train a model and does not use the historical cross-section cache;
instead it builds inference samples directly from the requested forward dates.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backtest.runtime import build_v9_backtest_config
from core.research_protocol import assert_alpha_rows_within_forward
from data.pipeline import build_inference_samples
from run.train_open_reranker_current_v9 import (
    load_alpha_rows,
    score_alpha_rows,
    write_alpha,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/forward_raw")
    parser.add_argument("--candidate-top-n", type=int, default=300)
    parser.add_argument("--candidate-start-rank", type=int, default=0)
    parser.add_argument("--alpha-weights", default="0.95")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_alpha_rows(args.alpha_jsonl)
    if not rows:
        raise ValueError(f"No rows in {args.alpha_jsonl}")
    assert_alpha_rows_within_forward(rows, context="open-reranker forward input")

    with Path(args.model).open("rb") as handle:
        bundle = pickle.load(handle)
    model = bundle["model"]
    features = bundle["feature_columns"]

    cfg = build_v9_backtest_config()
    cfg.data_dir = args.data_dir
    query_dates = [pd.Timestamp(row["date"]).strftime("%Y-%m-%d") for row in rows]
    stock_universe = sorted(
        {
            str(code)
            for row in rows
            for code in row["codes"][: max(int(args.candidate_top_n), int(args.candidate_start_rank) + 1)]
        }
    )
    print(f"forward stock_universe={len(stock_universe)} from top candidates", flush=True)
    samples = build_inference_samples(cfg, query_dates, stock_universe=stock_universe)
    samples_by_date = {pd.Timestamp(sample["date"]): sample for sample in samples}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for weight_raw in args.alpha_weights.split(","):
        weight = float(weight_raw.strip())
        scored, scored_dates = score_alpha_rows(
            rows,
            samples_by_date,
            model,
            features,
            args.candidate_top_n,
            args.candidate_start_rank,
            weight,
        )
        rank_suffix = ""
        if int(args.candidate_start_rank) != 0 or int(args.candidate_top_n) != 300:
            rank_suffix = f"_r{int(args.candidate_start_rank):03d}_{int(args.candidate_top_n):03d}"
        out_path = out_dir / f"forward_openrerank_w{int(round(weight * 100)):03d}{rank_suffix}.jsonl"
        write_alpha(out_path, scored)
        print(
            f"forward weight={weight:.2f} scored_dates={scored_dates}/{len(rows)} -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
