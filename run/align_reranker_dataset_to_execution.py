"""Align reranker candidate positions and state to the maxret execution filter."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.transforms import load_signal_returns
from run.build_reranker_v3_dataset import add_state_features


STATE_COLUMNS = {
    "v3_was_held",
    "v3_is_kept",
    "v3_holding_age",
    "v3_vacancies",
    "v3_rerank_slots",
    "v3_protected_fill",
    "v3_eligible",
    "v3_baseline_fill",
}


def align_candidate_positions(frame, signal_returns, max_signal_return=0.095):
    aligned = frame.copy()
    aligned["date"] = pd.to_datetime(aligned["date"])
    aligned["code"] = aligned["code"].astype(str)
    aligned["raw_candidate_position"] = aligned["candidate_position"].astype(int)
    aligned["execution_signal_return"] = [
        signal_returns.get(pd.Timestamp(date), {}).get(str(code))
        for date, code in zip(aligned["date"], aligned["code"])
    ]
    aligned["execution_demoted"] = (
        aligned["execution_signal_return"].notna()
        & (aligned["execution_signal_return"] >= float(max_signal_return))
    ).astype("int8")
    aligned = aligned.sort_values(
        ["date", "execution_demoted", "raw_candidate_position", "code"],
        kind="mergesort",
    )
    aligned["candidate_position"] = aligned.groupby("date").cumcount().astype(int)
    aligned["group_size"] = aligned.groupby("date")["code"].transform("size").astype(int)
    return aligned


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--max-signal-return", type=float, default=0.095)
    parser.add_argument("--target-frac", type=float, default=0.006)
    parser.add_argument("--hold-frac", type=float, required=True)
    parser.add_argument("--candidate-end", type=int, default=80)
    parser.add_argument("--max-reranked-fills", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_parquet(args.source)
    frame = frame.drop(columns=[c for c in STATE_COLUMNS if c in frame], errors="ignore")
    dates = pd.to_datetime(frame["date"])
    returns = load_signal_returns(
        args.data_dir,
        set(frame["code"].astype(str)),
        dates.min(),
        dates.max(),
    )
    frame = align_candidate_positions(frame, returns, args.max_signal_return)
    audit = pd.read_csv(args.audit)
    frame = add_state_features(
        frame,
        audit,
        target_frac=args.target_frac,
        hold_frac=args.hold_frac,
        max_candidate_rank=args.candidate_end,
        max_reranked_fills=args.max_reranked_fills,
    )
    frame = frame[frame["candidate_position"] < args.candidate_end].copy()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    summary = {
        "source": args.source,
        "ranking": "maxret095",
        "target_frac": args.target_frac,
        "hold_frac": args.hold_frac,
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "demoted_rows": int(frame["execution_demoted"].sum()),
        "mean_vacancies": float(frame.groupby("date")["v3_vacancies"].first().mean()),
    }
    destination.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
