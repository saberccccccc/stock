"""Build a state-aware reranker dataset for the retrained M0 model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run.build_reranker_v2_dataset import (
    daily_normalize,
    market_features,
    stock_open_label_rows,
)
from run.build_reranker_v3_dataset import add_state_features


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--index-file", default="hs300_index.csv")
    parser.add_argument("--target-frac", type=float, default=0.006)
    parser.add_argument("--hold-frac", type=float, required=True)
    parser.add_argument("--candidate-end", type=int, default=80)
    parser.add_argument("--max-reranked-fills", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    data_dir = Path(args.data_dir)
    destination = Path(args.output)
    frame = pd.read_parquet(source)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    max_label_date = pd.Timestamp(f"{frame['date'].dt.year.max()}-12-31")
    label_frame = frame[frame["candidate_position"] < args.candidate_end]
    wanted = {
        code: set(group["date"])
        for code, group in label_frame.groupby("code", sort=False)
    }
    records = []
    for index, (code, dates) in enumerate(wanted.items(), start=1):
        path = data_dir / f"{code}.csv"
        if path.exists():
            rows = stock_open_label_rows(path, dates, max_label_date)
            for row in rows:
                row["code"] = code
            records.extend(rows)
        if index % 500 == 0:
            print(f"labels {index}/{len(wanted)}", flush=True)
    labels = pd.DataFrame(records)
    frame = frame.merge(labels, on=["date", "code"], how="left", validate="one_to_one")
    market = market_features(data_dir / args.index_file)
    frame = frame.merge(
        market.reset_index().rename(columns={market.index.name or "index": "date"}),
        on="date",
        how="left",
        validate="many_to_one",
    )
    frame["exec_target"] = np.nan
    labelled = frame["exec_target_raw"].notna()
    normalized = daily_normalize(frame.loc[labelled].copy())
    frame.loc[labelled, "exec_target"] = normalized["exec_target"].to_numpy()
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    eligible = frame["v3_eligible"].eq(1) & frame["exec_target"].notna()
    summary = {
        "source": str(source),
        "label_mode": "open_to_open",
        "target_frac": args.target_frac,
        "hold_frac": args.hold_frac,
        "candidate_end": args.candidate_end,
        "max_reranked_fills": args.max_reranked_fills,
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "eligible_labelled_rows": int(eligible.sum()),
        "eligible_labelled_dates": int(frame.loc[eligible, "date"].nunique()),
        "mean_vacancies": float(frame.groupby("date")["v3_vacancies"].first().mean()),
    }
    destination.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
