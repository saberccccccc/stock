"""Generate frozen M0 Alpha and V3 candidate features from forward-only data."""

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backtest.engine import detect_regime
from core.research_protocol import FORWARD_START_DATE
from data.pipeline import build_inference_samples
from run.build_reranker_dataset import alpha_history_features, percentile_rank_desc
from run.build_reranker_v2_dataset import market_features
from run.forward_frozen_strategy import (
    assert_fundamental_data_current,
    assert_market_data_current,
    load_frozen_predictor,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/forward_raw")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints_loss_ablation_M0_nomulti/epochs/epoch_006.pt",
    )
    parser.add_argument("--start-date", default="2026-05-19")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", default="forward_results/m0_v3_20260615")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if start < FORWARD_START_DATE:
        raise ValueError("Forward start precedes frozen forward boundary")
    assert_market_data_current(args.data_dir, end)
    assert_fundamental_data_current()
    predictor = load_frozen_predictor(args.checkpoint, args.device, "avgw3")

    from backtest.runtime import build_v9_backtest_config

    cfg = build_v9_backtest_config()
    cfg.data_dir = args.data_dir
    query_start = start - pd.Timedelta(days=7)
    query_dates = [
        value.strftime("%Y-%m-%d")
        for value in pd.bdate_range(query_start, end)
    ]
    samples = build_inference_samples(cfg, query_dates)
    market = market_features(Path(args.data_dir) / "hs300_index.csv")
    alpha_history = defaultdict(lambda: deque(maxlen=3))
    rank_history = defaultdict(lambda: deque(maxlen=3))
    alpha_rows = []
    candidate_rows = []
    seen = set()

    for sample in samples:
        date = pd.Timestamp(sample["date"])
        if date in seen:
            continue
        seen.add(date)
        codes = np.asarray(sample["codes"], dtype=object)
        valid = np.ones(len(codes), dtype=bool)
        regime = detect_regime(sample)
        alpha = np.asarray(
            predictor.predict_alpha(sample, valid, regime), dtype=np.float64
        )
        rank_pct = percentile_rank_desc(alpha)
        order = np.argsort(-alpha, kind="mergesort")
        if date >= query_start:
            ordered_codes = [str(codes[index]) for index in order]
            alpha_rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "codes": ordered_codes,
                    "alpha": alpha[order].astype(float).tolist(),
                    "n_stocks": len(ordered_codes),
                    "regime": regime,
                }
            )
            for position, index in enumerate(order[:500]):
                code = str(codes[index])
                row = {
                    "date": date,
                    "code": code,
                    "candidate_position": position,
                    "industry_id": int(sample["industry_ids"][index]),
                    "market_regime": 0,
                    "group_size": min(500, len(order)),
                }
                row.update(
                    alpha_history_features(
                        code,
                        alpha[index],
                        rank_pct[index],
                        alpha_history,
                        rank_history,
                    )
                )
                row.update(
                    {
                        f"x_{feature_index:03d}": float(value)
                        for feature_index, value in enumerate(sample["X"][index])
                    }
                )
                row.update(
                    {
                        f"risk_{feature_index:03d}": float(value)
                        for feature_index, value in enumerate(sample["risk"][index])
                    }
                )
                if date in market.index:
                    row.update(market.loc[date].to_dict())
                candidate_rows.append(row)
        for index, code_value in enumerate(codes):
            code = str(code_value)
            alpha_history[code].append(float(alpha[index]))
            rank_history[code].append(float(rank_pct[index]))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "alpha_raw.jsonl").open("w", encoding="utf-8") as handle:
        for row in alpha_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(candidate_rows).to_parquet(
        output / "reranker_v3_forward.parquet", index=False
    )
    print(
        f"Saved {len(alpha_rows)} Alpha dates and {len(candidate_rows):,} candidates "
        f"to {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
