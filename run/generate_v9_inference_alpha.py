"""Generate V9 Alpha rankings for label-free inference dates."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backtest.engine import detect_regime
from backtest.predictors import PersistentPredictor
from backtest.runtime import build_v9_backtest_config, load_dl_predictor
from core.research_protocol import assert_alpha_rows_within_research
from data.pipeline import build_cross_section_dataset, build_inference_samples, samples_from_precomputed_metadata
from run.v9_long_only_optimization import V9RankPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Generate label-free V9 Alpha rankings")
    parser.add_argument("--checkpoint", default="checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictor-mode", default="average", choices=["none", "average", "momentum", "composite"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def trading_dates(start_date, end_date):
    data_dir = Path("data/raw")
    dates = set()
    for path in data_dir.glob("*.csv"):
        if not path.name[:1].isdigit():
            continue
        try:
            frame = pd.read_csv(path, usecols=["trade_date"])
        except Exception:
            continue
        values = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
        dates.update(values[(values >= start_date) & (values <= end_date)])
        if len(dates) >= 1:
            # One normal stock is enough to discover the exchange calendar.
            break
    return sorted(pd.Timestamp(d) for d in dates)


def load_predictor(args, cfg):
    meta = build_cross_section_dataset(cfg, use_cache=True)
    if not isinstance(meta, dict):
        train_samples = meta[0]
    else:
        schema_meta = dict(meta)
        schema_meta["train_indices"] = [meta["train_indices"][0]]
        train_samples = samples_from_precomputed_metadata(schema_meta, "train")
    base = load_dl_predictor(args.checkpoint, train_samples, cfg, args.device)
    raw = V9RankPredictor(base, "v9_raw", cache={})
    if args.predictor_mode == "none":
        return raw
    return PersistentPredictor(raw, window=args.window, mode=args.predictor_mode)


def compute_rows(samples, predictor, progress_every):
    rows = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        n = int(sample["X"].shape[0])
        if n < 10:
            continue
        valid = np.ones(n, dtype=bool)
        regime = detect_regime(sample)
        alpha = predictor.predict_alpha(sample, valid, regime)
        if alpha.shape[0] != n:
            raise RuntimeError(f"alpha length mismatch: alpha={alpha.shape[0]} n={n}")
        order = np.argsort(alpha)[::-1]
        codes = np.asarray(sample["codes"], dtype=object)
        rows.append({
            "date": pd.Timestamp(sample["date"]),
            "codes": codes[order].tolist(),
            "alpha": alpha[order].astype(float).tolist(),
            "n_stocks": n,
        })
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"alpha {i + 1}/{len(samples)} | date={pd.Timestamp(sample['date']).date()} | time={(time.time() - t0) / 60:.1f}m", flush=True)
    return rows


def main():
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    dates = trading_dates(start, end)
    if not dates:
        raise ValueError(f"No trading dates found between {start.date()} and {end.date()}")

    cfg = build_v9_backtest_config()
    predictor = load_predictor(args, cfg)
    print(
        f"Generating inference Alpha: dates={len(dates)} "
        f"{dates[0].date()}~{dates[-1].date()} predictor={predictor.name}",
        flush=True,
    )
    samples = build_inference_samples(cfg, dates)
    samples.sort(key=lambda s: pd.Timestamp(s["date"]))
    rows = compute_rows(samples, predictor, args.progress_every)
    assert_alpha_rows_within_research(rows, context="V9 inference alpha")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "date": str(row["date"])}, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} Alpha rows: {out}", flush=True)


if __name__ == "__main__":
    main()
