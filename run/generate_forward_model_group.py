"""Generate raw forward Alpha rankings for several frozen V9 checkpoints."""

import argparse
import gc
import json
import os
import sys
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
from run.forward_frozen_strategy import (
    assert_fundamental_data_current,
    assert_market_data_current,
    load_frozen_predictor,
)


def parse_model_specs(raw):
    specs = []
    for token in raw.split(","):
        name, separator, checkpoint = token.strip().partition("=")
        if not separator or not name or not checkpoint:
            raise ValueError(f"Invalid model spec: {token!r}")
        specs.append((name, checkpoint))
    if not specs:
        raise ValueError("At least one model spec is required")
    return specs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-specs", required=True)
    parser.add_argument("--data-dir", default="data/forward_raw")
    parser.add_argument("--start-date", default=str(FORWARD_START_DATE.date()))
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially generated alpha_raw.jsonl.partial file and only score missing dates.",
    )
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip a model when the existing output already covers every requested date.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--allow-research-dates",
        action="store_true",
        help=(
            "Allow arbitrary inference date ranges before the legacy forward "
            "boundary. Use only for split-style holdout scoring, not for "
            "training or checkpoint selection."
        ),
    )
    return parser.parse_args()


def ordered_alpha_row(sample, predictor):
    valid = np.ones(len(sample["codes"]), dtype=bool)
    alpha = predictor.predict_alpha(sample, valid, detect_regime(sample))
    order = np.argsort(alpha)[::-1]
    codes = np.asarray(sample["codes"], dtype=object)
    return {
        "date": pd.Timestamp(sample["date"]).strftime("%Y-%m-%d"),
        "codes": codes[order].tolist(),
        "alpha": np.asarray(alpha)[order].astype(float).tolist(),
        "n_stocks": int(len(codes)),
    }


def unique_samples_by_actual_date(samples):
    """Keep one inference sample per actual trading date."""
    unique = []
    seen = set()
    for sample in samples:
        key = pd.Timestamp(sample["date"]).strftime("%Y-%m-%d")
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def load_existing_rows(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["date"])[:10]] = row
    return rows


def write_rows(path, rows_by_date):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for date in sorted(rows_by_date):
            handle.write(json.dumps(rows_by_date[date], ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if start < FORWARD_START_DATE and not args.allow_research_dates:
        raise ValueError(f"Forward start must be on or after {FORWARD_START_DATE.date()}")
    if end < start:
        raise ValueError("end-date precedes start-date")
    print(f"Checking forward data freshness through {end.date()}...", flush=True)
    assert_market_data_current(args.data_dir, end)
    assert_fundamental_data_current()
    print("Forward data checks passed.", flush=True)

    query_start = start - pd.Timedelta(days=7)
    query_dates = [date.strftime("%Y-%m-%d") for date in pd.bdate_range(query_start, end)]

    from backtest.runtime import build_v9_backtest_config

    cfg = build_v9_backtest_config()
    cfg.data_dir = args.data_dir
    print(
        f"Building inference samples for {len(query_dates)} query dates "
        f"{query_dates[0]}~{query_dates[-1]}...",
        flush=True,
    )
    samples = build_inference_samples(cfg, query_dates)
    samples = [sample for sample in samples if pd.Timestamp(sample["date"]) >= start]
    samples = unique_samples_by_actual_date(samples)
    print(
        f"Built {len(samples)} unique samples "
        f"{pd.Timestamp(samples[0]['date']).date() if samples else 'none'}~"
        f"{pd.Timestamp(samples[-1]['date']).date() if samples else 'none'}",
        flush=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_dates = [
        pd.Timestamp(sample["date"]).strftime("%Y-%m-%d")
        for sample in samples
    ]
    for name, checkpoint in parse_model_specs(args.model_specs):
        output = output_dir / name / "alpha_raw.jsonl"
        partial = output.with_suffix(output.suffix + ".partial")
        rows_by_date = {}
        if args.resume:
            rows_by_date.update(load_existing_rows(output))
            rows_by_date.update(load_existing_rows(partial))
        if args.skip_complete and requested_dates and set(requested_dates).issubset(rows_by_date):
            write_rows(output, rows_by_date)
            if partial.exists():
                partial.unlink()
            print(
                f"Skipped {name}: existing output covers "
                f"{requested_dates[0]}~{requested_dates[-1]}",
                flush=True,
            )
            continue

        pending = [
            sample
            for sample in samples
            if pd.Timestamp(sample["date"]).strftime("%Y-%m-%d") not in rows_by_date
        ]
        predictor = load_frozen_predictor(checkpoint, args.device, "raw")
        partial.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("a", encoding="utf-8") as handle:
            for index, sample in enumerate(pending, start=1):
                row = ordered_alpha_row(sample, predictor)
                rows_by_date[row["date"]] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if index == 1 or index == len(pending) or index % max(args.progress_every, 1) == 0:
                    print(
                        f"{name}: {index}/{len(pending)} scored {row['date']}",
                        flush=True,
                    )
        write_rows(output, rows_by_date)
        if partial.exists():
            partial.unlink()
        dates = sorted(rows_by_date)
        print(
            f"Saved {name}: {len(dates)} dates "
            f"{dates[0] if dates else 'none'}~{dates[-1] if dates else 'none'}",
            flush=True,
        )
        del predictor, rows_by_date
        gc.collect()


if __name__ == "__main__":
    main()
