"""Combine saved daily alpha-rank JSONL files into a rank ensemble."""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Combine alpha JSONL files by weighted rank/alpha mean")
    parser.add_argument("--inputs", required=True, help="Comma-separated JSONL paths")
    parser.add_argument("--weights", default=None, help="Comma-separated non-negative weights")
    parser.add_argument("--mode", default="rank_mean", choices=["rank_mean", "alpha_mean"])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["date"] = pd.Timestamp(row["date"])
            rows[row["date"]] = row
    return rows


def parse_weights(raw, n):
    if raw is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray([float(x.strip()) for x in raw.split(",") if x.strip()], dtype=np.float64)
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} != inputs length {n}")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights sum must be positive")
    return weights / total


def combine_for_date(rows_by_model, weights, mode):
    scores = {}
    cover = {}
    for model_rows, weight in zip(rows_by_model, weights):
        codes = list(model_rows.get("codes", []))
        alphas = list(model_rows.get("alpha", []))
        n = len(codes)
        if n == 0 or weight <= 0:
            continue
        if mode == "rank_mean":
            denom = max(n - 1, 1)
            values = [1.0 - rank / denom for rank in range(n)]
        else:
            arr = np.asarray(alphas, dtype=np.float64)
            finite = np.isfinite(arr)
            if np.count_nonzero(finite) >= 2:
                mean = float(np.nanmean(arr[finite]))
                std = float(np.nanstd(arr[finite])) + 1e-8
                arr = np.where(finite, arr, mean)
                values = ((arr - mean) / std).tolist()
            else:
                values = [0.0] * n
        for code, value in zip(codes, values):
            scores[code] = scores.get(code, 0.0) + float(weight) * float(value)
            cover[code] = cover.get(code, 0) + 1
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "codes": [code for code, _ in items],
        "alpha": [float(score) for _, score in items],
        "n_stocks": len(items),
        "avg_model_coverage": float(np.mean([cover[c] for c, _ in items])) if items else 0.0,
    }


def main():
    args = parse_args()
    input_paths = [p.strip() for p in args.inputs.split(",") if p.strip()]
    weights = parse_weights(args.weights, len(input_paths))
    maps = [load_rows(path) for path in input_paths]
    common_dates = sorted(set.intersection(*[set(m.keys()) for m in maps]))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for date in common_dates:
            combined = combine_for_date([m[date] for m in maps], weights, args.mode)
            row = {"date": str(date), **combined}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "inputs": input_paths,
        "weights": weights.tolist(),
        "mode": args.mode,
        "common_dates": len(common_dates),
        "output": str(out_path),
    }
    (out_path.parent / f"{out_path.stem}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
