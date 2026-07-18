"""Build e15-primary conditional blends with an auxiliary alpha ranking."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import iter_aligned_alpha_rows, write_alpha_rows
from alpha.transforms import make_rank_alpha, percentile_map


def parse_args():
    parser = argparse.ArgumentParser(description="Build conditional alpha blends")
    parser.add_argument("--base-alpha", required=True)
    parser.add_argument("--aux-alpha", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    return parser.parse_args()


def _code_features(data_dir, codes, start_date, end_date):
    features = {}
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    for code in sorted(codes):
        path = Path(data_dir) / f"{code}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["trade_date", "close"])
        if frame.empty:
            continue
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.sort_values("trade_date")
        close = pd.to_numeric(frame["close"], errors="coerce")
        frame["prior5"] = close.shift(1) / close.shift(6) - 1.0
        frame["prior20"] = close.shift(1) / close.shift(21) - 1.0
        frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)]
        for row in frame.itertuples(index=False):
            features.setdefault(pd.Timestamp(row.trade_date), {})[code] = (
                float(row.prior5) if np.isfinite(row.prior5) else np.nan,
                float(row.prior20) if np.isfinite(row.prior20) else np.nan,
            )
    return features


VARIANTS = [
    {
        "name": "agree_top10_w20",
        "base_min": 0.90,
        "aux_min": 0.90,
        "aux_weight": 0.20,
        "max_gap": None,
        "prior5_floor": None,
        "prior20_floor": None,
    },
    {
        "name": "agree_top20_w20",
        "base_min": 0.80,
        "aux_min": 0.80,
        "aux_weight": 0.20,
        "max_gap": None,
        "prior5_floor": None,
        "prior20_floor": None,
    },
    {
        "name": "agree_top20_gap10_w25",
        "base_min": 0.80,
        "aux_min": 0.80,
        "aux_weight": 0.25,
        "max_gap": 0.10,
        "prior5_floor": None,
        "prior20_floor": None,
    },
    {
        "name": "agree_top20_mom20floor_w20",
        "base_min": 0.80,
        "aux_min": 0.80,
        "aux_weight": 0.20,
        "max_gap": None,
        "prior5_floor": None,
        "prior20_floor": -0.20,
    },
    {
        "name": "agree_top20_mom5_20floor_w20",
        "base_min": 0.80,
        "aux_min": 0.80,
        "aux_weight": 0.20,
        "max_gap": None,
        "prior5_floor": -0.10,
        "prior20_floor": -0.30,
    },
]


def _passes_variant(code, date, base_score, aux_score, features_by_date, variant):
    if base_score < variant["base_min"] or aux_score < variant["aux_min"]:
        return False
    max_gap = variant["max_gap"]
    if max_gap is not None and abs(base_score - aux_score) > max_gap:
        return False
    prior5, prior20 = features_by_date.get(pd.Timestamp(date), {}).get(
        code, (np.nan, np.nan)
    )
    if variant["prior5_floor"] is not None:
        if not np.isfinite(prior5) or prior5 < variant["prior5_floor"]:
            return False
    if variant["prior20_floor"] is not None:
        if not np.isfinite(prior20) or prior20 < variant["prior20_floor"]:
            return False
    return True


def build_variant_rows(base_path, aux_path, features_by_date, variant):
    rows = []
    total_gated = 0
    total_changed_top30 = 0
    for base, aux in iter_aligned_alpha_rows(base_path, aux_path):
        date = pd.Timestamp(base["date"])
        base_pct = percentile_map(base["codes"])
        aux_pct = percentile_map(aux["codes"])
        codes = list(base["codes"])
        scores = {}
        gated = 0
        for code in codes:
            base_score = float(base_pct.get(code, 0.0))
            aux_score = float(aux_pct.get(code, 0.0))
            score = base_score
            if _passes_variant(code, date, base_score, aux_score, features_by_date, variant):
                weight = float(variant["aux_weight"])
                score = (1.0 - weight) * base_score + weight * aux_score
                gated += 1
            scores[code] = score
        ordered = sorted(codes, key=lambda code: (-scores[code], code))
        total_gated += gated
        total_changed_top30 += int(ordered[:30] != codes[:30])
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "codes": ordered,
                "alpha": make_rank_alpha(len(ordered)),
                "n_stocks": len(ordered),
                "conditional_blend": {
                    "name": variant["name"],
                    "base_alpha": str(base_path),
                    "aux_alpha": str(aux_path),
                    "gated_count": gated,
                    "base_min": variant["base_min"],
                    "aux_min": variant["aux_min"],
                    "aux_weight": variant["aux_weight"],
                    "max_gap": variant["max_gap"],
                    "prior5_floor": variant["prior5_floor"],
                    "prior20_floor": variant["prior20_floor"],
                },
            }
        )
    return rows, {"gated_total": total_gated, "top30_changed_days": total_changed_top30}


def main():
    args = parse_args()
    base_path = Path(args.base_alpha)
    aux_path = Path(args.aux_alpha)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(iter_aligned_alpha_rows(base_path, aux_path))
    if not pairs:
        raise ValueError("No aligned alpha rows")
    all_codes = {code for base, aux in pairs for code in base.get("codes", []) + aux.get("codes", [])}
    dates = [pd.Timestamp(base["date"]) for base, _ in pairs]
    features = _code_features(args.data_dir, all_codes, min(dates), max(dates))

    summary = []
    for variant in VARIANTS:
        rows, stats = build_variant_rows(base_path, aux_path, features, variant)
        output = output_dir / f"{variant['name']}.jsonl"
        write_alpha_rows(output, rows)
        summary.append(
            {
                "name": variant["name"],
                "output": str(output),
                "dates": len(rows),
                **stats,
                **variant,
            }
        )
        print(f"wrote {output} gated_total={stats['gated_total']}")

    pd.DataFrame(summary).to_csv(output_dir / "conditional_blend_summary.csv", index=False)


if __name__ == "__main__":
    main()
