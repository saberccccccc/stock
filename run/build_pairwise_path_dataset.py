"""Build path-aware pairwise replacement rows from a policy dataset.

V1 pairwise labels asked whether a candidate was better than the baseline fill
on a single executable target.  This V2 dataset asks a more portfolio-like
question: does the replacement improve the expected holding path after costs,
downside, risk-load, and unnecessary turnover penalties?
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = (
    "candidate_rank_pct",
    "candidate_industry_top_share",
    "top_industry_share",
    "top_industry_hhi",
    "global_us_hk_pressure",
    "global_defensive_pressure",
    "global_hk_risk_pressure",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "vol_60d",
    "drawdown_20d",
    "beta_60d",
    "specific_vol_60d",
    "money_ma20",
    "was_held",
    "holding_age",
    "protected_fill",
    "baseline_fill",
)

RETURN_HORIZONS = (1, 3, 5, 10)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--max-pairs-per-day", type=int, default=80)
    parser.add_argument("--base-cost", type=float, default=0.0025)
    parser.add_argument("--risk-cost", type=float, default=0.004)
    parser.add_argument("--downside-cost", type=float, default=0.65)
    parser.add_argument("--turnover-cost", type=float, default=0.0015)
    parser.add_argument("--deep-rank-cost", type=float, default=0.0020)
    parser.add_argument("--quick-fade-cost", type=float, default=0.25)
    parser.add_argument("--horizon-weights", default="0.05,0.20,0.40,0.35")
    return parser.parse_args(argv)


def finite(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def parse_weights(text):
    weights = np.asarray([float(x.strip()) for x in str(text).split(",") if x.strip()], dtype=np.float64)
    if len(weights) != len(RETURN_HORIZONS):
        raise ValueError(f"--horizon-weights must have {len(RETURN_HORIZONS)} values")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("--horizon-weights must sum to a positive value")
    return weights / total


def risk_load(row):
    vol = max(finite(row.get("specific_vol_60d"), 0.0), 0.0)
    beta = finite(row.get("beta_60d"), 1.0)
    crowd = max(finite(row.get("candidate_industry_top_share"), 0.0), 0.0)
    ret20 = max(finite(row.get("ret_20d"), 0.0), 0.0)
    return (
        np.clip(vol / 0.20, 0.0, 3.0) * 0.35
        + np.clip((beta - 1.0) / 1.0, 0.0, 2.0) * 0.25
        + np.clip(crowd / 0.50, 0.0, 2.0) * 0.25
        + np.clip(ret20 / 0.30, 0.0, 2.0) * 0.15
    )


def path_return(row, weights):
    values = []
    for horizon in RETURN_HORIZONS:
        value = finite(row.get(f"exec_return_{horizon}d"))
        if not np.isfinite(value):
            return np.nan
        values.append(value)
    return float(np.dot(np.asarray(values, dtype=np.float64), weights))


def quick_fade(row):
    r1 = finite(row.get("exec_return_1d"))
    r5 = finite(row.get("exec_return_5d"))
    r10 = finite(row.get("exec_return_10d"))
    if not all(np.isfinite(x) for x in (r1, r5, r10)):
        return np.nan
    # Penalize candidates that look good immediately but lose that edge over the holding path.
    return float(max(0.0, r1 - max(r5, r10)))


def build_path_pairwise(frame, args):
    weights = parse_weights(args.horizon_weights)
    rows = []
    eligible = frame[frame["eligible"].eq(1) & frame["label_available"].eq(1)].copy()
    for date, group in eligible.groupby("date", sort=True):
        group = group.sort_values("candidate_position")
        baseline = group[group["baseline_fill"].eq(1)].head(1)
        if baseline.empty:
            baseline = group.head(1)
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        base_path = path_return(base, weights)
        base_down = finite(base.get("exec_max_downside"))
        base_risk = risk_load(base)
        base_quick_fade = quick_fade(base)
        if not all(np.isfinite(x) for x in (base_path, base_down, base_risk, base_quick_fade)):
            continue
        candidates = group.head(int(args.max_pairs_per_day))
        for _, cand in candidates.iterrows():
            cand_path = path_return(cand, weights)
            cand_down = finite(cand.get("exec_max_downside"))
            cand_risk = risk_load(cand)
            cand_quick_fade = quick_fade(cand)
            if not all(np.isfinite(x) for x in (cand_path, cand_down, cand_risk, cand_quick_fade)):
                continue
            risk_delta = cand_risk - base_risk
            downside_delta = cand_down - base_down
            rank_delta = max(0.0, finite(cand.get("candidate_position"), 0.0) - finite(base.get("candidate_position"), 0.0))
            turnover_delta = max(0.0, finite(base.get("was_held"), 0.0) - finite(cand.get("was_held"), 0.0))
            quick_fade_delta = cand_quick_fade - base_quick_fade
            raw_edge = cand_path - base_path
            utility = (
                raw_edge
                - float(args.base_cost)
                - float(args.risk_cost) * max(risk_delta, 0.0)
                - float(args.downside_cost) * max(downside_delta, 0.0)
                - float(args.turnover_cost) * turnover_delta
                - float(args.deep_rank_cost) * rank_delta / 50.0
                - float(args.quick_fade_cost) * max(quick_fade_delta, 0.0)
            )
            rec = {
                "split": args.split_name,
                "date": date,
                "code": cand["code"],
                "baseline_code": base["code"],
                "candidate_position": int(cand["candidate_position"]),
                "baseline_position": int(base["candidate_position"]),
                "is_baseline": int(cand["code"] == base["code"]),
                "label_available": 1,
                "eligible": 1,
                "pair_path_utility": float(utility),
                "pair_path_raw_edge": float(raw_edge),
                "pair_path_return": float(cand_path),
                "base_path_return": float(base_path),
                "pair_downside_delta": float(downside_delta),
                "pair_risk_delta": float(risk_delta),
                "pair_turnover_delta": float(turnover_delta),
                "pair_quick_fade_delta": float(quick_fade_delta),
                "pair_rank_delta": float(rank_delta),
            }
            for col in FEATURES:
                cval = finite(cand.get(col))
                bval = finite(base.get(col))
                rec[f"cand_{col}"] = cval
                rec[f"base_{col}"] = bval
                rec[f"diff_{col}"] = cval - bval if np.isfinite(cval) and np.isfinite(bval) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse_args(argv)
    frame = pd.read_parquet(args.input_dataset)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    pairwise = build_path_pairwise(frame, args)
    if pairwise.empty:
        raise ValueError("path pairwise dataset is empty")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairwise.to_parquet(output / "pairwise_path_dataset.parquet", index=False)
    summary = {
        "input_dataset": str(args.input_dataset),
        "split_name": args.split_name,
        "rows": int(len(pairwise)),
        "dates": int(pairwise["date"].nunique()),
        "mean_pair_path_utility": float(pairwise["pair_path_utility"].mean()),
        "positive_rate": float((pairwise["pair_path_utility"] > 0).mean()),
        "mean_raw_edge": float(pairwise["pair_path_raw_edge"].mean()),
        "features": [c for c in pairwise.columns if c.startswith(("cand_", "base_", "diff_"))],
        "params": {
            "base_cost": args.base_cost,
            "risk_cost": args.risk_cost,
            "downside_cost": args.downside_cost,
            "turnover_cost": args.turnover_cost,
            "deep_rank_cost": args.deep_rank_cost,
            "quick_fade_cost": args.quick_fade_cost,
            "horizon_weights": args.horizon_weights,
        },
    }
    (output / "pairwise_path_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
