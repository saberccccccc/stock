"""Build pairwise candidate-vs-baseline replacement rows from a policy dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_FEATURES = (
    "candidate_rank_pct",
    "candidate_industry_top_share",
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
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--max-pairs-per-day", type=int, default=80)
    parser.add_argument("--base-cost", type=float, default=0.0025)
    parser.add_argument("--risk-cost", type=float, default=0.004)
    return parser.parse_args(argv)


def finite(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def risk_load(row):
    vol = max(finite(row.get("specific_vol_60d"), 0.0), 0.0)
    beta = finite(row.get("beta_60d"), 1.0)
    crowd = max(finite(row.get("candidate_industry_top_share"), 0.0), 0.0)
    return (
        np.clip(vol / 0.20, 0.0, 3.0) * 0.40
        + np.clip((beta - 1.0) / 1.0, 0.0, 2.0) * 0.30
        + np.clip(crowd / 0.50, 0.0, 2.0) * 0.30
    )


def build_pairwise(frame, args):
    rows = []
    eligible = frame[frame["eligible"].eq(1) & frame["label_available"].eq(1)].copy()
    for date, group in eligible.groupby("date", sort=True):
        group = group.sort_values("candidate_position")
        baseline = group[group["baseline_fill"].eq(1)].head(1)
        if baseline.empty:
            baseline = group.head(1)
        base = baseline.iloc[0]
        candidates = group.head(int(args.max_pairs_per_day))
        base_target = finite(base.get("exec_target_raw"))
        base_return = finite(base.get("exec_base_return"))
        base_risk = risk_load(base)
        if not np.isfinite(base_target):
            continue
        for _, cand in candidates.iterrows():
            cand_target = finite(cand.get("exec_target_raw"))
            if not np.isfinite(cand_target):
                continue
            cand_risk = risk_load(cand)
            risk_delta = cand_risk - base_risk
            net_edge = cand_target - base_target - float(args.base_cost) - float(args.risk_cost) * max(risk_delta, 0.0)
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
                "pair_net_edge": float(net_edge),
                "pair_raw_edge": float(cand_target - base_target),
                "pair_base_return_edge": float(finite(cand.get("exec_base_return")) - base_return)
                if np.isfinite(base_return)
                else np.nan,
                "pair_risk_delta": float(risk_delta),
            }
            for col in PAIR_FEATURES:
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
    pairwise = build_pairwise(frame, args)
    if pairwise.empty:
        raise ValueError("pairwise dataset is empty")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairwise.to_parquet(output / "pairwise_dataset.parquet", index=False)
    summary = {
        "input_dataset": str(args.input_dataset),
        "split_name": args.split_name,
        "rows": int(len(pairwise)),
        "dates": int(pairwise["date"].nunique()),
        "mean_pair_net_edge": float(pairwise["pair_net_edge"].mean()),
        "positive_rate": float((pairwise["pair_net_edge"] > 0).mean()),
        "features": [c for c in pairwise.columns if c.startswith(("cand_", "base_", "diff_"))],
    }
    (output / "pairwise_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
