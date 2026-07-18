"""Build top-k portfolio proposal rows from legacy OOF reranker datasets.

This is a proxy training set, not a replacement for realistic open-ledger
validation.  It turns each date in the 2018-2023 OOF reranker data into several
portfolio proposals and labels them with the average future target improvement
over the baseline top-alpha selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROPOSALS = [
    {"name": "baseline", "alpha": 1.0, "ma3": 0.0, "risk": 0.0, "industry": 0.0},
    {"name": "alpha_ma3", "alpha": 0.7, "ma3": 0.3, "risk": 0.0, "industry": 0.0},
    {"name": "low_risk", "alpha": 1.0, "ma3": 0.0, "risk": 0.15, "industry": 0.0},
    {"name": "alpha_ma3_low_risk", "alpha": 0.7, "ma3": 0.3, "risk": 0.10, "industry": 0.0},
    {"name": "industry_diverse", "alpha": 1.0, "ma3": 0.0, "risk": 0.05, "industry": 0.03},
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Path to reranker_dataset.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-candidates-per-day", type=int, default=160)
    parser.add_argument("--target-frac", type=float, default=0.006)
    parser.add_argument("--min-target-n", type=int, default=3)
    parser.add_argument("--target-col", default="future_target")
    return parser.parse_args(argv)


def finite(frame, column, default=0.0):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def risk_proxy(frame):
    cols = [c for c in frame.columns if c.startswith("risk_")]
    if not cols:
        return pd.Series(0.0, index=frame.index)
    values = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return values.abs().mean(axis=1)


def industry_crowd(frame):
    counts = frame["industry_id"].map(frame["industry_id"].value_counts())
    return counts.astype(float) / max(len(frame), 1)


def proposal_score(frame, proposal):
    alpha = finite(frame, "m0_alpha", 0.0)
    ma3 = finite(frame, "m0_alpha_ma3", 0.0)
    rank = 1.0 - finite(frame, "m0_rank_pct", 0.0)
    score = float(proposal["alpha"]) * alpha + 0.25 * rank
    score += float(proposal["ma3"]) * ma3
    score -= float(proposal["risk"]) * finite(frame, "risk_proxy", 0.0)
    score -= float(proposal["industry"]) * finite(frame, "industry_crowd", 0.0)
    return score


def aggregate(selected, group, proposal, baseline_codes, target_col):
    selected_codes = set(selected["code"].astype(str))
    baseline_set = set(baseline_codes)
    return {
        "proposal": proposal["name"],
        "portfolio_size": int(len(selected)),
        "mean_m0_alpha": float(finite(selected, "m0_alpha", 0.0).mean()),
        "mean_m0_rank_pct": float(finite(selected, "m0_rank_pct", 0.0).mean()),
        "max_m0_rank_pct": float(finite(selected, "m0_rank_pct", 0.0).max()),
        "mean_alpha_change_1d": float(finite(selected, "m0_alpha_change_1d", 0.0).mean()),
        "mean_alpha_change_3d": float(finite(selected, "m0_alpha_change_3d", 0.0).mean()),
        "mean_risk_proxy": float(finite(selected, "risk_proxy", 0.0).mean()),
        "max_industry_crowd": float(finite(selected, "industry_crowd", 0.0).max()),
        "industry_count": int(selected["industry_id"].nunique()) if "industry_id" in selected.columns else 0,
        "baseline_overlap": int(len(selected_codes & baseline_set)),
        "market_regime": float(finite(group, "market_regime", 0.0).mean()),
        "cfg_alpha": float(proposal["alpha"]),
        "cfg_ma3": float(proposal["ma3"]),
        "cfg_risk": float(proposal["risk"]),
        "cfg_industry": float(proposal["industry"]),
        "portfolio_utility": float(finite(selected, target_col, 0.0).mean()),
    }


def build_day(group, args):
    group = group.sort_values("candidate_position").head(int(args.max_candidates_per_day)).copy()
    if group.empty or args.target_col not in group.columns:
        return []
    group["risk_proxy"] = risk_proxy(group)
    group["industry_crowd"] = industry_crowd(group)
    group_size = int(group["group_size"].iloc[0]) if "group_size" in group.columns else len(group)
    target_n = max(int(group_size * float(args.target_frac)), int(args.min_target_n))
    target_n = min(target_n, len(group))
    baseline = group.sort_values("candidate_position").head(target_n)
    baseline_codes = baseline["code"].astype(str).tolist()
    baseline_utility = float(finite(baseline, args.target_col, 0.0).mean())
    rows = []
    for proposal in PROPOSALS:
        if proposal["name"] == "baseline":
            selected = baseline
        else:
            scored = group.copy()
            scored["proposal_score"] = proposal_score(scored, proposal)
            selected = scored.sort_values(["proposal_score", "candidate_position"], ascending=[False, True]).head(target_n)
        rec = {
            "split": str(group["split"].iloc[0]) if "split" in group.columns else "",
            "date": pd.Timestamp(group["date"].iloc[0]).strftime("%Y-%m-%d"),
            "label_available": 1,
            "baseline_utility": baseline_utility,
            "selected_codes": ";".join(selected["code"].astype(str).tolist()),
        }
        rec.update(aggregate(selected, group, proposal, baseline_codes, args.target_col))
        rec["utility_delta_vs_baseline"] = float(rec["portfolio_utility"] - baseline_utility)
        rows.append(rec)
    return rows


def load_inputs(paths):
    frames = []
    for raw in paths:
        frame = pd.read_parquet(raw)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    args = parse_args(argv)
    frame = load_inputs(args.input)
    rows = []
    for _, group in frame.groupby("date", sort=True):
        rows.extend(build_day(group, args))
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("OOF top-k dataset is empty")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output / "topk_oof_portfolio_policy_dataset.parquet", index=False)
    summary = {
        "inputs": [str(x) for x in args.input],
        "rows": int(len(out)),
        "dates": int(out["date"].nunique()),
        "splits": sorted(out["split"].astype(str).unique().tolist()),
        "proposals": sorted(out["proposal"].unique().tolist()),
        "mean_delta_by_proposal": out.groupby("proposal")["utility_delta_vs_baseline"].mean().to_dict(),
        "positive_rate_by_proposal": out.assign(pos=out["utility_delta_vs_baseline"] > 0)
        .groupby("proposal")["pos"]
        .mean()
        .to_dict(),
        "params": vars(args),
    }
    (output / "topk_oof_portfolio_policy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
