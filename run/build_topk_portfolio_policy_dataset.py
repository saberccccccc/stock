"""Build top-k portfolio-policy rows from state-aware candidate datasets.

Each row is a full daily portfolio-construction proposal rather than a
candidate-vs-baseline pair.  The label is an executable path utility computed
from future open-price returns; features are signal-day portfolio exposures.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORIZON_WEIGHTS = {
    "exec_return_1d": 0.05,
    "exec_return_3d": 0.20,
    "exec_return_5d": 0.40,
    "exec_return_10d": 0.35,
}

SCORING_CONFIGS = [
    {
        "name": "baseline",
        "rank_weight": 0.0,
        "ret20_weight": 0.0,
        "specific_vol_weight": 0.0,
        "beta_weight": 0.0,
        "industry_weight": 0.0,
        "active_risk_weight": 0.0,
        "momentum_plateau_weight": 0.0,
    },
    {
        "name": "raw_top",
        "rank_weight": 1.0,
        "ret20_weight": 0.0,
        "specific_vol_weight": 0.0,
        "beta_weight": 0.0,
        "industry_weight": 0.0,
        "active_risk_weight": 0.0,
        "momentum_plateau_weight": 0.0,
    },
    {
        "name": "active_state",
        "rank_weight": 1.0,
        "ret20_weight": 0.0,
        "specific_vol_weight": 0.0,
        "beta_weight": 0.0,
        "industry_weight": 0.0,
        "active_risk_weight": 0.35,
        "momentum_plateau_weight": 0.0,
    },
    {
        "name": "active_state_plateau",
        "rank_weight": 1.0,
        "ret20_weight": 0.0,
        "specific_vol_weight": 0.0,
        "beta_weight": 0.0,
        "industry_weight": 0.0,
        "active_risk_weight": 0.35,
        "momentum_plateau_weight": 0.08,
    },
    {
        "name": "risk_mild",
        "rank_weight": 1.0,
        "ret20_weight": 0.0,
        "specific_vol_weight": 0.06,
        "beta_weight": 0.03,
        "industry_weight": 0.02,
        "active_risk_weight": 0.0,
        "momentum_plateau_weight": 0.0,
    },
    {
        "name": "crowd_momentum",
        "rank_weight": 1.0,
        "ret20_weight": 0.04,
        "specific_vol_weight": 0.02,
        "beta_weight": 0.0,
        "industry_weight": 0.05,
        "active_risk_weight": 0.0,
        "momentum_plateau_weight": 0.10,
    },
]

DIAG_COLUMNS = (
    "gross_weight",
    "market_mult",
    "portfolio_beta_60d",
    "portfolio_beta_per_gross_60d",
    "portfolio_specific_vol_60d",
    "avg_live_age",
    "turnover",
    "desired_turnover",
    "executed_turnover",
    "cost",
    "active_drawdown_trailing_return",
    "global_risk_pressure",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dataset", required=True)
    parser.add_argument("--diagnostics-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--max-replace", type=int, default=6)
    parser.add_argument("--candidate-end", type=int, default=120)
    parser.add_argument("--round-trip-cost", type=float, default=0.0017)
    parser.add_argument("--downside-cost", type=float, default=0.50)
    parser.add_argument("--delayed-cost", type=float, default=0.25)
    parser.add_argument("--blocked-buy-cost", type=float, default=0.10)
    return parser.parse_args(argv)


def finite_series(frame, column, default=0.0):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def positive(value):
    return max(float(value), 0.0) if np.isfinite(value) else 0.0


def path_return(frame):
    total = np.zeros(len(frame), dtype=np.float64)
    for col, weight in HORIZON_WEIGHTS.items():
        total += finite_series(frame, col, 0.0).to_numpy(dtype=np.float64) * float(weight)
    return pd.Series(total, index=frame.index, dtype=float)


def momentum_plateau(frame):
    ret20 = finite_series(frame, "ret_20d", 0.0).clip(lower=0.0)
    ret5 = finite_series(frame, "ret_5d", 0.0)
    drawdown = finite_series(frame, "drawdown_20d", 0.0).abs()
    stall = (-ret5).clip(lower=0.0) + (0.02 - ret5).clip(lower=0.0) * 0.5
    return (ret20 / 0.30).clip(upper=2.0) * (stall + 0.25 * drawdown)


def add_base_columns(frame):
    out = frame.copy()
    out["path_return"] = path_return(out)
    out["momentum_plateau_load"] = momentum_plateau(out)
    out["risk_load"] = (
        finite_series(out, "specific_vol_60d", 0.0).clip(lower=0.0) / 0.20 * 0.35
        + (finite_series(out, "beta_60d", 1.0) - 1.0).clip(lower=0.0) / 1.0 * 0.25
        + finite_series(out, "candidate_industry_top_share", 0.0).clip(lower=0.0) / 0.50 * 0.25
        + finite_series(out, "ret_20d", 0.0).clip(lower=0.0) / 0.30 * 0.15
    ).clip(0.0, 3.0)
    return out


def portfolio_utility(selected, args):
    if selected.empty:
        return np.nan
    path = finite_series(selected, "path_return", 0.0).mean()
    downside = finite_series(selected, "exec_max_downside", 0.0).clip(lower=0.0).mean()
    delayed_gap = (
        finite_series(selected, "path_return", 0.0)
        - finite_series(selected, "exec_delayed_return", 0.0)
    ).clip(lower=0.0).mean()
    blocked = finite_series(selected, "exec_blocked_buy", 0.0).clip(lower=0.0).mean()
    cost = float(args.round_trip_cost)
    return float(
        path
        - float(args.downside_cost) * downside
        - float(args.delayed_cost) * delayed_gap
        - float(args.blocked_buy_cost) * blocked
        - cost
    )


def aggregate_features(selected, group, baseline_codes, replace_count):
    if selected.empty:
        return {}
    top_industry_share = finite_series(selected, "candidate_industry_top_share", 0.0)
    weights = np.ones(len(selected), dtype=np.float64) / max(len(selected), 1)
    industry_hhi = float(np.sum(np.square(top_industry_share.to_numpy(dtype=np.float64) * weights)))
    selected_codes = set(selected["code"].astype(str))
    baseline_set = set(str(x) for x in baseline_codes)
    features = {
        "portfolio_size": int(len(selected)),
        "replace_count": int(replace_count),
        "replace_frac": float(replace_count / max(len(selected), 1)),
        "mean_rank_pct": float(finite_series(selected, "candidate_rank_pct", 0.0).mean()),
        "max_rank_pct": float(finite_series(selected, "candidate_rank_pct", 0.0).max()),
        "mean_ret_5d": float(finite_series(selected, "ret_5d", 0.0).mean()),
        "mean_ret_20d": float(finite_series(selected, "ret_20d", 0.0).mean()),
        "mean_beta_60d": float(finite_series(selected, "beta_60d", 1.0).mean()),
        "mean_specific_vol_60d": float(finite_series(selected, "specific_vol_60d", 0.0).mean()),
        "mean_drawdown_20d": float(finite_series(selected, "drawdown_20d", 0.0).mean()),
        "mean_money_ma20": float(finite_series(selected, "money_ma20", 0.0).mean()),
        "top_industry_share": float(top_industry_share.max()),
        "industry_hhi_proxy": industry_hhi,
        "mean_momentum_plateau": float(finite_series(selected, "momentum_plateau_load", 0.0).mean()),
        "mean_risk_load": float(finite_series(selected, "risk_load", 0.0).mean()),
        "kept_count": int(finite_series(selected, "is_kept", 0.0).sum()),
        "was_held_count": int(finite_series(selected, "was_held", 0.0).sum()),
        "baseline_overlap": int(len(selected_codes & baseline_set)),
        "global_us_hk_pressure": float(finite_series(group, "global_us_hk_pressure", 0.0).mean()),
        "global_defensive_pressure": float(finite_series(group, "global_defensive_pressure", 0.0).mean()),
        "global_hk_risk_pressure": float(finite_series(group, "global_hk_risk_pressure", 0.0).mean()),
    }
    for col in DIAG_COLUMNS:
        diag_col = f"diag_{col}"
        if diag_col in group.columns:
            features[diag_col] = float(finite_series(group, diag_col, 0.0).mean())
    return features


def score_candidates(candidates, group, cfg):
    rank_score = 1.0 - finite_series(candidates, "candidate_rank_pct", 0.0)
    active_drawdown = positive(-finite_series(group, "diag_active_drawdown_trailing_return", 0.0).mean())
    score = float(cfg["rank_weight"]) * rank_score
    score -= float(cfg["ret20_weight"]) * finite_series(candidates, "ret_20d", 0.0).clip(lower=0.0)
    score -= float(cfg["specific_vol_weight"]) * finite_series(candidates, "specific_vol_60d", 0.0).clip(lower=0.0)
    score -= float(cfg["beta_weight"]) * (finite_series(candidates, "beta_60d", 1.0) - 1.0).clip(lower=0.0)
    score -= float(cfg["industry_weight"]) * finite_series(candidates, "candidate_industry_top_share", 0.0).clip(lower=0.0)
    score -= float(cfg["active_risk_weight"]) * active_drawdown * finite_series(candidates, "risk_load", 0.0)
    score -= float(cfg["momentum_plateau_weight"]) * finite_series(candidates, "momentum_plateau_load", 0.0)
    return score


def build_for_day(group, args):
    group = group.sort_values("candidate_position").copy()
    valid = group[group["label_available"].eq(1)].copy()
    if valid.empty:
        return []
    selected_base = valid[valid["is_kept"].eq(1) | valid["baseline_fill"].eq(1)].copy()
    if selected_base.empty:
        selected_base = valid.head(int(valid["target_n"].iloc[0])).copy()
    baseline_codes = selected_base["code"].astype(str).tolist()
    protected = valid[valid["is_kept"].eq(1) | valid["protected_fill"].eq(1)].copy()
    protected_codes = set(protected["code"].astype(str))
    target_n = int(max(selected_base.shape[0], valid["target_n"].max()))
    replace_slots = max(target_n - len(protected), 0)
    replace_slots = min(replace_slots, int(args.max_replace))
    candidates = valid[
        valid["eligible"].eq(1)
        & ~valid["code"].astype(str).isin(protected_codes)
        & valid["candidate_position"].lt(int(args.candidate_end))
    ].copy()
    rows = []
    base_utility = portfolio_utility(selected_base, args)
    for cfg in SCORING_CONFIGS:
        if cfg["name"] == "baseline" or replace_slots <= 0 or candidates.empty:
            selected = selected_base.copy()
        else:
            ranked = candidates.copy()
            ranked["portfolio_policy_score"] = score_candidates(ranked, valid, cfg)
            chosen = ranked.sort_values(
                ["portfolio_policy_score", "candidate_position"],
                ascending=[False, True],
            ).head(replace_slots)
            selected = pd.concat([protected, chosen], axis=0, ignore_index=True)
            if len(selected) < target_n:
                missing = target_n - len(selected)
                fallback = selected_base[
                    ~selected_base["code"].astype(str).isin(set(selected["code"].astype(str)))
                ].head(missing)
                selected = pd.concat([selected, fallback], axis=0, ignore_index=True)
        utility = portfolio_utility(selected, args)
        features = aggregate_features(selected, valid, baseline_codes, replace_slots)
        rec = {
            "split": args.split_name,
            "date": pd.Timestamp(valid["date"].iloc[0]).strftime("%Y-%m-%d"),
            "proposal": cfg["name"],
            "label_available": int(np.isfinite(utility) and np.isfinite(base_utility)),
            "portfolio_utility": utility,
            "baseline_utility": base_utility,
            "utility_delta_vs_baseline": float(utility - base_utility)
            if np.isfinite(utility) and np.isfinite(base_utility)
            else np.nan,
            "selected_codes": ";".join(selected["code"].astype(str).tolist()),
        }
        rec.update({f"cfg_{k}": v for k, v in cfg.items() if k != "name"})
        rec.update(features)
        rows.append(rec)
    return rows


def build_dataset(frame, args):
    frame = add_base_columns(frame)
    rows = []
    for _, group in frame.groupby("date", sort=True):
        rows.extend(build_for_day(group, args))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[out["label_available"].eq(1)].reset_index(drop=True)
    return out


def load_diagnostics(path):
    if not path:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        return pd.DataFrame()
    keep = ["date"] + [col for col in DIAG_COLUMNS if col in frame.columns]
    frame = frame[keep].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.rename(columns={col: f"diag_{col}" for col in keep if col != "date"}).drop_duplicates(
        "date",
        keep="last",
    )


def merge_diagnostics(frame, diagnostics):
    if diagnostics.empty:
        return frame
    return frame.merge(diagnostics, on="date", how="left")


def main(argv=None):
    args = parse_args(argv)
    frame = pd.read_parquet(args.policy_dataset)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = merge_diagnostics(frame, load_diagnostics(args.diagnostics_csv))
    dataset = build_dataset(frame, args)
    if dataset.empty:
        raise ValueError("top-k portfolio policy dataset is empty")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output / "topk_portfolio_policy_dataset.parquet", index=False)
    summary = {
        "policy_dataset": str(args.policy_dataset),
        "diagnostics_csv": str(args.diagnostics_csv) if args.diagnostics_csv else None,
        "split_name": args.split_name,
        "rows": int(len(dataset)),
        "dates": int(dataset["date"].nunique()),
        "proposals": sorted(dataset["proposal"].unique().tolist()),
        "mean_utility_delta_by_proposal": dataset.groupby("proposal")[
            "utility_delta_vs_baseline"
        ].mean().to_dict(),
        "positive_delta_rate_by_proposal": dataset.assign(
            positive=dataset["utility_delta_vs_baseline"] > 0
        ).groupby("proposal")["positive"].mean().to_dict(),
        "params": vars(args),
    }
    (output / "topk_portfolio_policy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
