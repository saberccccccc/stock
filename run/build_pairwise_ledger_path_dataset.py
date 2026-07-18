"""Build ledger-weighted path labels for replacement policy training.

This V3 dataset uses actual open-ledger diagnostics from a baseline strategy.
For each signal date, it maps to the next execution diagnostic date, finds the
baseline fill that was actually held, and scores candidate replacements by the
portfolio-weighted future path delta.
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

DIAG_FEATURES = (
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

RETURN_HORIZONS = (1, 3, 5, 10)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dataset", required=True)
    parser.add_argument("--diagnostics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--max-pairs-per-day", type=int, default=80)
    parser.add_argument("--horizon-weights", default="0.05,0.20,0.40,0.35")
    parser.add_argument("--base-cost-bps", type=float, default=25.0)
    parser.add_argument("--risk-cost", type=float, default=0.0025)
    parser.add_argument("--downside-cost", type=float, default=0.40)
    parser.add_argument("--quick-fade-cost", type=float, default=0.15)
    parser.add_argument("--beta-cost", type=float, default=0.0)
    parser.add_argument("--specific-vol-cost", type=float, default=0.0)
    parser.add_argument("--industry-concentration-cost", type=float, default=0.0)
    parser.add_argument("--active-drawdown-cost", type=float, default=0.0)
    parser.add_argument("--lag1-decay-cost", type=float, default=0.0)
    parser.add_argument("--momentum-plateau-cost", type=float, default=0.0)
    parser.add_argument("--min-baseline-weight", type=float, default=0.002)
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


def parse_holdings(text):
    out = {}
    if not isinstance(text, str) or not text:
        return out
    for part in text.split(";"):
        if "=" not in part:
            continue
        code, weight = part.split("=", 1)
        value = finite(weight)
        if np.isfinite(value):
            out[str(code).strip()] = float(value)
    return out


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
    return float(max(0.0, r1 - max(r5, r10)))


def lag1_decay(row):
    """Penalty input for signals that lose edge after one-day delayed execution.

    The path dataset currently stores direct execution returns.  Some future
    datasets may also include lagged execution return columns; when absent this
    component remains zero so old experiments are unaffected.
    """
    r5 = finite(row.get("exec_return_5d"))
    lag5 = finite(row.get("exec_lag1_return_5d"))
    if not all(np.isfinite(x) for x in (r5, lag5)):
        return 0.0
    return float(max(0.0, r5 - lag5))


def momentum_plateau_load(row):
    ret20 = max(finite(row.get("ret_20d"), 0.0), 0.0)
    ret5 = finite(row.get("ret_5d"), 0.0)
    drawdown20 = max(finite(row.get("drawdown_20d"), 0.0), 0.0)
    if ret20 <= 0:
        return 0.0
    stall = max(0.0, -ret5) + 0.5 * max(0.0, 0.02 - ret5)
    return float(np.clip(ret20 / 0.30, 0.0, 2.0) * (stall + 0.25 * drawdown20))


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


def positive_delta(cand, base, column):
    cval = finite(cand.get(column))
    bval = finite(base.get(column))
    if not all(np.isfinite(x) for x in (cval, bval)):
        return 0.0
    return float(max(cval - bval, 0.0))


def industry_concentration_delta(cand, base):
    values = [
        positive_delta(cand, base, "candidate_industry_top_share"),
        positive_delta(cand, base, "top_industry_share"),
        positive_delta(cand, base, "top_industry_hhi"),
    ]
    return float(max(values))


def load_diag(path):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["holdings_map"] = frame["holdings"].map(parse_holdings)
    keep = ["date", "holdings_map"] + [c for c in DIAG_FEATURES if c in frame.columns]
    return frame[keep].sort_values("date").drop_duplicates("date", keep="last")


def next_diag_lookup(diag_dates):
    diag_dates = np.asarray(pd.to_datetime(diag_dates), dtype="datetime64[ns]")

    def lookup(signal_date):
        value = np.datetime64(pd.Timestamp(signal_date).to_datetime64())
        pos = int(np.searchsorted(diag_dates, value, side="right"))
        if pos >= len(diag_dates):
            return None
        return pd.Timestamp(diag_dates[pos]).normalize()

    return lookup


def build_dataset(policy, diag, args):
    weights = parse_weights(args.horizon_weights)
    diag_by_date = {pd.Timestamp(row.date).normalize(): row for row in diag.itertuples(index=False)}
    lookup_exec_date = next_diag_lookup(diag["date"])
    rows = []
    eligible = policy[policy["eligible"].eq(1) & policy["label_available"].eq(1)].copy()
    for signal_date, group in eligible.groupby("date", sort=True):
        exec_date = lookup_exec_date(signal_date)
        if exec_date is None or exec_date not in diag_by_date:
            continue
        diag_row = diag_by_date[exec_date]
        holdings = getattr(diag_row, "holdings_map")
        if not holdings:
            continue
        group = group.sort_values("candidate_position")
        baseline = group[group["baseline_fill"].eq(1)].head(1)
        if baseline.empty:
            baseline = group.head(1)
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        base_code = str(base["code"])
        base_weight = finite(holdings.get(base_code), 0.0)
        if base_weight < float(args.min_baseline_weight):
            continue
        base_path = path_return(base, weights)
        base_down = finite(base.get("exec_max_downside"))
        base_risk = risk_load(base)
        base_quick = quick_fade(base)
        if not all(np.isfinite(x) for x in (base_path, base_down, base_risk, base_quick)):
            continue
        for _, cand in group.head(int(args.max_pairs_per_day)).iterrows():
            cand_code = str(cand["code"])
            cand_path = path_return(cand, weights)
            cand_down = finite(cand.get("exec_max_downside"))
            cand_risk = risk_load(cand)
            cand_quick = quick_fade(cand)
            if not all(np.isfinite(x) for x in (cand_path, cand_down, cand_risk, cand_quick)):
                continue
            raw_edge = cand_path - base_path
            weighted_edge = base_weight * raw_edge
            risk_delta = cand_risk - base_risk
            downside_delta = cand_down - base_down
            quick_delta = cand_quick - base_quick
            already_held_weight = finite(holdings.get(cand_code), 0.0)
            rank_delta = max(0.0, finite(cand.get("candidate_position"), 0.0) - finite(base.get("candidate_position"), 0.0))
            turnover_weight = 0.0 if already_held_weight > 0 else base_weight
            cost = turnover_weight * float(args.base_cost_bps) / 10000.0
            risk_penalty = base_weight * float(args.risk_cost) * max(risk_delta, 0.0)
            downside_penalty = base_weight * float(args.downside_cost) * max(downside_delta, 0.0)
            quick_fade_penalty = base_weight * float(args.quick_fade_cost) * max(quick_delta, 0.0)
            rank_penalty = base_weight * 0.0010 * rank_delta / 50.0
            beta_delta = positive_delta(cand, base, "beta_60d")
            specific_vol_delta = positive_delta(cand, base, "specific_vol_60d")
            industry_delta = industry_concentration_delta(cand, base)
            active_drawdown = max(-finite(getattr(diag_row, "active_drawdown_trailing_return", 0.0), 0.0), 0.0)
            lag1_delta = lag1_decay(cand) - lag1_decay(base)
            plateau_delta = momentum_plateau_load(cand) - momentum_plateau_load(base)
            beta_penalty = base_weight * float(args.beta_cost) * beta_delta
            specific_vol_penalty = base_weight * float(args.specific_vol_cost) * specific_vol_delta
            industry_penalty = base_weight * float(args.industry_concentration_cost) * industry_delta
            active_drawdown_penalty = (
                base_weight
                * float(args.active_drawdown_cost)
                * active_drawdown
                * max(risk_delta, 0.0)
            )
            lag1_decay_penalty = base_weight * float(args.lag1_decay_cost) * max(lag1_delta, 0.0)
            momentum_plateau_penalty = (
                base_weight * float(args.momentum_plateau_cost) * max(plateau_delta, 0.0)
            )
            utility = weighted_edge - cost - risk_penalty - downside_penalty - quick_fade_penalty - rank_penalty
            risk_adjusted_utility = (
                utility
                - beta_penalty
                - specific_vol_penalty
                - industry_penalty
                - active_drawdown_penalty
                - lag1_decay_penalty
                - momentum_plateau_penalty
            )
            rec = {
                "split": args.split_name,
                "date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                "execution_date": exec_date.strftime("%Y-%m-%d"),
                "code": cand_code,
                "baseline_code": base_code,
                "candidate_position": int(cand["candidate_position"]),
                "baseline_position": int(base["candidate_position"]),
                "is_baseline": int(cand_code == base_code),
                "label_available": 1,
                "eligible": 1,
                "ledger_path_utility": float(utility),
                "ledger_risk_adjusted_utility": float(risk_adjusted_utility),
                "ledger_weighted_raw_edge": float(weighted_edge),
                "pair_path_raw_edge": float(raw_edge),
                "baseline_weight": float(base_weight),
                "candidate_already_held_weight": float(already_held_weight),
                "ledger_cost": float(cost),
                "ledger_cost_penalty": float(cost),
                "ledger_risk_penalty": float(risk_penalty),
                "ledger_downside_penalty": float(downside_penalty),
                "ledger_quick_fade_penalty": float(quick_fade_penalty),
                "ledger_rank_penalty": float(rank_penalty),
                "ledger_beta_penalty": float(beta_penalty),
                "ledger_specific_vol_penalty": float(specific_vol_penalty),
                "ledger_industry_concentration_penalty": float(industry_penalty),
                "ledger_active_drawdown_penalty": float(active_drawdown_penalty),
                "ledger_lag1_decay_penalty": float(lag1_decay_penalty),
                "ledger_momentum_plateau_penalty": float(momentum_plateau_penalty),
                "pair_risk_delta": float(risk_delta),
                "pair_downside_delta": float(downside_delta),
                "pair_quick_fade_delta": float(quick_delta),
                "pair_rank_delta": float(rank_delta),
                "pair_beta_delta": float(beta_delta),
                "pair_specific_vol_delta": float(specific_vol_delta),
                "pair_industry_concentration_delta": float(industry_delta),
                "pair_lag1_decay_delta": float(lag1_delta),
                "pair_momentum_plateau_delta": float(plateau_delta),
            }
            for col in DIAG_FEATURES:
                rec[f"diag_{col}"] = finite(getattr(diag_row, col, np.nan))
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
    policy = pd.read_parquet(args.policy_dataset)
    policy["date"] = pd.to_datetime(policy["date"]).dt.normalize()
    diag = load_diag(args.diagnostics_csv)
    dataset = build_dataset(policy, diag, args)
    if dataset.empty:
        raise ValueError("ledger path dataset is empty")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output / "pairwise_ledger_path_dataset.parquet", index=False)
    summary = {
        "policy_dataset": str(args.policy_dataset),
        "diagnostics_csv": str(args.diagnostics_csv),
        "split_name": args.split_name,
        "rows": int(len(dataset)),
        "dates": int(dataset["date"].nunique()),
        "mean_ledger_path_utility": float(dataset["ledger_path_utility"].mean()),
        "positive_rate": float((dataset["ledger_path_utility"] > 0).mean()),
        "mean_ledger_risk_adjusted_utility": float(dataset["ledger_risk_adjusted_utility"].mean()),
        "risk_adjusted_positive_rate": float((dataset["ledger_risk_adjusted_utility"] > 0).mean()),
        "mean_weighted_raw_edge": float(dataset["ledger_weighted_raw_edge"].mean()),
        "mean_baseline_weight": float(dataset["baseline_weight"].mean()),
        "params": vars(args),
    }
    (output / "pairwise_ledger_path_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
