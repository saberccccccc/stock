"""Attribute pairwise policy changes against a baseline policy audit."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ledger_path_utility",
    "ledger_weighted_raw_edge",
    "pair_path_raw_edge",
    "pair_risk_delta",
    "pair_downside_delta",
    "pair_quick_fade_delta",
    "pair_rank_delta",
    "diag_active_drawdown_trailing_return",
    "diag_global_risk_pressure",
    "diag_portfolio_beta_60d",
    "diag_portfolio_specific_vol_60d",
    "diag_turnover",
    "diff_candidate_industry_top_share",
    "diff_top_industry_share",
    "diff_top_industry_hhi",
    "diff_ret_1d",
    "diff_ret_5d",
    "diff_ret_20d",
    "diff_vol_20d",
    "diff_vol_60d",
    "diff_drawdown_20d",
    "diff_beta_60d",
    "diff_specific_vol_60d",
    "diff_money_ma20",
    "cand_ret_20d",
    "cand_specific_vol_60d",
    "cand_beta_60d",
    "cand_top_industry_share",
    "cand_global_us_hk_pressure",
    "cand_global_hk_risk_pressure",
    "cand_global_defensive_pressure",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-audit", required=True)
    parser.add_argument("--candidate-audit", required=True)
    parser.add_argument("--label-dataset", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--active-threshold", type=float, default=0.0)
    parser.add_argument("--global-threshold", type=float, default=0.04)
    return parser.parse_args(argv)


def load_audit(path, prefix):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    keep = [
        "date",
        "policy_applied",
        "best_score",
        "threshold",
        "baseline_code",
        "chosen_code",
        "chosen_original_position",
        "baseline_position",
        "risk_guard_active",
        "risk_guard_penalty",
        "concentration_penalty_active",
        "concentration_penalty",
    ]
    existing = [col for col in keep if col in frame.columns]
    out = frame[existing].copy()
    return out.rename(columns={col: f"{prefix}_{col}" for col in existing if col != "date"})


def load_label(path):
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    for col in ("code", "baseline_code"):
        if col in frame.columns:
            frame[col] = frame[col].astype(str)
    return frame


def attach_label(events, label):
    keyed = label.set_index(["date", "code", "baseline_code"], drop=False)
    rows = []
    for row in events.itertuples(index=False):
        date = getattr(row, "date")
        code = getattr(row, "candidate_chosen_code")
        baseline_code = getattr(row, "candidate_baseline_code")
        rec = row._asdict()
        match = None
        if isinstance(code, str) and code and isinstance(baseline_code, str) and baseline_code:
            key = (date, code, baseline_code)
            if key in keyed.index:
                value = keyed.loc[key]
                match = value.iloc[0] if isinstance(value, pd.DataFrame) else value
        if match is None and isinstance(code, str) and code:
            same_day = label[(label["date"].eq(date)) & (label["code"].eq(code))]
            if not same_day.empty:
                match = same_day.sort_values("candidate_position").iloc[0]
        for col in FEATURE_COLUMNS:
            rec[col] = match.get(col, np.nan) if match is not None and col in match.index else np.nan
        rec["label_matched"] = int(match is not None)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_numeric(frame, group_col):
    rows = []
    metrics = [
        "ledger_path_utility",
        "ledger_weighted_raw_edge",
        "pair_path_raw_edge",
        "pair_risk_delta",
        "pair_downside_delta",
        "pair_quick_fade_delta",
        "diff_candidate_industry_top_share",
        "diff_top_industry_hhi",
        "diff_ret_20d",
        "diff_specific_vol_60d",
        "diag_active_drawdown_trailing_return",
        "diag_global_risk_pressure",
    ]
    for name, group in frame.groupby(group_col, dropna=False):
        rec = {group_col: name, "rows": int(len(group)), "matched": int(group["label_matched"].sum())}
        for metric in metrics:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce")
                rec[f"mean_{metric}"] = float(values.mean()) if values.notna().any() else np.nan
                rec[f"pos_rate_{metric}"] = float((values > 0).mean()) if values.notna().any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def add_buckets(frame, args):
    out = frame.copy()
    out["active_bucket"] = np.where(
        pd.to_numeric(out["diag_active_drawdown_trailing_return"], errors="coerce")
        <= float(args.active_threshold),
        "active_weak",
        "active_ok",
    )
    out["global_bucket"] = np.where(
        pd.to_numeric(out["diag_global_risk_pressure"], errors="coerce")
        >= float(args.global_threshold),
        "global_high",
        "global_low",
    )
    ret20 = pd.to_numeric(out["cand_ret_20d"], errors="coerce")
    out["cand_ret20_bucket"] = pd.cut(
        ret20,
        bins=[-np.inf, -0.05, 0.0, 0.10, 0.25, np.inf],
        labels=["ret20<-5", "-5..0", "0..10", "10..25", "ret20>25"],
    )
    svol = pd.to_numeric(out["cand_specific_vol_60d"], errors="coerce")
    out["cand_svol_bucket"] = pd.cut(
        svol,
        bins=[-np.inf, 0.04, 0.06, 0.08, 0.12, np.inf],
        labels=["svol<4", "4..6", "6..8", "8..12", "svol>12"],
    )
    diff_svol = pd.to_numeric(out["diff_specific_vol_60d"], errors="coerce")
    out["diff_svol_bucket"] = pd.cut(
        diff_svol,
        bins=[-np.inf, -0.02, 0.0, 0.02, 0.05, np.inf],
        labels=["diff<-2", "-2..0", "0..2", "2..5", "diff>5"],
    )
    return out


def write_markdown(events, summaries, output_path, args):
    lines = [
        f"# Pairwise Policy Change Attribution: {args.candidate_name}",
        "",
        f"- split: `{args.split}`",
        f"- baseline audit: `{args.baseline_audit}`",
        f"- candidate audit: `{args.candidate_audit}`",
        f"- label dataset: `{args.label_dataset}`",
        "",
        "## Overview",
        "",
        f"- days: {events['date'].nunique()}",
        f"- changed decisions: {int(events['decision_changed'].sum())}",
        f"- candidate applied days: {int(pd.to_numeric(events['candidate_policy_applied'], errors='coerce').fillna(0).sum())}",
        f"- matched labels: {int(events['label_matched'].sum())}",
        "",
    ]
    changed = events[events["decision_changed"].eq(1)]
    if not changed.empty:
        lines.extend([
            "## Changed Decision Averages",
            "",
            "| metric | value |",
            "|---|---:|",
        ])
        for metric in [
            "ledger_path_utility",
            "ledger_weighted_raw_edge",
            "pair_path_raw_edge",
            "pair_risk_delta",
            "pair_downside_delta",
            "pair_quick_fade_delta",
            "diff_ret_20d",
            "diff_specific_vol_60d",
            "diag_active_drawdown_trailing_return",
            "diag_global_risk_pressure",
        ]:
            values = pd.to_numeric(changed[metric], errors="coerce")
            lines.append(f"| mean {metric} | {values.mean():.6g} |")
    for name, table in summaries.items():
        lines.extend(["", f"## {name}", ""])
        if table.empty:
            lines.append("_empty_")
            continue
        lines.append(table.to_markdown(index=False))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_audit(args.baseline_audit, "baseline")
    candidate = load_audit(args.candidate_audit, "candidate")
    merged = baseline.merge(candidate, on="date", how="outer").sort_values("date")
    for col in ["baseline_chosen_code", "candidate_chosen_code", "baseline_baseline_code", "candidate_baseline_code"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("").astype(str)
    merged["decision_changed"] = (
        merged.get("baseline_chosen_code", "").astype(str)
        != merged.get("candidate_chosen_code", "").astype(str)
    ).astype(int)
    label = load_label(args.label_dataset)
    events = attach_label(merged, label)
    events = add_buckets(events, args)
    events.to_csv(output_dir / "policy_change_events.csv", index=False)
    summaries = {
        "By Decision Changed": summarize_numeric(events, "decision_changed"),
        "By Active Bucket": summarize_numeric(events[events["decision_changed"].eq(1)], "active_bucket"),
        "By Global Bucket": summarize_numeric(events[events["decision_changed"].eq(1)], "global_bucket"),
        "By Candidate Ret20": summarize_numeric(events[events["decision_changed"].eq(1)], "cand_ret20_bucket"),
        "By Candidate Specific Vol": summarize_numeric(events[events["decision_changed"].eq(1)], "cand_svol_bucket"),
        "By Diff Specific Vol": summarize_numeric(events[events["decision_changed"].eq(1)], "diff_svol_bucket"),
    }
    for name, table in summaries.items():
        file_name = name.lower().replace(" ", "_") + ".csv"
        table.to_csv(output_dir / file_name, index=False)
    write_markdown(events, summaries, output_dir / "policy_change_attribution.md", args)
    print(f"wrote attribution to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
