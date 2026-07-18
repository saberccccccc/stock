"""Attribute changed replacements in a state-aware ledger pilot."""

import argparse
from pathlib import Path

import pandas as pd


def _codes(value):
    if value is None or pd.isna(value):
        return set()
    return {item for item in str(value).split(",") if item}


def _holding_codes(value):
    if value is None or pd.isna(value):
        return set()
    return {
        part.split("=", 1)[0]
        for part in str(value).split(";")
        if "=" in part and part.split("=", 1)[0]
    }


def _one_file(directory, prefix, capital):
    capital_tag = int(float(capital) / 10000)
    matches = sorted(directory.glob(f"{prefix}_*pv{capital_tag:04d}w_target006_hold100.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected one {prefix} file for {capital} in {directory}, found {matches}")
    return matches[0]


def attribute_pair(root, split, capital, candidate_name):
    base_dir = root / "path_details" / "baseline" / split / "paths"
    cand_dir = root / "path_details" / candidate_name / split / "paths"
    base_diag = pd.read_csv(_one_file(base_dir, "diagnostics", capital))
    cand_diag = pd.read_csv(_one_file(cand_dir, "diagnostics", capital))
    base_ret = pd.read_csv(_one_file(base_dir, "returns", capital))
    cand_ret = pd.read_csv(_one_file(cand_dir, "returns", capital))
    for frame in (base_diag, cand_diag, base_ret, cand_ret):
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    diag = base_diag.merge(cand_diag, on="date", suffixes=("_base", "_candidate"), validate="one_to_one")
    returns = base_ret.merge(cand_ret, on="date", suffixes=("_base", "_candidate"), validate="one_to_one")
    frame = diag.merge(returns, on="date", validate="one_to_one")
    frame = frame.copy()
    frame["split"] = split
    frame["portfolio_value"] = capital
    rows = []
    for _, row in frame.iterrows():
        base_selected = _codes(row.get("selected_after_state_codes_base"))
        cand_selected = _codes(row.get("selected_after_state_codes_candidate"))
        base_actual = _holding_codes(row.get("holdings_base", ""))
        cand_actual = _holding_codes(row.get("holdings_candidate", ""))
        added = sorted(cand_selected - base_selected)
        removed = sorted(base_selected - cand_selected)
        actual_added = sorted(cand_actual - base_actual)
        actual_removed = sorted(base_actual - cand_actual)
        rows.append({
            "split": split,
            "portfolio_value": capital,
            "date": row["date"],
            "state_active": int(row.get("state_aware_selection_active_candidate", 0)),
            "state_changed": int(row.get("state_aware_selection_changed_candidate", 0)),
            "pressure": row.get("state_aware_selection_pressure_candidate"),
            "stress": row.get("state_aware_selection_stress_candidate", 0.0),
            "added_count": len(added),
            "removed_count": len(removed),
            "added_codes": ",".join(added),
            "removed_codes": ",".join(removed),
            "actual_added_count": len(actual_added),
            "actual_removed_count": len(actual_removed),
            "actual_added_codes": ",".join(actual_added),
            "actual_removed_codes": ",".join(actual_removed),
            "return_base": row["return_base"],
            "return_candidate": row["return_candidate"],
            "return_delta": row["return_candidate"] - row["return_base"],
            "beta_base": row.get("portfolio_beta_60d_base"),
            "beta_candidate": row.get("portfolio_beta_60d_candidate"),
            "beta_delta": row.get("portfolio_beta_60d_candidate") - row.get("portfolio_beta_60d_base"),
            "specific_vol_base": row.get("portfolio_specific_vol_60d_base"),
            "specific_vol_candidate": row.get("portfolio_specific_vol_60d_candidate"),
            "specific_vol_delta": row.get("portfolio_specific_vol_60d_candidate") - row.get("portfolio_specific_vol_60d_base"),
            "turnover_base": row.get("turnover_base"),
            "turnover_candidate": row.get("turnover_candidate"),
            "turnover_delta": row.get("turnover_candidate") - row.get("turnover_base"),
            "cost_base": row.get("cost_base"),
            "cost_candidate": row.get("cost_candidate"),
            "cost_delta": row.get("cost_candidate") - row.get("cost_base"),
        })
    return pd.DataFrame(rows)


def summarize(events):
    rows = []
    for (split, capital), group in events.groupby(["split", "portfolio_value"], sort=True):
        changed = group[(group["added_count"] > 0) | (group["removed_count"] > 0)]
        unchanged = group[(group["added_count"] == 0) & (group["removed_count"] == 0)]
        direct = group[group["state_changed"] > 0]
        path_only = changed[changed["state_changed"] == 0]
        rows.append({
            "split": split,
            "portfolio_value": capital,
            "days": len(group),
            "changed_days": len(changed),
            "state_active_days": int(group["state_active"].sum()),
            "state_changed_days": int(group["state_changed"].sum()),
            "path_only_divergence_days": len(path_only),
            "total_added": int(group["added_count"].sum()),
            "total_removed": int(group["removed_count"].sum()),
            "mean_return_delta_all": group["return_delta"].mean(),
            "mean_return_delta_changed": changed["return_delta"].mean() if not changed.empty else 0.0,
            "mean_return_delta_unchanged": unchanged["return_delta"].mean() if not unchanged.empty else 0.0,
            "positive_changed_day_rate": (changed["return_delta"] > 0).mean() if not changed.empty else 0.0,
            "mean_return_delta_direct_state": direct["return_delta"].mean() if not direct.empty else 0.0,
            "mean_return_delta_path_only": path_only["return_delta"].mean() if not path_only.empty else 0.0,
            "positive_direct_state_rate": (direct["return_delta"] > 0).mean() if not direct.empty else 0.0,
            "positive_path_only_rate": (path_only["return_delta"] > 0).mean() if not path_only.empty else 0.0,
            "mean_beta_delta": group["beta_delta"].mean(),
            "mean_specific_vol_delta": group["specific_vol_delta"].mean(),
            "mean_turnover_delta": group["turnover_delta"].mean(),
            "mean_cost_delta": group["cost_delta"].mean(),
        })
    return pd.DataFrame(rows)


def _pct(value):
    return f"{float(value) * 100:.4f}%"


def build_report(summary, events, output_csv):
    lines = [
        "# State-aware pilot replacement attribution (2026-07-15)",
        "",
        "This report explains changed desired holdings for the fixed `compact_v14_eq_rank` alpha. It uses only normal realistic open-ledger path details for 2024 Val and 2025 Test; it is diagnostic evidence, not a forward selection result.",
        "",
        "## Summary",
        "",
        "| Split | Capital | Days | Path-diverged days | Direct state changes | State active | Added | Removed | Delta direct state days | Delta path-only days | Positive direct | Positive path-only | Beta delta | Specific-vol delta | Turnover delta | Cost delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['split']} | {row['portfolio_value']/10000:.0f}W | {int(row['days'])} | {int(row['changed_days'])} | {int(row['state_changed_days'])} | {int(row['state_active_days'])} | {int(row['total_added'])} | {int(row['total_removed'])} | {_pct(row['mean_return_delta_direct_state'])} | {_pct(row['mean_return_delta_path_only'])} | {row['positive_direct_state_rate']:.1%} | {row['positive_path_only_rate']:.1%} | {row['mean_beta_delta']:.4f} | {row['mean_specific_vol_delta']:.4f} | {row['mean_turnover_delta']:.6f} | {row['mean_cost_delta']:.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The attribution separates direct state-triggered changes from later path-only divergence caused by retention carrying a changed portfolio forward. A positive mean on either group is useful evidence, but it is not sufficient by itself: beta, specific volatility, turnover, cost, and the full stress ledger remain part of the decision.",
        "",
        f"- Replacement event CSV: `{output_csv}`",
        "- No forward data was used.",
        "- Historical ST coverage remains a separate execution-coverage gate.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument(
        "--candidate-name",
        default="risk_rank_t035_p010",
        help="Candidate directory under experiment_root/path_details.",
    )
    args = parser.parse_args(argv)
    root = Path(args.experiment_root)
    frames = [
        attribute_pair(root, split, capital, args.candidate_name)
        for split in ("val_2024", "test_2025")
        for capital in (500000, 1000000)
    ]
    events = pd.concat(frames, ignore_index=True)
    summary = summarize(events)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)
    Path(args.output_report).write_text(
        build_report(summary, events, args.output_csv),
        encoding="utf-8",
    )
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_summary_csv}")
    print(f"wrote {args.output_report}")


if __name__ == "__main__":
    main()
