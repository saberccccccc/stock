"""Summarize a state-aware portfolio-construction pilot.

The script compares two proposals that already ran through the same realistic
open-price ledger. It refuses to produce a report when the mandatory date
fields are absent or incomplete.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_DATE_FIELDS = (
    "signal_start",
    "signal_end",
    "backtest_start",
    "backtest_end",
)
KEY_FIELDS = (
    "stress",
    "portfolio_value",
    "target_frac",
    "hold_frac",
    "rebalance_band",
    "max_new_names",
)
METRICS = (
    "ann",
    "sharpe",
    "mdd",
    "avg_turnover",
    "avg_executed_turnover",
    "avg_unfilled_turnover",
    "total_cost",
    "blocked_buy",
    "avg_portfolio_beta_60d",
    "avg_portfolio_specific_vol_60d",
)


def _read_summary(path):
    frame = pd.read_csv(path)
    missing = [field for field in REQUIRED_DATE_FIELDS if field not in frame]
    if missing:
        raise ValueError(f"{path} is missing mandatory date fields: {missing}")
    for field in REQUIRED_DATE_FIELDS:
        if frame[field].isna().any() or (frame[field].astype(str).str.strip() == "").any():
            raise ValueError(f"{path} has empty mandatory date field: {field}")
    missing_keys = [field for field in KEY_FIELDS if field not in frame]
    if missing_keys:
        raise ValueError(f"{path} is missing comparison keys: {missing_keys}")
    return frame


def compare_split(root, split, baseline_dir, candidate_dir):
    baseline = _read_summary(root / baseline_dir / split / "open_price_ledger_param_sweep_summary.csv")
    candidate = _read_summary(root / candidate_dir / split / "open_price_ledger_param_sweep_summary.csv")
    candidate = candidate.copy()
    optional_defaults = {
        "state_aware_selection_risk_delta_threshold": 0.15,
        "state_aware_selection_suppressed_days": 0,
        "total_state_aware_selection_suppressed": 0,
        "avg_state_aware_selection_suppressed_risk_delta": 0.0,
    }
    for field, default in optional_defaults.items():
        if field not in candidate.columns:
            candidate[field] = default
    baseline_cols = list(KEY_FIELDS) + list(METRICS) + list(REQUIRED_DATE_FIELDS)
    candidate_cols = list(KEY_FIELDS) + list(METRICS) + [
        "state_aware_selection_mode",
        "state_aware_selection_pressure_col",
        "state_aware_selection_pressure_threshold",
        "state_aware_selection_pressure_width",
        "state_aware_selection_rank_penalty",
        "state_aware_selection_min_stress",
        "state_aware_selection_top_frac",
        "state_aware_selection_crowd_scale",
        "state_aware_selection_momentum_weight",
        "state_aware_selection_beta_weight",
        "state_aware_selection_vol_weight",
        "state_aware_selection_industry_weight",
        "state_aware_selection_risk_delta_threshold",
        "state_aware_selection_days",
        "state_aware_selection_changed_days",
        "state_aware_selection_suppressed_days",
        "total_state_aware_selection_suppressed",
        "avg_state_aware_selection_suppressed_risk_delta",
        "avg_state_aware_selection_stress",
        "avg_state_aware_selection_new_risk",
        "avg_state_aware_selection_top_industry_share",
    ]
    merged = baseline[baseline_cols].merge(
        candidate[candidate_cols],
        on=list(KEY_FIELDS),
        suffixes=("_base", "_candidate"),
        validate="one_to_one",
    )
    merged.insert(0, "split", split)
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[f"{metric}_candidate"] - merged[f"{metric}_base"]
    return merged


def _fmt(value, digits=3):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _table(frame, columns, headers=None):
    headers = headers or columns
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column.startswith("delta_") or column in {"ann_base", "ann_candidate"}:
                if column == "delta_mdd":
                    values.append(f"{float(value) * 100:.2f}%")
                else:
                    values.append(_fmt(value, 2))
            elif column in {"sharpe_base", "sharpe_candidate"}:
                values.append(_fmt(value, 3))
            elif column in {"mdd_base", "mdd_candidate"}:
                values.append(f"{float(value) * 100:.2f}%")
            elif column == "portfolio_value":
                values.append(f"{float(value) / 10000:.0f}W")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(frame, config_path, root, baseline_dir, candidate_dir, output_csv):
    candidate_name = candidate_dir
    sharpe_wins = int((frame["delta_sharpe"] >= 0).sum())
    ann_wins = int((frame["delta_ann"] >= 0).sum())
    mdd_not_worse = int((frame["delta_mdd"] <= 0).sum())
    total = len(frame)
    date_rows = frame[["split", *REQUIRED_DATE_FIELDS]].drop_duplicates().sort_values("split")
    summary_rows = []
    for split in ("val_2024", "test_2025"):
        part = frame[frame["split"] == split]
        summary_rows.append({
            "split": split,
            "Sharpe >= baseline": f"{int((part['delta_sharpe'] >= 0).sum())}/{len(part)}",
            "ann >= baseline": f"{int((part['delta_ann'] >= 0).sum())}/{len(part)}",
            "MDD no worse": f"{int((part['delta_mdd'] <= 0).sum())}/{len(part)}",
            "normal_1m_delta_sharpe": _fmt(part[(part.portfolio_value == 1000000) & (part.stress == "normal")]["delta_sharpe"].iloc[0], 3),
            "normal_1m_delta_ann": f"{part[(part.portfolio_value == 1000000) & (part.stress == 'normal')]['delta_ann'].iloc[0]:.2f}pct",
        })

    preferred = frame[(frame["portfolio_value"] == 1000000) & (frame["stress"] == "normal")].copy()
    preferred = preferred.sort_values("split")
    table_cols = [
        "split", "stress", "portfolio_value", "ann_base", "ann_candidate", "delta_ann",
        "sharpe_base", "sharpe_candidate", "delta_sharpe", "mdd_base", "mdd_candidate",
        "delta_mdd", "avg_executed_turnover_base", "avg_executed_turnover_candidate",
        "delta_avg_executed_turnover", "state_aware_selection_days",
        "state_aware_selection_changed_days",
    ]
    all_cols = [
        "split", "stress", "portfolio_value", "delta_ann", "delta_sharpe", "delta_mdd",
        "delta_avg_executed_turnover", "delta_total_cost", "state_aware_selection_days",
        "state_aware_selection_changed_days", "state_aware_selection_suppressed_days",
        "total_state_aware_selection_suppressed",
    ]
    candidate_name = Path(candidate_dir).name
    if candidate_name == "risk_suppress_d015":
        next_step = (
            "Trade-level attribution confirms lower risk and turnover, but Val lag1 is weaker. "
            "Keep this family frozen as a conditional risk candidate; do not tune its threshold further."
        )
    else:
        next_step = (
            "Trade-level attribution confirms lower beta/specific volatility was not uniform across the path. "
            "Keep this family frozen as a conditional return candidate; do not tune its threshold further."
        )
    lines = [
        "# State-aware portfolio-construction pilot (2026-07-15)",
        "",
        "## Status",
        "",
        f"This is a Phase 5 research comparison of the fixed `{candidate_name}` proposal against the same-alpha baseline. Both arms use the unchanged realistic open-price share-ledger; no forward data was used for selection.",
        "",
        f"- Proposal config: `{config_path}`",
        f"- Baseline directory: `{root / baseline_dir}`",
        f"- Candidate directory: `{root / candidate_dir}`",
        f"- Sharpe non-worse: **{sharpe_wins}/{total}** cells",
        f"- Annualized return non-worse: **{ann_wins}/{total}** cells",
        f"- Maximum drawdown non-worse: **{mdd_not_worse}/{total}** cells",
        "- Decision: **conditional research candidate; no registry or forward promotion**.",
        "",
        "## Contract",
        "",
        "- Fixed alpha: `compact_v14_eq_rank` (50% compact + 50% v14 rank mean).",
        "- Target/hold/band: `0.006 / 0.10 / 0.20`; maximum 5 new names.",
        "- Capitals: CNY 500k and CNY 1m.",
        "- Stresses: `normal`, `lag1`, `cost2x`, `capacity_3pct`.",
        "- Risk state: execution-date `global_defensive_pressure`; candidate parameters are recorded in the sweep summary.",
        "- Forward 2026: observation-only and not present in this selection report.",
        "",
        "## Mandatory Date Fields",
        "",
        _table(date_rows, ["split", *REQUIRED_DATE_FIELDS]),
        "",
        "## CNY 1m Normal Comparison",
        "",
        _table(
            preferred,
            table_cols,
            ["Split", "Stress", "Capital", "Base ann", "Candidate ann", "Delta ann", "Base Sharpe", "Candidate Sharpe", "Delta Sharpe", "Base MDD", "Candidate MDD", "Delta MDD", "Base exec turnover", "Candidate exec turnover", "Delta turnover", "Active days", "Changed days"],
        ),
        "",
        "## Full Stress Delta",
        "",
        _table(
            frame.sort_values(["split", "stress", "portfolio_value"]),
            all_cols,
            ["Split", "Stress", "Capital", "Delta ann", "Delta Sharpe", "Delta MDD", "Delta exec turnover", "Delta cost", "Active days", "Changed days", "Suppressed days", "Suppressed count"],
        ),
        "",
        "## Interpretation",
        "",
        "The candidate's selection-layer activity and suppression counts are recorded per cell in the full-stress table. A positive result still requires both selection splits, the full stress suite, and unchanged realistic execution. This report is evidence only; it does not promote a registry candidate.",
        "",
        next_step,
        "",
        "## Evidence",
        "",
        f"- Delta CSV: `{output_csv}`",
        f"- Config: `{config_path}`",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--candidate-name",
        default="risk_rank_t035_p010",
        help="Proposal directory/name to compare against baseline.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    root = Path(args.experiment_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    proposals = {item["name"]: item for item in config["proposals"]}
    baseline_dir = proposals["baseline"]["name"]
    candidate_dir = proposals[args.candidate_name]["name"]
    frames = [compare_split(root, split, baseline_dir, candidate_dir) for split in ("val_2024", "test_2025")]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.output_csv, index=False)
    report = build_report(frame, config_path, root, baseline_dir, candidate_dir, args.output_csv)
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report, encoding="utf-8")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_report}")
    print(f"cells={len(frame)} sharpe_non_worse={(frame['delta_sharpe'] >= 0).sum()} ann_non_worse={(frame['delta_ann'] >= 0).sum()} mdd_non_worse={(frame['delta_mdd'] <= 0).sum()}")


if __name__ == "__main__":
    main()
