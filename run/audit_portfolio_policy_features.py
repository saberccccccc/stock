"""Audit feature coverage for state-aware portfolio policy events."""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_FEATURE_GROUPS = {
    "portfolio_state": [
        "diag_gross_weight",
        "diag_market_mult",
        "diag_portfolio_beta_60d",
        "diag_portfolio_beta_per_gross_60d",
        "diag_portfolio_specific_vol_60d",
        "diag_turnover",
        "diag_executed_turnover",
    ],
    "market_state": [
        "diag_active_drawdown_trailing_return",
        "diag_global_risk_pressure",
        "cand_global_us_hk_pressure",
        "cand_global_defensive_pressure",
        "cand_global_hk_risk_pressure",
    ],
    "concentration": [
        "diff_candidate_industry_top_share",
        "diff_top_industry_share",
        "diff_top_industry_hhi",
        "cand_top_industry_share",
        "base_top_industry_share",
    ],
    "candidate_vs_baseline": [
        "diff_candidate_rank_pct",
        "diff_ret_1d",
        "diff_ret_5d",
        "diff_ret_20d",
        "diff_vol_20d",
        "diff_vol_60d",
        "diff_drawdown_20d",
        "diff_beta_60d",
        "diff_specific_vol_60d",
        "diff_money_ma20",
        "diff_was_held",
        "diff_holding_age",
    ],
    "execution_cost_risk": [
        "ledger_cost",
        "pair_risk_delta",
        "pair_rank_delta",
    ],
}

LABEL_OR_FUTURE_EVIDENCE_COLUMNS = [
    "label_available",
    "ledger_path_utility",
    "ledger_weighted_raw_edge",
    "pair_path_raw_edge",
    "pair_downside_delta",
    "pair_quick_fade_delta",
]


def parse_event_spec(raw):
    parts = str(raw).split(":", 3)
    if len(parts) != 4:
        raise ValueError("event spec must be candidate:split:scenario:path")
    candidate, split, scenario, path = parts
    return {
        "candidate": candidate.strip(),
        "split": split.strip(),
        "scenario": scenario.strip(),
        "path": Path(path.strip()),
    }


def audit_event_file(spec):
    path = spec["path"]
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    rows = []
    for group, columns in REQUIRED_FEATURE_GROUPS.items():
        for column in columns:
            if column not in frame.columns:
                rows.append(
                    {
                        **{key: spec[key] for key in ("candidate", "split", "scenario")},
                        "path": str(path),
                        "group": group,
                        "column": column,
                        "status": "missing",
                        "missing_rate": 1.0,
                        "non_null": 0,
                        "rows": int(len(frame)),
                    }
                )
                continue
            series = frame[column]
            missing_rate = float(series.isna().mean()) if len(series) else 0.0
            rows.append(
                {
                    **{key: spec[key] for key in ("candidate", "split", "scenario")},
                    "path": str(path),
                    "group": group,
                    "column": column,
                    "status": "present" if missing_rate < 1.0 else "all_missing",
                    "missing_rate": missing_rate,
                    "non_null": int(series.notna().sum()),
                    "rows": int(len(frame)),
                }
            )
    for column in LABEL_OR_FUTURE_EVIDENCE_COLUMNS:
        rows.append(
            {
                **{key: spec[key] for key in ("candidate", "split", "scenario")},
                "path": str(path),
                "group": "label_or_future_evidence",
                "column": column,
                "status": "present" if column in frame.columns else "absent",
                "missing_rate": float(frame[column].isna().mean()) if column in frame.columns and len(frame) else 1.0,
                "non_null": int(frame[column].notna().sum()) if column in frame.columns else 0,
                "rows": int(len(frame)),
            }
        )
    return rows


def summarize_audit(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    required = frame[frame["group"] != "label_or_future_evidence"].copy()
    grouped = []
    for (candidate, split, scenario), group in required.groupby(["candidate", "split", "scenario"], sort=False):
        missing = group[group["status"] == "missing"]
        all_missing = group[group["status"] == "all_missing"]
        grouped.append(
            {
                "candidate": candidate,
                "split": split,
                "scenario": scenario,
                "required_columns": int(len(group)),
                "missing_columns": int(len(missing)),
                "all_missing_columns": int(len(all_missing)),
                "max_missing_rate": float(group["missing_rate"].max()),
                "status": "complete" if missing.empty and all_missing.empty else "incomplete",
            }
        )
    return pd.DataFrame(grouped)


def write_markdown(detail, summary, output_path):
    lines = [
        "# Portfolio Policy Feature Audit",
        "",
        "This audit checks whether replacement-event files contain the feature groups needed by the state-aware portfolio layer.",
        "Label/future-evidence columns may exist for research audit, but they must not be used as deployable features.",
        "",
        "## Summary",
        "",
        "| candidate | split | scenario | status | required | missing | all missing | max missing rate |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['split']} | {row['scenario']} | {row['status']} | "
            f"{int(row['required_columns'])} | {int(row['missing_columns'])} | "
            f"{int(row['all_missing_columns'])} | {row['max_missing_rate']:.3f} |"
        )
    label_rows = detail[detail["group"] == "label_or_future_evidence"]
    if not label_rows.empty:
        lines.extend(["", "## Label/Future Evidence Columns", ""])
        lines.append("| candidate | split | scenario | column | status | non-null |")
        lines.append("|---|---|---|---|---|---:|")
        for _, row in label_rows.iterrows():
            lines.append(
                f"| {row['candidate']} | {row['split']} | {row['scenario']} | "
                f"{row['column']} | {row['status']} | {int(row['non_null'])} |"
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event",
        action="append",
        default=[],
        help="candidate:split:scenario:path to replacement_events.csv",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    specs = [parse_event_spec(raw) for raw in args.event]
    if not specs:
        raise SystemExit("at least one --event is required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        rows.extend(audit_event_file(spec))
    detail = pd.DataFrame(rows)
    summary = summarize_audit(rows)
    detail.to_csv(output_dir / "portfolio_policy_feature_audit_detail.csv", index=False)
    summary.to_csv(output_dir / "portfolio_policy_feature_audit_summary.csv", index=False)
    write_markdown(detail, summary, output_dir / "portfolio_policy_feature_audit.md")
    print(f"wrote portfolio policy feature audit to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
