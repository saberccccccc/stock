"""Build an APM scorecard from registry/reports.csv.

This is the registry-driven replacement for ad-hoc path scans.  It refuses to
mix execution modes by default and separates selection splits from forward
observation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
root_path = str(ROOT)
if root_path in sys.path:
    sys.path.remove(root_path)
sys.path.insert(0, root_path)

from core.research_protocol import SPLIT_SPECS, validate_report_role, validate_result_dates
from experiments.recording import validate_manifest_for_formal_use

METRICS = [
    "ann",
    "sharpe",
    "mdd",
    "active_ann",
    "information_ratio",
    "avg_turnover",
    "avg_executed_turnover",
    "total_cost",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-csv", default="registry/reports.csv")
    parser.add_argument("--candidates-csv", default="registry/candidates.csv")
    parser.add_argument("--output-dir", default="reports/official_registry_scorecard_20260710")
    parser.add_argument("--decision-rules", default="registry/decision_rules.json")
    parser.add_argument(
        "--attribution-coverage",
        default="reports/official_registry_attribution_20260710/attribution_coverage.csv",
    )
    parser.add_argument("--execution-mode", default="realistic")
    parser.add_argument("--candidate-id", action="append", default=None)
    parser.add_argument("--expected-split", action="append", choices=sorted(SPLIT_SPECS), default=None)
    parser.add_argument("--allow-mixed-execution-mode", action="store_true")
    return parser.parse_args(argv)


def read_csv(path):
    full = ROOT / path
    if not full.exists():
        raise FileNotFoundError(full)
    return pd.read_csv(full)


def read_json(path):
    full = ROOT / path
    if not full.exists():
        raise FileNotFoundError(full)
    return json.loads(full.read_text(encoding="utf-8"))


def load_summary(path):
    full = ROOT / str(path)
    if not full.exists():
        raise FileNotFoundError(full)
    return pd.read_csv(full)


def to_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def build_long(reports, candidates, execution_mode, allow_mixed):
    candidate_status = dict(zip(candidates["candidate_id"], candidates["status"]))
    records = []
    for _, report in reports.iterrows():
        validate_report_role(
            str(report.get("split", "")),
            selection_eligible=report.get("selection_eligible", ""),
            is_forward=report.get("is_forward", ""),
        )
        evidence_class = str(report.get("evidence_class", "")).strip()
        if evidence_class == "formal_experiment":
            manifest = str(report.get("experiment_manifest", "")).strip()
            if not manifest:
                raise ValueError("formal registry report is missing experiment_manifest")
            validate_manifest_for_formal_use(ROOT / manifest)
        elif evidence_class != "legacy_registered":
            raise ValueError(f"unsupported or missing evidence_class={evidence_class!r}")
        mode = str(report.get("execution_mode", ""))
        if not allow_mixed and mode != execution_mode:
            continue
        frame = load_summary(report["path"])
        capital = to_float(report.get("capital"))
        if "portfolio_value" in frame.columns and not np.isnan(capital):
            frame = frame.loc[np.isclose(frame["portfolio_value"].astype(float), capital)]
        for _, row in frame.iterrows():
            if evidence_class == "formal_experiment":
                validate_result_dates(
                    str(report["split"]),
                    signal_start=row.get("signal_start", report.get("signal_start", "")),
                    signal_end=row.get("signal_end", report.get("signal_end", "")),
                    backtest_start=row.get("backtest_start", report.get("backtest_start", "")),
                    backtest_end=row.get("backtest_end", report.get("backtest_end", "")),
                )
            rec = {
                "candidate": report["candidate_id"],
                "status": candidate_status.get(report["candidate_id"], ""),
                "split": report["split"],
                "stress": report["stress"],
                "capital": to_float(row.get("portfolio_value", report.get("capital"))),
                "execution_mode": str(row.get("execution_constraint_mode", mode)),
                "selection_eligible": str(report.get("selection_eligible", "")).lower() == "true",
                "is_forward": str(report.get("is_forward", "")).lower() == "true",
                "source": report["path"],
                "signal_start": str(row.get("signal_start", report.get("signal_start", ""))),
                "signal_end": str(row.get("signal_end", report.get("signal_end", ""))),
                "backtest_start": str(row.get("backtest_start", report.get("backtest_start", ""))),
                "backtest_end": str(row.get("backtest_end", report.get("backtest_end", ""))),
            }
            for metric in METRICS:
                rec[metric] = to_float(row.get(metric))
            records.append(rec)
    return pd.DataFrame(records)


def summarize(frame):
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for candidate, group in frame.groupby("candidate", sort=False):
        rows.append(
            {
                "candidate": candidate,
                "status": group["status"].iloc[0],
                "rows": int(len(group)),
                "splits": ",".join(sorted(group["split"].unique())),
                "stresses": ",".join(sorted(group["stress"].unique())),
                "capital_count": int(group["capital"].nunique(dropna=True)),
                "mean_ir": float(group["information_ratio"].mean()),
                "mean_active_ann": float(group["active_ann"].mean()),
                "mean_ann": float(group["ann"].mean()),
                "mean_sharpe": float(group["sharpe"].mean()),
                "min_sharpe": float(group["sharpe"].min()),
                "worst_mdd": float(group["mdd"].max()),
                "mean_turnover": float(group["avg_executed_turnover"].mean()),
                "mean_cost": float(group["total_cost"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["mean_sharpe", "mean_ann"], ascending=False)
    return out


def coverage(frame, expected_splits, expected_stresses, expected_capitals):
    rows = []
    for candidate, group in frame.groupby("candidate", sort=False):
        observed = {
            (str(r.split), str(r.stress), float(r.capital))
            for r in group.itertuples()
            if not pd.isna(r.capital)
        }
        missing = []
        for split in expected_splits:
            for stress in expected_stresses:
                for capital in expected_capitals:
                    if (split, stress, capital) not in observed:
                        missing.append(f"{split}/{stress}/{int(capital)}")
        rows.append(
            {
                "candidate": candidate,
                "expected_rows": len(expected_splits) * len(expected_stresses) * len(expected_capitals),
                "observed_rows": len(observed),
                "missing_count": len(missing),
                "missing": ";".join(missing),
            }
        )
    return pd.DataFrame(rows)


def decision_table(selection, coverage_frame, attribution_coverage, baseline_id, decision_config):
    """Compare each candidate with the formal baseline on identical evidence."""
    summary = summarize(selection).set_index("candidate")
    if baseline_id not in summary.index:
        raise ValueError(f"Formal baseline {baseline_id!r} has no selection evidence")
    baseline = summary.loc[baseline_id]
    coverage_by_candidate = coverage_frame.set_index("candidate")
    attribution_by_candidate = attribution_coverage.set_index("candidate") if not attribution_coverage.empty else pd.DataFrame()
    rows = []
    for candidate, row in summary.iterrows():
        missing = int(coverage_by_candidate.loc[candidate, "missing_count"])
        attribution_missing = 0
        attribution_pass = True
        if candidate != baseline_id and decision_config.get("require_selection_attribution", False):
            if attribution_by_candidate.empty or candidate not in attribution_by_candidate.index:
                attribution_missing = 16
                attribution_pass = False
            else:
                attribution_missing = int(attribution_by_candidate.loc[candidate, "selection_missing_count"])
                registered = attribution_by_candidate.loc[candidate, "registered"]
                attribution_pass = str(registered).strip().lower() == "true" and attribution_missing == 0
        checks = {
            "coverage_pass": missing == 0,
            "attribution_pass": attribution_pass,
            "mean_sharpe_pass": row["mean_sharpe"] >= baseline["mean_sharpe"] + decision_config["rules"]["min_mean_sharpe_gain"],
            "min_sharpe_pass": row["min_sharpe"] >= baseline["min_sharpe"] - decision_config["rules"]["max_min_sharpe_drop"],
            "mean_ann_pass": row["mean_ann"] >= baseline["mean_ann"] - decision_config["rules"]["max_mean_ann_drop"],
            "mdd_pass": row["worst_mdd"] <= baseline["worst_mdd"] + decision_config["rules"]["max_worst_mdd_increase"],
            "turnover_pass": row["mean_turnover"] <= baseline["mean_turnover"] * decision_config["rules"]["max_mean_turnover_ratio"],
            "cost_pass": row["mean_cost"] <= baseline["mean_cost"] * decision_config["rules"]["max_mean_cost_ratio"],
        }
        if candidate == baseline_id:
            decision = "formal_baseline"
        elif not checks["coverage_pass"] or not checks["attribution_pass"]:
            decision = decision_config["decision_labels"]["incomplete"]
        elif all(checks.values()):
            decision = decision_config["decision_labels"]["pass"]
        else:
            decision = decision_config["decision_labels"]["fail"]
        rows.append({
            "candidate": candidate,
            "decision": decision,
            "mean_sharpe_delta": row["mean_sharpe"] - baseline["mean_sharpe"],
            "mean_ann_delta": row["mean_ann"] - baseline["mean_ann"],
            "min_sharpe_delta": row["min_sharpe"] - baseline["min_sharpe"],
            "worst_mdd_delta": row["worst_mdd"] - baseline["worst_mdd"],
            "turnover_ratio": row["mean_turnover"] / baseline["mean_turnover"],
            "cost_ratio": row["mean_cost"] / baseline["mean_cost"],
            "missing_count": missing,
            "attribution_missing_count": attribution_missing,
            **checks,
        })
    return pd.DataFrame(rows).sort_values(["decision", "mean_sharpe_delta"], ascending=[True, False])


def write_markdown(path, selection_summary, forward_summary, coverage_frame, decisions, rules_path):
    lines = [
        "# Registry APM Scorecard",
        "",
        "Selection summary uses only `val_2024` and `test_2025`.",
        "`forward_2026` is reported separately as observation-only.",
        f"Decision rules: `{rules_path}`.",
        "",
        "## Selection Summary",
        "",
        selection_summary.to_markdown(index=False) if not selection_summary.empty else "(empty)",
        "",
        "## Decision Against Formal Baseline",
        "",
        decisions.to_markdown(index=False) if not decisions.empty else "(empty)",
        "",
        "## Forward Observation",
        "",
        forward_summary.to_markdown(index=False) if not forward_summary.empty else "(empty)",
        "",
        "## Coverage",
        "",
        coverage_frame.to_markdown(index=False) if not coverage_frame.empty else "(empty)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    reports = read_csv(args.reports_csv)
    candidates = read_csv(args.candidates_csv)
    if args.candidate_id:
        requested = set(args.candidate_id)
        reports = reports.loc[reports["candidate_id"].isin(requested)].copy()
        candidates = candidates.loc[candidates["candidate_id"].isin(requested)].copy()
        missing = requested - set(candidates["candidate_id"])
        if missing:
            raise KeyError(f"unknown candidate IDs in scorecard request: {sorted(missing)}")
    decision_config = read_json(args.decision_rules)
    attribution_coverage_path = ROOT / args.attribution_coverage
    if decision_config.get("require_selection_attribution", False) and not attribution_coverage_path.exists():
        raise FileNotFoundError(f"Required attribution coverage is missing: {attribution_coverage_path}")
    attribution_coverage = (
        pd.read_csv(attribution_coverage_path)
        if attribution_coverage_path.exists()
        else pd.DataFrame(columns=["candidate", "registered", "selection_missing_count"])
    )
    long_df = build_long(
        reports,
        candidates,
        execution_mode=args.execution_mode,
        allow_mixed=args.allow_mixed_execution_mode,
    )
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_dir / "registry_scorecard_long.csv", index=False, encoding="utf-8-sig")
    selection_splits = decision_config["selection_splits"]
    selection = long_df.loc[
        long_df["selection_eligible"]
        & long_df["split"].isin(selection_splits)
        & ~long_df["is_forward"]
    ].copy()
    forward = long_df.loc[long_df["split"].eq("forward_2026") | long_df["is_forward"]].copy()
    selection_summary = summarize(selection)
    forward_summary = summarize(forward)
    expected_splits = args.expected_split or selection_splits + ["forward_2026"]
    coverage_frame = coverage(
        long_df,
        expected_splits=expected_splits,
        expected_stresses=decision_config["required_stresses"],
        expected_capitals=[float(x) for x in decision_config["required_capitals"]],
    )
    selection_coverage = coverage(
        selection,
        expected_splits=selection_splits,
        expected_stresses=decision_config["required_stresses"],
        expected_capitals=[float(x) for x in decision_config["required_capitals"]],
    )
    decisions = decision_table(
        selection,
        selection_coverage,
        attribution_coverage,
        baseline_id=decision_config["formal_baseline"],
        decision_config=decision_config,
    )
    selection_summary.to_csv(out_dir / "registry_selection_summary.csv", index=False, encoding="utf-8-sig")
    forward_summary.to_csv(out_dir / "registry_forward_summary.csv", index=False, encoding="utf-8-sig")
    coverage_frame.to_csv(out_dir / "registry_coverage.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(out_dir / "registry_decisions.csv", index=False, encoding="utf-8-sig")
    write_markdown(out_dir / "registry_apm_scorecard.md", selection_summary, forward_summary, coverage_frame, decisions, args.decision_rules)
    print(
        {
            "output_dir": str(out_dir),
            "long_rows": int(len(long_df)),
            "selection_rows": int(len(selection)),
            "forward_rows": int(len(forward)),
            "coverage_missing": int(coverage_frame["missing_count"].sum()) if not coverage_frame.empty else 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
