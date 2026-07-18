"""Audit whether a candidate has a complete Active Portfolio Management packet."""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_SUMMARY_COLUMNS = {
    "measurement": [
        "ann",
        "sharpe",
        "mdd",
        "benchmark_ann",
        "active_ann",
        "tracking_error",
        "information_ratio",
        "beta_to_benchmark",
    ],
    "execution": [
        "avg_executed_turnover",
        "avg_unfilled_turnover",
        "total_cost",
        "blocked_buy",
        "blocked_sell",
        "adv_blocked",
    ],
    "risk": [
        "avg_portfolio_beta_60d",
        "avg_portfolio_beta_per_gross_60d",
        "avg_portfolio_specific_vol_60d",
    ],
}

REQUIRED_ATTRIBUTION_FILES = [
    "apm_attribution_summary.csv",
    "apm_industry_exposure.csv",
    "apm_style_exposure.csv",
    "apm_slice_summary.csv",
    "apm_attribution_report.md",
]

REQUIRED_STRESSES = {"normal", "lag1", "cost2x", "capacity_3pct"}
REQUIRED_SPLITS = {"validation_2024", "test_2025_20260518", "forward_shadow"}


def parse_summary_spec(raw):
    parts = str(raw).split(":", 3)
    if len(parts) != 4:
        raise ValueError("summary spec must be candidate:split:scenario:path")
    candidate, split, scenario, path = parts
    return {
        "candidate": candidate.strip(),
        "split": split.strip(),
        "scenario": scenario.strip(),
        "path": Path(path.strip()),
    }


def item(name, status, evidence="", severity="info", note=""):
    return {
        "item": name,
        "status": status,
        "severity": severity,
        "evidence": evidence,
        "note": note,
    }


def audit_summary(spec):
    path = spec["path"]
    prefix = f"summary:{spec['candidate']}:{spec['split']}:{spec['scenario']}"
    if not path.exists():
        return [item(prefix, "missing", str(path), "error", "summary file not found")]
    frame = pd.read_csv(path)
    rows = [item(prefix, "present", str(path), "info", f"rows={len(frame)}")]
    columns = set(frame.columns)
    for group, required in REQUIRED_SUMMARY_COLUMNS.items():
        missing = [col for col in required if col not in columns]
        status = "complete" if not missing else "incomplete"
        rows.append(
            item(
                f"{prefix}:{group}_columns",
                status,
                str(path),
                "error" if missing else "info",
                "missing=" + ",".join(missing) if missing else "all required columns present",
            )
        )
    return rows


def audit_attribution_dir(path):
    path = Path(path)
    rows = []
    if not path.exists():
        return [item(f"attribution:{path.name}", "missing", str(path), "error", "directory not found")]
    for filename in REQUIRED_ATTRIBUTION_FILES:
        file_path = path / filename
        rows.append(
            item(
                f"attribution:{path.name}:{filename}",
                "present" if file_path.exists() else "missing",
                str(file_path),
                "info" if file_path.exists() else "error",
            )
        )
    slice_path = path / "apm_slice_summary.csv"
    if slice_path.exists():
        frame = pd.read_csv(slice_path)
        required_slices = {"market_state", "industry_concentration"}
        found = set(frame.get("slice", pd.Series(dtype=str)).astype(str))
        missing = sorted(required_slices - found)
        rows.append(
            item(
                f"attribution:{path.name}:required_slices",
                "complete" if not missing else "incomplete",
                str(slice_path),
                "error" if missing else "info",
                "missing=" + ",".join(missing) if missing else "market and concentration slices present",
            )
        )
    return rows


def audit_coverage(summary_specs):
    rows = []
    splits = {spec["split"] for spec in summary_specs}
    scenarios = {spec["scenario"] for spec in summary_specs}
    scenarios_by_split = {}
    for spec in summary_specs:
        scenarios_by_split.setdefault(spec["split"], set()).add(spec["scenario"])
    missing_splits = sorted(REQUIRED_SPLITS - splits)
    missing_stresses = sorted(REQUIRED_STRESSES - scenarios)
    rows.append(
        item(
            "coverage:required_splits",
            "complete" if not missing_splits else "incomplete",
            ",".join(sorted(splits)),
            "error" if missing_splits else "info",
            "missing=" + ",".join(missing_splits) if missing_splits else "all required splits present",
        )
    )
    rows.append(
        item(
            "coverage:required_stresses",
            "complete" if not missing_stresses else "incomplete",
            ",".join(sorted(scenarios)),
            "error" if missing_stresses else "info",
            "missing=" + ",".join(missing_stresses) if missing_stresses else "all required stresses present",
        )
    )
    for split in sorted(REQUIRED_SPLITS):
        split_scenarios = scenarios_by_split.get(split, set())
        missing = sorted(REQUIRED_STRESSES - split_scenarios)
        rows.append(
            item(
                f"coverage:split_stresses:{split}",
                "complete" if not missing else "incomplete",
                ",".join(sorted(split_scenarios)),
                "error" if missing else "info",
                "missing=" + ",".join(missing)
                if missing
                else "all required stresses present for split",
            )
        )
    return rows


def audit_protocol(candidate, trusted_protocol=False):
    legacy_like = "v9" in str(candidate).lower()
    if trusted_protocol:
        return [item("protocol:trust_status", "complete", candidate, "info", "marked trusted by caller")]
    if legacy_like:
        return [
            item(
                "protocol:trust_status",
                "weak",
                candidate,
                "warning",
                "legacy V9 evidence is treated as unverified; high Sharpe is not sufficient",
            )
        ]
    return [
        item(
            "protocol:trust_status",
            "weak",
            candidate,
            "warning",
            "candidate has not been explicitly marked as protocol-trusted",
        )
    ]


def build_markdown(rows, output_path):
    frame = pd.DataFrame(rows)
    counts = frame["status"].value_counts().to_dict() if not frame.empty else {}
    lines = [
        "# APM Completeness Audit",
        "",
        "This audit checks evidence completeness, not performance quality.",
        "",
        "## Status Counts",
        "",
    ]
    for status in sorted(counts):
        lines.append(f"- {status}: {counts[status]}")
    lines.extend(["", "## Findings", ""])
    lines.append("| item | status | severity | note |")
    lines.append("|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['item']} | {row['status']} | {row['severity']} | {row.get('note', '')} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(candidate, summary_specs, attribution_dirs, trusted_protocol=False):
    rows = []
    rows.extend(audit_protocol(candidate, trusted_protocol=trusted_protocol))
    rows.extend(audit_coverage(summary_specs))
    for spec in summary_specs:
        rows.extend(audit_summary(spec))
    for path in attribution_dirs:
        rows.extend(audit_attribution_dir(path))
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="candidate:split:scenario:path for an open_ledger_summary.csv",
    )
    parser.add_argument("--attribution-dir", action="append", default=[])
    parser.add_argument("--trusted-protocol", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    specs = [parse_summary_spec(raw) for raw in args.summary]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_audit(
        args.candidate,
        specs,
        args.attribution_dir,
        trusted_protocol=args.trusted_protocol,
    )
    pd.DataFrame(rows).to_csv(output_dir / "apm_completeness_audit.csv", index=False)
    build_markdown(rows, output_dir / "apm_completeness_audit.md")
    print(f"wrote APM completeness audit to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
