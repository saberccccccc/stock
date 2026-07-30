"""Run official registry-driven open-ledger backtests.

This wrapper is the Phase 2 bridge from ad-hoc manifest commands to a standard
registry-driven flow.  It prefers the batch sweep runner so OHLC data is loaded
once per split, then materializes the sweep output back into the familiar
``open_ledger_summary.csv`` layout for compatibility with existing scorecards.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
root_path = str(ROOT)
if root_path in sys.path:
    sys.path.remove(root_path)
# Direct script execution puts run/ first, where run/backtest.py would shadow
# the project backtest package.
sys.path.insert(0, root_path)

from backtest.market_data_contract import add_execution_market_data_args
from core.research_protocol import (
    RESEARCH_END_DATE,
    SPLIT_SPECS,
    get_split_spec,
    validate_result_dates,
)
from experiments.recording import (
    append_event,
    canonical_json_hash,
    create_experiment,
    declared_range,
    finalize_artifact_index,
    fingerprint_path,
    not_applicable_range,
    record_artifact,
    sha256_file,
    validate_manifest_for_formal_use,
)
from experiments.prediction_artifacts import resolve_split_alpha_path

PYTHON = sys.executable

DEFAULT_STRESSES = "normal,lag1,cost2x,capacity_3pct"
DEFAULT_CAPITALS = "500000,1000000"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", default="registry/candidates.csv")
    parser.add_argument("--reports-csv", default="registry/reports.csv")
    parser.add_argument(
        "--research-data-dir",
        default="data/raw",
        help="Frozen research data used for validation and test splits.",
    )
    parser.add_argument(
        "--forward-data-dir",
        default="data/forward_raw",
        help="Forward-only data used for forward observation splits.",
    )
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--split", action="append", default=None, choices=sorted(SPLIT_SPECS))
    parser.add_argument("--output-root", default="reports/official")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stresses", default=DEFAULT_STRESSES)
    parser.add_argument("--portfolio-values", default=DEFAULT_CAPITALS)
    parser.add_argument("--target-fracs", default="0.006")
    parser.add_argument("--hold-fracs", default="0.10")
    parser.add_argument("--rebalance-bands", default="0.20")
    parser.add_argument("--max-new-names-list", default="5")
    parser.add_argument("--exit-hold-fracs", default="0")
    parser.add_argument("--switch-gap-fracs", default="0")
    parser.add_argument("--execution-mode", default="realistic", choices=("proxy", "realistic"))
    add_execution_market_data_args(parser)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--append-registry", action="store_true")
    parser.add_argument("--parent-fit-end", default=None)
    parser.add_argument("--parent-selection-end", default=None)
    return parser.parse_args(argv)


def read_candidates(path):
    full = ROOT / path
    frame = pd.read_csv(full)
    frame = frame.set_index("candidate_id", drop=False)
    return frame


def split_alpha_path(signal_path, split):
    return resolve_split_alpha_path(ROOT, signal_path, split)


def alpha_specs(candidates, candidate_ids, split):
    specs = []
    skipped = []
    for candidate_id in candidate_ids:
        if candidate_id not in candidates.index:
            raise KeyError(f"unknown candidate_id: {candidate_id}")
        row = candidates.loc[candidate_id]
        signal_path = str(row.get("signal_path", "")).strip()
        if not signal_path:
            skipped.append({"candidate_id": candidate_id, "reason": "empty signal_path"})
            continue
        try:
            alpha_path = split_alpha_path(signal_path, split)
        except FileNotFoundError as exc:
            skipped.append({"candidate_id": candidate_id, "reason": str(exc)})
            continue
        specs.append(f"{candidate_id}={alpha_path}")
    if not specs:
        raise ValueError(f"no runnable alpha specs for split={split}; skipped={skipped}")
    return ",".join(specs), skipped


def build_command(args, split, alpha_spec_text, out_dir):
    spec = get_split_spec(split)
    start, end, max_date = spec.command_dates()
    data_dir = args.forward_data_dir if spec.is_forward else args.research_data_dir
    cmd = [
        PYTHON,
        "run/sweep_open_price_ledger_params.py",
        "--alpha-specs",
        alpha_spec_text,
        "--output-dir",
        str(out_dir),
        "--data-dir",
        data_dir,
        "--target-fracs",
        args.target_fracs,
        "--hold-fracs",
        args.hold_fracs,
        "--rebalance-bands",
        args.rebalance_bands,
        "--stresses",
        args.stresses,
        "--portfolio-values",
        args.portfolio_values,
        "--max-new-names-list",
        args.max_new_names_list,
        "--exit-hold-fracs",
        args.exit_hold_fracs,
        "--switch-gap-fracs",
        args.switch_gap_fracs,
        "--execution-constraint-mode",
        args.execution_mode,
        "--ohlc-backend",
        getattr(args, "ohlc_backend", "legacy"),
        "--market-daily-store-root",
        getattr(args, "market_daily_store_root", "data/market_daily_candidate_v2"),
        "--ohlc-monthly-cache-dir",
        getattr(args, "ohlc_monthly_cache_dir", "cache/ohlcv_monthly_v3_candidate"),
        "--start-date",
        start,
        "--end-date",
        end,
        "--max-data-date",
        max_date,
        "--save-path-details",
    ]
    if args.resume:
        cmd.append("--resume")
    return cmd


def materialize_open_ledger_summaries(sweep_summary, output_root):
    frame = pd.read_csv(sweep_summary)
    required = {"alpha_name", "stress", "portfolio_value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"sweep summary missing required columns: {sorted(missing)}")
    written = []
    for (alpha_name, stress), group in frame.groupby(["alpha_name", "stress"], sort=False):
        out_dir = output_root / str(alpha_name) / str(stress)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "open_ledger_summary.csv"
        group.to_csv(out_path, index=False)
        written.append(out_path)
    return written


def record_detailed_ledger_artifacts(experiment_dir, split, path_index):
    index_path = Path(path_index).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"missing detailed ledger artifact index: {index_path}")
    frame = pd.read_csv(index_path)
    artifact_columns = (
        "equity_curve",
        "diagnostics",
        "positions",
        "orders",
        "rejections",
        "costs",
    )
    missing = set(artifact_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"ledger artifact index missing columns: {sorted(missing)}")
    record_artifact(
        experiment_dir,
        name=f"ledger_path_index:{split}",
        path=index_path,
        kind="ledger_path_artifact_index_v1",
    )
    recorded = 0
    for _, row in frame.iterrows():
        key_hash = str(row["sweep_key_sha256"])
        for artifact_name in artifact_columns:
            artifact_path = Path(str(row[artifact_name])).resolve()
            record_artifact(
                experiment_dir,
                name=f"ledger_detail:{split}:{key_hash}:{artifact_name}",
                path=artifact_path,
                kind=f"open_ledger_{artifact_name}_v1",
            )
            recorded += 1
    return recorded


def raise_for_failed_backtests(statuses):
    failures = [item for item in statuses if item.get("status") == "failed"]
    if failures:
        details = [
            {
                "split": item.get("split"),
                "returncode": item.get("returncode"),
                "output_dir": item.get("output_dir"),
            }
            for item in failures
        ]
        raise RuntimeError(f"official backtest subprocess failed: {details}")


def append_report_registry(reports_csv, split, written_paths, experiment_manifest):
    spec = get_split_spec(split)
    manifest_path = Path(experiment_manifest).resolve()
    validate_manifest_for_formal_use(manifest_path, require_artifacts=False)
    path = ROOT / reports_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = []
    for summary in written_paths:
        frame = pd.read_csv(summary)
        stress = summary.parent.name
        candidate_id = summary.parent.parent.name
        for _, row in frame.iterrows():
            validate_result_dates(
                split,
                signal_start=row.get("signal_start"),
                signal_end=row.get("signal_end"),
                backtest_start=row.get("backtest_start"),
                backtest_end=row.get("backtest_end"),
            )
            capital = int(float(row.get("portfolio_value", 0)))
            rows.append(
                {
                    "report_id": f"{candidate_id}:{split}:{stress}:{capital}",
                    "candidate_id": candidate_id,
                    "path": str(summary.relative_to(ROOT)).replace("\\", "/"),
                    "split": split,
                    "stress": stress,
                    "capital": capital,
                    "execution_mode": str(row.get("execution_constraint_mode", "")),
                    "signal_start": str(row.get("signal_start", "")),
                    "signal_end": str(row.get("signal_end", "")),
                    "backtest_start": str(row.get("backtest_start", "")),
                    "backtest_end": str(row.get("backtest_end", "")),
                    "selection_eligible": str(spec.selection_eligible).lower(),
                    "is_forward": str(spec.is_forward).lower(),
                    "source_script": "run/official_backtest_from_registry.py",
                    "notes": "materialized from sweep_open_price_ledger_params.py",
                    "evidence_class": "formal_experiment",
                    "experiment_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
                    "canonical_evidence": "true",
                    "superseded_by": "",
                }
            )
    new_rows = pd.DataFrame(rows)
    if existing.empty:
        combined = new_rows
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["report_id", "path"], keep="last")
    combined.to_csv(path, index=False, encoding="utf-8")
    return len(rows)


def main(argv=None):
    args = parse_args(argv)
    candidates = read_candidates(args.candidates_csv)
    splits = args.split or list(SPLIT_SPECS)
    run_id = args.run_id or pd.Timestamp.now().strftime("official_%Y%m%d_%H%M%S")
    output_root = ROOT / args.output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for split in splits:
        alpha_spec_text, skipped = alpha_specs(candidates, args.candidate_id, split)
        split_out = output_root / "_sweeps" / split
        prepared.append(
            {
                "split": split,
                "alpha_spec_text": alpha_spec_text,
                "skipped": skipped,
                "split_out": split_out,
                "command": build_command(args, split, alpha_spec_text, split_out),
            }
        )

    manifest_path = None
    if not args.dry_run:
        has_forward = any(get_split_spec(split).is_forward for split in splits)
        if has_forward and (not args.parent_fit_end or not args.parent_selection_end):
            raise ValueError(
                "full-year forward requires --parent-fit-end and --parent-selection-end"
            )
        data_sources = []
        used_roots = {
            ("forward_market_data" if get_split_spec(split).is_forward else "selection_market_data"):
            (args.forward_data_dir if get_split_spec(split).is_forward else args.research_data_dir)
            for split in splits
        }
        for role, root in used_roots.items():
            data_sources.append(
                {"role": role, "root": str((ROOT / root).resolve()), "fingerprint": fingerprint_path(ROOT / root)}
            )
        alpha_fingerprints = []
        for item in prepared:
            for token in item["alpha_spec_text"].split(","):
                name, path = token.split("=", 1)
                alpha_fingerprints.append({"name": name, "path": path, "sha256": sha256_file(path)})
        signal_start = min(str(get_split_spec(split).start.date()) for split in splits)
        signal_end = max(str(get_split_spec(split).end.date()) for split in splits)
        scope = {
            "stage": "ledger_evaluation",
            "data_sources": data_sources,
            "ranges": {
                "feature_warmup": not_applicable_range("consumes frozen dated alpha"),
                "train": not_applicable_range("consumes frozen dated alpha"),
                "valid": not_applicable_range("consumes frozen dated alpha"),
                "signal": declared_range(signal_start, signal_end),
                "backtest": declared_range(signal_start, signal_end),
            },
            "max_data_date": max(get_split_spec(split).command_dates()[2] for split in splits),
            "split_roles": [
                {
                    "split": split,
                    "selection_eligible": get_split_spec(split).selection_eligible,
                    "forward_used": get_split_spec(split).is_forward,
                }
                for split in splits
            ],
            "transform": {
                "state_sha256": canonical_json_hash({"alpha_sources": alpha_fingerprints}),
                "fit_range": not_applicable_range("alpha transform is frozen in parent artifacts"),
            },
            "lineage": (
                {
                    "parent_fit_end": args.parent_fit_end,
                    "parent_selection_end": args.parent_selection_end,
                }
                if has_forward
                else {}
            ),
        }
        manifest_path = create_experiment(
            output_root,
            experiment_id=run_id,
            config={
                "candidates": list(args.candidate_id),
                "splits": list(splits),
                "stresses": args.stresses,
                "capitals": args.portfolio_values,
                "execution_mode": args.execution_mode,
            },
            protocol={
                "selection_splits": [name for name in splits if get_split_spec(name).selection_eligible],
                "observation_splits": [name for name in splits if get_split_spec(name).is_forward],
            },
            cache_contract={"data_sources": data_sources, "alpha_sources": alpha_fingerprints},
            project_root=ROOT,
            formal=True,
            experiment_scope=scope,
        )
        record_artifact(output_root, name="candidates_registry", path=ROOT / args.candidates_csv, kind="registry_input")
        append_event(output_root, status="running", event_type="official_backtest_started")
    all_status = []
    for item in prepared:
        split = item["split"]
        alpha_spec_text = item["alpha_spec_text"]
        skipped = item["skipped"]
        split_out = item["split_out"]
        cmd = item["command"]
        status = {
            "split": split,
            "alpha_specs": alpha_spec_text,
            "skipped": skipped,
            "output_dir": str(split_out),
            "command": cmd,
            "dry_run": bool(args.dry_run),
        }
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
            all_status.append({**status, "status": "dry_run"})
            continue
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            all_status.append({**status, "status": "failed", "returncode": int(result.returncode)})
            break
        summary = split_out / "open_price_ledger_param_sweep_summary.csv"
        path_index = split_out / "path_artifact_index.csv"
        written = materialize_open_ledger_summaries(summary, output_root / split)
        for path in written:
            record_artifact(output_root, name=f"ledger_summary:{split}:{path.parent.name}", path=path, kind="ledger_summary")
        detail_artifacts = record_detailed_ledger_artifacts(output_root, split, path_index)
        registry_rows = 0
        if args.append_registry:
            registry_rows = append_report_registry(args.reports_csv, split, written, manifest_path)
        all_status.append(
            {
                **status,
                "status": "ok",
                "summary": str(summary),
                "materialized": [str(path) for path in written],
                "path_artifact_index": str(path_index),
                "detail_artifacts": detail_artifacts,
                "registry_rows": registry_rows,
            }
        )
    status_path = output_root / "official_backtest_status.json"
    status_path.write_text(json.dumps(all_status, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.dry_run:
        record_artifact(output_root, name="official_backtest_status", path=status_path, kind="run_status")
        failed = any(item.get("status") == "failed" for item in all_status)
        append_event(
            output_root,
            status="failed" if failed else "completed",
            event_type="official_backtest_failed" if failed else "official_backtest_completed",
        )
        finalize_artifact_index(output_root)
    print(json.dumps({"status_path": str(status_path)}, ensure_ascii=False, indent=2), flush=True)
    raise_for_failed_backtests(all_status)


if __name__ == "__main__":
    main()
