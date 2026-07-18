"""Validate a frozen forward-shadow scorecard without selecting or tuning anything.

The validator is deliberately read-only with respect to the registry and model
configuration. It checks that forward evidence belongs to a frozen manifest,
stays inside the declared execution period, contains the required contribution
fields, and remains observation-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.recording import sha256_file, source_state
from core.research_protocol import (
    FORWARD_END_DATE,
    FORWARD_START_DATE,
    RESEARCH_END_DATE,
    assert_forward_parent_frozen,
    validate_report_role,
)


DEFAULT_MANIFEST = ROOT / "reports/experiments/phase6_shadow_bundle_20260715_v4/experiment_manifest.json"


def _result(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _nonempty(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].notna() & frame[column].astype(str).str.strip().ne("")


def validate_forward_frame(
    frame: pd.DataFrame,
    *,
    path: str | Path,
    required_fields: list[str],
    research_end: str,
    forward_start: str,
) -> list[dict[str, str]]:
    """Return field/date checks for one forward summary frame."""
    source = str(path)
    checks: list[dict[str, str]] = []
    missing = [field for field in required_fields if field not in frame.columns]
    if missing:
        checks.append(_result(f"forward_fields:{source}", "error", f"missing={','.join(missing)}"))
        return checks
    if frame.empty:
        checks.append(_result(f"forward_rows:{source}", "error", "summary has no rows"))
        return checks

    empty_fields = [field for field in required_fields if not bool(_nonempty(frame, field).all())]
    checks.append(
        _result(
            f"forward_nonempty_fields:{source}",
            "complete" if not empty_fields else "error",
            "all required fields are populated"
            if not empty_fields
            else f"empty_or_null={','.join(empty_fields)}",
        )
    )

    date_fields = ["signal_start", "signal_end", "backtest_start", "backtest_end"]
    parsed = {}
    invalid_dates = []
    for field in date_fields:
        values = pd.to_datetime(frame[field], errors="coerce")
        parsed[field] = values
        if values.isna().any():
            invalid_dates.append(field)
    checks.append(
        _result(
            f"forward_date_parse:{source}",
            "complete" if not invalid_dates else "error",
            "all date fields parse"
            if not invalid_dates
            else f"invalid={','.join(invalid_dates)}",
        )
    )
    if invalid_dates:
        return checks

    research_cutoff = pd.Timestamp(research_end)
    forward_cutoff = pd.Timestamp(forward_start)
    invalid_order = (parsed["signal_start"] > parsed["signal_end"]) | (
        parsed["backtest_start"] > parsed["backtest_end"]
    )
    invalid_boundary = (parsed["signal_start"] < research_cutoff) | (
        parsed["backtest_start"] < forward_cutoff
    ) | (parsed["backtest_end"] < forward_cutoff)
    checks.append(
        _result(
            f"forward_date_order:{source}",
            "complete" if not invalid_order.any() else "error",
            "signal and backtest intervals are ordered"
            if not invalid_order.any()
            else f"invalid_rows={int(invalid_order.sum())}",
        )
    )
    checks.append(
        _result(
            f"forward_date_boundary:{source}",
            "complete" if not invalid_boundary.any() else "error",
            "signal starts at or after research cutoff and execution stays forward-only"
            if not invalid_boundary.any()
            else f"invalid_rows={int(invalid_boundary.sum())}",
        )
    )

    if "split" in frame.columns:
        values = frame["split"].astype(str).str.lower().str.strip()
        invalid_split = ~values.isin({"forward_2026", "forward_shadow", "forward"})
        checks.append(
            _result(
                f"forward_split:{source}",
                "complete" if not invalid_split.any() else "error",
                "all rows are labeled forward observation"
                if not invalid_split.any()
                else f"invalid_rows={int(invalid_split.sum())}",
            )
        )
    if "is_forward" in frame.columns:
        values = frame["is_forward"].astype(str).str.lower().str.strip()
        invalid_flag = ~values.isin({"true", "1", "yes"})
        checks.append(
            _result(
                f"forward_flag:{source}",
                "complete" if not invalid_flag.any() else "error",
                "all rows are marked forward"
                if not invalid_flag.any()
                else f"invalid_rows={int(invalid_flag.sum())}",
            )
        )
    if "selection_eligible" in frame.columns:
        invalid_roles = []
        for index, row in frame.iterrows():
            try:
                validate_report_role(
                    str(row.get("split", "")),
                    selection_eligible=row.get("selection_eligible"),
                    is_forward=row.get("is_forward"),
                )
            except ValueError:
                invalid_roles.append(index)
        checks.append(
            _result(
                f"forward_selection_role:{source}",
                "complete" if not invalid_roles else "error",
                "all forward rows are observation-only"
                if not invalid_roles
                else f"invalid_rows={len(invalid_roles)}",
            )
        )
    if "date" in frame.columns:
        duplicate = frame["date"].astype(str).duplicated(keep=False)
        checks.append(
            _result(
                f"forward_signal_dates:{source}",
                "complete" if not duplicate.any() else "error",
                "dated signal rows are unique"
                if not duplicate.any()
                else f"duplicate_rows={int(duplicate.sum())}",
            )
        )
    return checks


def validate_manifest_controls(manifest: dict, *, manifest_path: Path, project_root: Path, output_dir: Path):
    checks: list[dict[str, str]] = []
    config = manifest.get("config", {})
    boundary = config.get("research_boundary", {})
    activation = config.get("activation", {})
    checks.append(
        _result(
            "control:forward_selection",
            "complete" if boundary.get("forward_selection_allowed") is False else "error",
            "forward selection is disabled" if boundary.get("forward_selection_allowed") is False else "must be false",
        )
    )
    expected_boundary = {
        "research_end": str(RESEARCH_END_DATE.date()),
        "forward_start": str(FORWARD_START_DATE.date()),
        "forward_end": str(FORWARD_END_DATE.date()),
    }
    boundary_matches = all(boundary.get(field) == value for field, value in expected_boundary.items())
    checks.append(
        _result(
            "control:canonical_split_dates",
            "complete" if boundary_matches else "error",
            "Phase 6 uses the canonical 2024/2025 selection and full-year 2026 forward contract"
            if boundary_matches
            else f"expected={expected_boundary}",
        )
    )
    try:
        assert_forward_parent_frozen(
            boundary.get("parent_fit_end"),
            boundary.get("parent_selection_end"),
        )
    except (TypeError, ValueError) as exc:
        checks.append(_result("control:forward_parent_freeze", "error", str(exc)))
    else:
        checks.append(
            _result(
                "control:forward_parent_freeze",
                "complete",
                "model fitting and selection end before full-year 2026 forward",
            )
        )
    activation_flags = [
        activation.get("automatic_trading"),
        activation.get("automatic_retraining"),
        activation.get("automatic_promotion"),
        activation.get("current_forward_activation_allowed"),
    ]
    checks.append(
        _result(
            "control:activation",
            "complete" if all(flag is False for flag in activation_flags) else "error",
            "automatic trading, retraining, promotion, and current activation are disabled"
            if all(flag is False for flag in activation_flags)
            else "all activation flags must be false",
        )
    )
    expected = manifest.get("source_state", {})
    current = source_state(project_root, exclude_paths=(manifest_path.parent, output_dir))
    source_match = (
        expected.get("source_revision") == current.get("source_revision")
        and expected.get("working_tree_status_sha256") == current.get("working_tree_status_sha256")
    )
    checks.append(
        _result(
            "integrity:source_state",
            "complete" if source_match else "error",
            "source revision and working-tree fingerprint match the frozen manifest"
            if source_match
            else "source state changed after freeze",
        )
    )
    return checks


def validate_artifact_index(index_path: Path):
    checks: list[dict[str, str]] = []
    if not index_path.is_file():
        return [_result("integrity:artifact_index", "error", f"missing={index_path}")]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    artifacts = index.get("artifacts", [])
    if not artifacts:
        return [_result("integrity:artifact_index", "error", "artifact index is empty")]
    mismatches = []
    for entry in artifacts:
        descriptor = entry.get("artifact", {})
        path = Path(descriptor.get("path", ""))
        expected_hash = descriptor.get("sha256")
        if not path.is_file() or not expected_hash or sha256_file(path) != expected_hash:
            mismatches.append(str(path))
    checks.append(
        _result(
            "integrity:artifact_hashes",
            "complete" if not mismatches else "error",
            f"validated={len(artifacts)} artifacts"
            if not mismatches
            else f"mismatches={','.join(mismatches)}",
        )
    )
    return checks


def validate_scorecard(manifest_path: str | Path, *, summary_paths: list[str | Path], output_dir: str | Path, project_root: str | Path = ROOT):
    manifest_file = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    project = Path(project_root).resolve()
    checks = validate_manifest_controls(manifest, manifest_path=manifest_file, project_root=project, output_dir=output)
    checks.extend(validate_artifact_index(manifest_file.parent / "artifact_index.json"))

    boundary = manifest["config"]["research_boundary"]
    required_fields = list(manifest["config"]["scorecard_contract"]["required_fields"])
    if not summary_paths:
        checks.append(_result("forward_inputs", "not_ready", "no forward scorecard supplied; shadow remains inactive"))
    for raw_path in summary_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            checks.append(_result(f"forward_input:{path}", "error", "summary file is missing"))
            continue
        frame = pd.read_csv(path)
        checks.extend(
            validate_forward_frame(
                frame,
                path=path,
                required_fields=required_fields,
                research_end=boundary["research_end"],
                forward_start=boundary["forward_start"],
            )
        )
    errors = [check for check in checks if check["status"] == "error"]
    has_forward_input = bool(summary_paths)
    overall = "invalid" if errors else "ready_for_observation" if has_forward_input else "not_ready"
    return {
        "schema_version": 1,
        "manifest": str(manifest_file),
        "overall_status": overall,
        "activation_allowed": False,
        "forward_selection_allowed": False,
        "checks": checks,
    }


def write_outputs(result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "forward_shadow_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Forward Shadow Scorecard Validation",
        "",
        f"- Overall status: **{result['overall_status']}**",
        "- Activation allowed: **false**",
        "- Forward selection allowed: **false**",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {check['name']} | {check['status']} | {check['detail']} |"
        for check in result["checks"]
    )
    (output_dir / "forward_shadow_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--forward-summary", action="append", default=[])
    parser.add_argument("--output-dir", default="reports/experiments/phase6_shadow_scorecard_validation_20260715")
    parser.add_argument("--project-root", default=str(ROOT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = validate_scorecard(
        args.manifest,
        summary_paths=args.forward_summary,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    write_outputs(result, Path(args.output_dir).resolve())
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "status": result["overall_status"]}, ensure_ascii=False))
    if result["overall_status"] == "invalid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
