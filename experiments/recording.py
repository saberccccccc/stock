"""Append-only experiment provenance for research and manual-shadow runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.research_protocol import SPLIT_SPECS, assert_forward_parent_frozen, validate_report_role


MANIFEST_NAME = "experiment_manifest.json"
EVENTS_NAME = "events.jsonl"
ARTIFACT_INDEX_NAME = "artifact_index.json"
SCHEMA_VERSION = 2
VALID_STATUSES = frozenset({"created", "running", "completed", "failed", "superseded", "archived"})
REQUIRED_SCOPE_RANGES = ("feature_warmup", "train", "valid", "signal", "backtest")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def declared_range(start: Any, end: Any, *, status: str = "declared") -> dict[str, Any]:
    return {"status": status, "start": str(start), "end": str(end)}


def not_applicable_range(reason: str) -> dict[str, Any]:
    if not str(reason).strip():
        raise ValueError("not-applicable range requires a reason")
    return {"status": "not_applicable", "reason": str(reason).strip()}


def fingerprint_path(path: str | Path) -> dict[str, Any]:
    """Return a content hash for a file or a deterministic inventory hash for a directory."""

    source = Path(path).expanduser().resolve()
    if source.is_file():
        return {"kind": "file_sha256", "value": sha256_file(source)}
    if not source.is_dir():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for item in sorted((item for item in source.rglob("*") if item.is_file()), key=lambda p: str(p)):
        stat = item.stat()
        relative = item.relative_to(source).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
        count += 1
        total_bytes += int(stat.st_size)
    return {
        "kind": "inventory_sha256",
        "value": digest.hexdigest(),
        "file_count": count,
        "total_bytes": total_bytes,
    }


def _scope_date(value: Any, field: str) -> date:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field} is required")
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date, got {value!r}") from exc


def _validate_scope_range(name: str, value: Any) -> tuple[date, date] | None:
    if not isinstance(value, Mapping):
        raise ValueError(f"experiment_scope.ranges.{name} must be an object")
    status = str(value.get("status", "")).strip()
    if status == "not_applicable":
        if not str(value.get("reason", "")).strip():
            raise ValueError(f"experiment_scope.ranges.{name} requires a not-applicable reason")
        return None
    if status not in {"declared", "actual"}:
        raise ValueError(f"experiment_scope.ranges.{name} has invalid status={status!r}")
    start = _scope_date(value.get("start"), f"ranges.{name}.start")
    end = _scope_date(value.get("end"), f"ranges.{name}.end")
    if start > end:
        raise ValueError(f"experiment_scope.ranges.{name} start exceeds end")
    return start, end


def validate_experiment_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the mandatory provenance contract for a formal experiment."""

    if not isinstance(scope, Mapping):
        raise ValueError("formal experiment_scope must be an object")
    stage = str(scope.get("stage", "")).strip()
    if not stage:
        raise ValueError("experiment_scope.stage is required")

    data_sources = scope.get("data_sources")
    if not isinstance(data_sources, list) or not data_sources:
        raise ValueError("experiment_scope.data_sources must contain at least one source")
    for index, source in enumerate(data_sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"data_sources[{index}] must be an object")
        for field in ("role", "root", "fingerprint"):
            if source.get(field) in (None, "", {}):
                raise ValueError(f"data_sources[{index}].{field} is required")

    ranges = scope.get("ranges")
    if not isinstance(ranges, Mapping):
        raise ValueError("experiment_scope.ranges must be an object")
    parsed_ranges = {
        name: _validate_scope_range(name, ranges.get(name))
        for name in REQUIRED_SCOPE_RANGES
    }
    max_data_date = _scope_date(scope.get("max_data_date"), "max_data_date")
    for name, bounds in parsed_ranges.items():
        if bounds is not None and bounds[1] > max_data_date:
            raise ValueError(f"ranges.{name}.end exceeds max_data_date")

    split_roles = scope.get("split_roles")
    if not isinstance(split_roles, list) or not split_roles:
        raise ValueError("experiment_scope.split_roles must contain at least one split")
    has_forward = False
    for item in split_roles:
        if not isinstance(item, Mapping):
            raise ValueError("split_roles entries must be objects")
        split = str(item.get("split", "")).strip()
        if split in SPLIT_SPECS:
            spec = validate_report_role(
                split,
                selection_eligible=item.get("selection_eligible"),
                is_forward=item.get("forward_used"),
            )
            has_forward = has_forward or spec.is_forward
        else:
            if not split or not str(item.get("role", "")).strip():
                raise ValueError("custom split role requires split and role")
            if _as_scope_bool(item.get("selection_eligible"), "selection_eligible"):
                raise ValueError("custom historical split cannot be selection-eligible")
            if _as_scope_bool(item.get("forward_used"), "forward_used"):
                raise ValueError("custom historical split cannot be marked forward")

    transform = scope.get("transform")
    if not isinstance(transform, Mapping) or not str(transform.get("state_sha256", "")).strip():
        raise ValueError("experiment_scope.transform.state_sha256 is required")
    _validate_scope_range("transform.fit_range", transform.get("fit_range"))

    lineage = scope.get("lineage", {})
    if has_forward:
        if not isinstance(lineage, Mapping):
            raise ValueError("forward experiment_scope.lineage must be an object")
        assert_forward_parent_frozen(
            lineage.get("parent_fit_end"),
            lineage.get("parent_selection_end"),
        )
    return {
        "complete": True,
        "stage": stage,
        "data_source_count": len(data_sources),
        "split_count": len(split_roles),
        "has_forward": has_forward,
    }


def _as_scope_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} must be boolean")


def validate_manifest_for_formal_use(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    require_artifacts: bool = True,
    require_completed: bool = True,
    require_current_index: bool = True,
) -> dict[str, Any]:
    manifest_path = None
    if isinstance(manifest_or_path, Mapping):
        manifest = dict(manifest_or_path)
    else:
        manifest_path = Path(manifest_or_path).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("experiment_class") != "formal":
        raise ValueError("experiment is not marked formal")
    result = validate_experiment_scope(manifest.get("experiment_scope", {}))
    if manifest.get("formal_completeness", {}).get("complete") is not True:
        raise ValueError("formal completeness marker is missing")
    if require_artifacts:
        if manifest_path is None:
            raise ValueError("artifact validation requires a manifest path")
        events = load_events(manifest_path.parent)
        if require_completed and (not events or events[-1].get("status") != "completed"):
            terminal = events[-1].get("status") if events else "missing"
            raise ValueError(f"formal experiment is not completed: terminal_status={terminal}")
        index_path = manifest_path.parent / ARTIFACT_INDEX_NAME
        if not index_path.is_file():
            raise ValueError(f"formal artifact index is missing: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8-sig"))
        if require_current_index and int(index.get("events", -1)) != len(events):
            raise ValueError("formal artifact index is stale relative to the event log")
        artifacts = index.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("formal artifact index is empty")
        for entry in artifacts:
            descriptor = entry.get("artifact", {})
            path = Path(descriptor.get("path", ""))
            expected = descriptor.get("sha256")
            if not path.is_file() or not expected or sha256_file(path) != expected:
                raise ValueError(f"formal artifact hash mismatch: {path}")
    return result


def infer_data_scope(protocol: Mapping[str, Any], cache_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return a uniform data-range record without guessing missing dates."""
    explicit = cache_contract.get("data_scope")
    if explicit is not None:
        return dict(explicit) if isinstance(explicit, Mapping) else {"value": explicit}

    effective_start = cache_contract.get("data_start") or cache_contract.get("effective_start")
    effective_end = (
        cache_contract.get("effective_end")
        or cache_contract.get("data_end")
        or protocol.get("research_end")
    )
    missing = []
    if not effective_start:
        missing.append("effective_start")
    if not effective_end:
        missing.append("effective_end")
    return {
        "role": "research" if protocol.get("research_end") else "unspecified",
        "effective_start": str(effective_start) if effective_start else None,
        "effective_end": str(effective_end) if effective_end else None,
        "physical_coverage": None,
        "complete": not missing,
        "missing_fields": missing,
        "source": "inferred_from_protocol_and_cache_contract",
    }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(project_root: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _status_line_paths(line: str) -> Iterable[str]:
    """Yield paths represented by one porcelain-v1 status line."""
    payload = line[3:] if len(line) >= 4 else line
    parts = payload.split(" -> ") if " -> " in payload else [payload]
    for part in parts:
        value = part.strip().strip('"')
        if value:
            yield value


def _is_excluded_status_line(line: str, project_root: Path, excluded: tuple[Path, ...]) -> bool:
    if not excluded:
        return False
    for raw_path in _status_line_paths(line):
        path = Path(raw_path)
        candidate = path if path.is_absolute() else project_root / path
        resolved = candidate.resolve()
        for excluded_path in excluded:
            try:
                resolved.relative_to(excluded_path)
            except ValueError:
                continue
            else:
                return True
    return False


def source_state(
    project_root: str | Path,
    *,
    exclude_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Return revision plus a compact fingerprint of the working-tree state."""
    root = Path(project_root)
    excluded = tuple(Path(path).resolve() for path in exclude_paths)
    revision = source_revision(root)
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        lines = [
            line
            for line in result.stdout.splitlines()
            if not _is_excluded_status_line(line, root, excluded)
        ]
        status_text = "\n".join(lines)
        if result.stdout.endswith("\n") and status_text:
            status_text += "\n"
        status_available = True
    except (OSError, subprocess.SubprocessError):
        status_text = ""
        status_available = False
    encoded = status_text.encode("utf-8")
    return {
        "source_revision": revision,
        "working_tree_dirty": bool(status_text.strip()) if status_available else None,
        "working_tree_status_sha256": hashlib.sha256(encoded).hexdigest(),
        "working_tree_status_entries": len(status_text.splitlines()) if status_available else None,
        "status_available": status_available,
    }


def artifact_descriptor(path: str | Path, *, include_sha256: bool = True) -> dict[str, Any]:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise ValueError(f"artifact must be a file: {artifact}")
    stat = artifact.stat()
    value: dict[str, Any] = {
        "path": str(artifact),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        value["sha256"] = sha256_file(artifact)
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path


def create_experiment(
    output_dir: str | Path,
    *,
    experiment_id: str,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    cache_contract: Mapping[str, Any],
    project_root: str | Path,
    parent_experiment_ids: tuple[str, ...] = (),
    formal: bool = False,
    experiment_scope: Mapping[str, Any] | None = None,
) -> Path:
    """Create the immutable experiment manifest and initial status event."""
    if not experiment_id or experiment_id.strip() != experiment_id:
        raise ValueError("experiment_id must be a non-empty trimmed string")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source = source_state(project_root, exclude_paths=(root,))
    completeness = validate_experiment_scope(experiment_scope or {}) if formal else {
        "complete": False,
        "reason": "exploratory_manifest_not_eligible_for_formal_ranking",
    }
    data_scope = (
        {
            "source": "formal_experiment_scope",
            "complete": True,
            "data_sources": list((experiment_scope or {}).get("data_sources", [])),
            "ranges": dict((experiment_scope or {}).get("ranges", {})),
            "max_data_date": (experiment_scope or {}).get("max_data_date"),
        }
        if formal
        else infer_data_scope(protocol, cache_contract)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "source_revision": source["source_revision"],
        "source_state": source,
        "experiment_class": "formal" if formal else "exploratory",
        "experiment_scope": dict(experiment_scope or {}),
        "formal_completeness": completeness,
        "config": dict(config),
        "config_sha256": canonical_json_hash(config),
        "protocol": dict(protocol),
        "cache_contract": dict(cache_contract),
        "data_scope": data_scope,
        "parent_experiment_ids": list(parent_experiment_ids),
    }
    path = _write_json_exclusive(root / MANIFEST_NAME, manifest)
    append_event(root, status="created", event_type="experiment_created")
    return path


def append_event(
    output_dir: str | Path,
    *,
    status: str,
    event_type: str,
    details: Mapping[str, Any] | None = None,
) -> Path:
    """Append an immutable status or artifact event to an existing experiment."""
    if status not in VALID_STATUSES:
        raise ValueError(f"unknown experiment status: {status}")
    root = Path(output_dir)
    if not (root / MANIFEST_NAME).is_file():
        raise FileNotFoundError(f"experiment manifest is missing under {root}")
    event = {
        "at": utc_now(),
        "status": status,
        "event_type": event_type,
        "details": dict(details or {}),
    }
    path = root / EVENTS_NAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return path


def load_events(output_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(output_dir) / EVENTS_NAME
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_artifact(
    output_dir: str | Path,
    *,
    name: str,
    path: str | Path,
    kind: str,
    include_sha256: bool = True,
) -> Path:
    if not name or not kind:
        raise ValueError("artifact name and kind are required")
    descriptor = artifact_descriptor(path, include_sha256=include_sha256)
    return append_event(
        output_dir,
        status="running",
        event_type="artifact_recorded",
        details={"name": name, "kind": kind, "artifact": descriptor},
    )


def finalize_artifact_index(output_dir: str | Path) -> Path:
    """Atomically rebuild the artifact index from immutable append-only events."""
    root = Path(output_dir)
    artifacts = []
    for event in load_events(root):
        if event.get("event_type") == "artifact_recorded":
            artifacts.append(event["details"])
    target = root / ARTIFACT_INDEX_NAME
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": artifacts,
        "events": len(load_events(root)),
        "built_at": utc_now(),
    }
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(target)
    return target
