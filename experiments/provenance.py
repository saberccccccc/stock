"""Immutable reproducibility manifests and low-overhead runtime sampling."""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.recording import ARTIFACT_INDEX_NAME, canonical_json_hash, sha256_file, source_state


PROVENANCE_FILES = (
    "environment_manifest.json",
    "source_manifest.json",
    "data_manifest.json",
    "feature_transform_manifest.json",
    "command_manifest.json",
    "runtime_metrics.json",
)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value["manifest_sha256"] = canonical_json_hash(value)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path


def environment_manifest(project_root: str | Path) -> dict[str, Any]:
    import lightgbm
    import psutil
    import torch

    root = Path(project_root).resolve()
    dependency_files = {}
    for name in ("requirements.txt", "environment.yml", "pyproject.toml", "uv.lock"):
        path = root / name
        if path.is_file():
            dependency_files[name] = {"path": str(path), "sha256": sha256_file(path)}
    cuda_available = bool(torch.cuda.is_available())
    cuda = {
        "available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else [],
    }
    memory = psutil.virtual_memory()
    return {
        "schema": "environment_manifest_v1",
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "ram_bytes": int(memory.total),
        },
        "packages": {
            "torch": torch.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "psutil": psutil.__version__,
        },
        "cuda": cuda,
        "dependency_files": dependency_files,
    }


def source_manifest(
    project_root: str | Path,
    source_paths: Sequence[str | Path],
    *,
    exclude_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    entries = []
    seen = set()
    for raw in source_paths:
        path = Path(raw)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            display = path.relative_to(root).as_posix()
        except ValueError:
            display = str(path)
        entries.append(
            {"path": str(path), "project_relative_path": display, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    if not entries:
        raise ValueError("source manifest requires participating source files")
    return {
        "schema": "source_manifest_v1",
        "repository": source_state(root, exclude_paths=exclude_paths),
        "participating_files": sorted(entries, key=lambda item: item["project_relative_path"]),
    }


def command_manifest(argv: Sequence[str], resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    if not argv:
        raise ValueError("command manifest requires argv")
    return {
        "schema": "command_manifest_v1",
        "cwd": str(Path.cwd().resolve()),
        "argv": [str(value) for value in argv],
        "resolved_config": dict(resolved_config),
        "resolved_config_sha256": canonical_json_hash(resolved_config),
    }


class RuntimeMetricsSampler:
    """Sample one runner process without changing child training semantics."""

    def __init__(self, *, interval_seconds: float = 0.5):
        self.interval_seconds = float(interval_seconds)
        if self.interval_seconds <= 0:
            raise ValueError("runtime sampling interval must be positive")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._samples = 0
        self._peak_rss = 0
        self._peak_tree_rss = 0
        self._peak_child_count = 0
        self._minimum_available = None
        self._start_cpu = None

    def __enter__(self):
        import psutil
        import torch

        process = psutil.Process(os.getpid())
        self._started_at = time.perf_counter()
        self._start_cpu = process.cpu_times()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def sample():
            while not self._stop.is_set():
                try:
                    rss = process.memory_info().rss
                    children = process.children(recursive=True)
                    tree_rss = int(rss)
                    for child in children:
                        try:
                            tree_rss += int(child.memory_info().rss)
                        except psutil.Error:
                            pass
                    available = psutil.virtual_memory().available
                    self._peak_rss = max(self._peak_rss, int(rss))
                    self._peak_tree_rss = max(self._peak_tree_rss, tree_rss)
                    self._peak_child_count = max(self._peak_child_count, len(children))
                    self._minimum_available = (
                        int(available)
                        if self._minimum_available is None
                        else min(self._minimum_available, int(available))
                    )
                    self._samples += 1
                except psutil.Error:
                    pass
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample, name="runtime-metrics", daemon=True)
        self._thread.start()
        return self

    def finish(self, *, status: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
        import psutil
        import torch

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
        process = psutil.Process(os.getpid())
        cpu = process.cpu_times()
        result = {
            "schema": "runtime_metrics_v1",
            "status": str(status),
            "wall_seconds": float(time.perf_counter() - self._started_at),
            "process_peak_rss_bytes": int(self._peak_rss),
            "process_tree_peak_rss_bytes": int(self._peak_tree_rss),
            "peak_child_processes": int(self._peak_child_count),
            "system_min_available_bytes": int(self._minimum_available or 0),
            "cpu_user_seconds": float(cpu.user - self._start_cpu.user),
            "cpu_system_seconds": float(cpu.system - self._start_cpu.system),
            "sample_interval_seconds": self.interval_seconds,
            "samples": int(self._samples),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            "details": dict(details or {}),
        }
        return result

    def __exit__(self, exc_type, exc, traceback):
        if not self._stop.is_set():
            self.finish(status="failed" if exc_type else "completed")


def write_provenance_bundle(
    output_dir: str | Path,
    *,
    environment: Mapping[str, Any],
    source: Mapping[str, Any],
    data: Mapping[str, Any],
    feature_transform: Mapping[str, Any],
    command: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> Path:
    root = Path(output_dir).resolve()
    payloads = {
        "environment_manifest.json": environment,
        "source_manifest.json": source,
        "data_manifest.json": data,
        "feature_transform_manifest.json": feature_transform,
        "command_manifest.json": command,
        "runtime_metrics.json": runtime,
    }
    records = {}
    for name in PROVENANCE_FILES:
        path = _write_exclusive(root / name, payloads[name])
        records[name] = {"path": str(path), "sha256": sha256_file(path)}
    bundle = {
        "schema": "provenance_bundle_v1",
        "complete": True,
        "artifacts": records,
    }
    bundle["bundle_sha256"] = canonical_json_hash(bundle)
    return _write_exclusive(root / "provenance_bundle.json", bundle)


def validate_provenance_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path).resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))
    if bundle.get("schema") != "provenance_bundle_v1" or bundle.get("complete") is not True:
        raise ValueError("incomplete or unsupported provenance bundle")
    artifacts = bundle.get("artifacts", {})
    if set(artifacts) != set(PROVENANCE_FILES):
        raise ValueError("provenance bundle does not contain the six mandatory manifests")
    for name, descriptor in artifacts.items():
        source = Path(descriptor["path"])
        if not source.is_file() or sha256_file(source) != descriptor["sha256"]:
            raise ValueError(f"provenance artifact mismatch: {name}")
    return {"complete": True, "artifacts": len(artifacts), "bundle": str(bundle_path)}


def validate_provenance_artifact_index(experiment_dir: str | Path) -> dict[str, Any]:
    root = Path(experiment_dir).resolve()
    bundle_path = root / "provenance" / "provenance_bundle.json"
    bundle_result = validate_provenance_bundle(bundle_path)
    index_path = root / ARTIFACT_INDEX_NAME
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    names = {
        str(item.get("name"))
        for item in index.get("artifacts", [])
        if isinstance(item, Mapping)
    }
    required = {f"provenance:{Path(name).stem}" for name in PROVENANCE_FILES}
    required.add("provenance:provenance_bundle")
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"artifact index is missing provenance records: {missing}")
    return {**bundle_result, "indexed_artifacts": len(required)}
