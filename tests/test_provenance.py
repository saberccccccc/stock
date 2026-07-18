import json
import sys
import time
from pathlib import Path

import pytest

from experiments.provenance import (
    PROVENANCE_FILES,
    RuntimeMetricsSampler,
    command_manifest,
    environment_manifest,
    source_manifest,
    validate_provenance_bundle,
    validate_provenance_artifact_index,
    write_provenance_bundle,
)


def test_environment_source_and_command_manifests_capture_replay_inputs(tmp_path):
    source = tmp_path / "runner.py"
    source.write_text("print('ok')\n", encoding="utf-8")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy\n", encoding="utf-8")

    environment = environment_manifest(tmp_path)
    sources = source_manifest(tmp_path, [source])
    command = command_manifest([sys.executable, str(source), "--flag", "1"], {"flag": 1})

    assert environment["schema"] == "environment_manifest_v1"
    assert environment["dependency_files"]["requirements.txt"]["sha256"]
    assert sources["participating_files"][0]["project_relative_path"] == "runner.py"
    assert sources["participating_files"][0]["sha256"]
    assert command["argv"][-2:] == ["--flag", "1"]
    assert command["resolved_config_sha256"]


def test_runtime_sampler_records_resource_metrics():
    sampler = RuntimeMetricsSampler(interval_seconds=0.01)
    with sampler:
        values = bytearray(1024 * 1024)
        time.sleep(0.04)
        assert values
        metrics = sampler.finish(status="completed", details={"cache_hit": True})

    assert metrics["status"] == "completed"
    assert metrics["wall_seconds"] > 0
    assert metrics["process_peak_rss_bytes"] > 0
    assert metrics["process_tree_peak_rss_bytes"] >= metrics["process_peak_rss_bytes"]
    assert metrics["system_min_available_bytes"] > 0
    assert metrics["samples"] >= 1
    assert metrics["details"]["cache_hit"] is True


def test_provenance_bundle_requires_six_hash_valid_artifacts(tmp_path):
    payload = {"schema": "test_v1", "value": 1}
    bundle = write_provenance_bundle(
        tmp_path,
        environment=payload,
        source=payload,
        data=payload,
        feature_transform=payload,
        command=payload,
        runtime=payload,
    )

    assert validate_provenance_bundle(bundle)["artifacts"] == 6
    manifest = json.loads(bundle.read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == set(PROVENANCE_FILES)

    (tmp_path / "runtime_metrics.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact mismatch"):
        validate_provenance_bundle(bundle)


def test_provenance_artifact_index_requires_bundle_and_all_six_manifests(tmp_path):
    experiment = tmp_path / "experiment"
    provenance = experiment / "provenance"
    payload = {"schema": "test_v1"}
    bundle = write_provenance_bundle(
        provenance,
        environment=payload,
        source=payload,
        data=payload,
        feature_transform=payload,
        command=payload,
        runtime=payload,
    )
    artifacts = [
        {"name": f"provenance:{path.stem}", "artifact": {"path": str(path)}}
        for path in provenance.glob("*.json")
    ]
    (experiment / "artifact_index.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )

    result = validate_provenance_artifact_index(experiment)

    assert result["indexed_artifacts"] == 7
    assert result["bundle"] == str(bundle)
