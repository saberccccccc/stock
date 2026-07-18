import json

import pytest

from alpha.io import load_alpha_rows, write_alpha_rows
from experiments.recording import sha256_file
from run.rolling_lgbm_alpha import (
    build_split_alpha_files,
    load_rolling_progress,
    write_rolling_progress,
)


def _alpha(path, date):
    write_alpha_rows(path, [{"date": date, "codes": ["000001.SZ"], "alpha": [1.0]}])
    return str(path)


def test_stitch_monthly_alpha_into_official_split_files(tmp_path):
    first = _alpha(tmp_path / "jan.jsonl", "2024-01-02")
    second = _alpha(tmp_path / "feb.jsonl", "2024-02-01")
    third = _alpha(tmp_path / "jan25.jsonl", "2025-01-02")

    result = build_split_alpha_files(
        [
            {"name": "oos_2024_01", "alpha_path": first},
            {"name": "oos_2024_02", "alpha_path": second},
            {"name": "oos_2025_01", "alpha_path": third},
        ],
        tmp_path / "out",
    )

    assert result["val_2024"]["rows"] == 2
    assert result["test_2025"]["rows"] == 1
    rows = load_alpha_rows(result["val_2024"]["path"])
    assert [row["owner_window"] for row in rows] == ["oos_2024_01", "oos_2024_02"]


def test_stitch_rejects_duplicate_oos_owner(tmp_path):
    first = _alpha(tmp_path / "a.jsonl", "2024-01-02")
    second = _alpha(tmp_path / "b.jsonl", "2024-01-02")

    with pytest.raises(ValueError, match="duplicate stitched OOS"):
        build_split_alpha_files(
            [{"name": "a", "alpha_path": first}, {"name": "b", "alpha_path": second}],
            tmp_path / "out",
        )


def _progress_entry(tmp_path):
    alpha_path = tmp_path / "alpha.jsonl"
    model_path = tmp_path / "model.txt"
    alpha_path.write_text('{"date":"2024-01-02"}\n', encoding="utf-8")
    model_path.write_text("tree\n", encoding="utf-8")
    return {
        "name": "oos_2024_01",
        "alpha_path": str(alpha_path),
        "model_path": str(model_path),
        "alpha_sha256": sha256_file(alpha_path),
        "model_sha256": sha256_file(model_path),
    }


def test_rolling_progress_round_trip(tmp_path):
    progress_path = tmp_path / "rolling_progress.json"
    entry = _progress_entry(tmp_path)

    write_rolling_progress(progress_path, config_sha256="frozen", window_entries=[entry])

    assert load_rolling_progress(progress_path, config_sha256="frozen") == [entry]


def test_rolling_progress_rejects_config_change(tmp_path):
    progress_path = tmp_path / "rolling_progress.json"
    write_rolling_progress(progress_path, config_sha256="old", window_entries=[_progress_entry(tmp_path)])

    with pytest.raises(ValueError, match="config hash mismatch"):
        load_rolling_progress(progress_path, config_sha256="new")


def test_rolling_progress_rejects_artifact_tampering(tmp_path):
    progress_path = tmp_path / "rolling_progress.json"
    entry = _progress_entry(tmp_path)
    write_rolling_progress(progress_path, config_sha256="frozen", window_entries=[entry])
    (tmp_path / "model.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact mismatch"):
        load_rolling_progress(progress_path, config_sha256="frozen")
