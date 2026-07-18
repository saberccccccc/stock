import json

import pandas as pd
import pytest

from alpha.io import (
    assert_same_date,
    iter_aligned_alpha_rows,
    iter_alpha_rows,
    load_alpha_dates,
    load_alpha_manifest,
    load_alpha_rows,
    resolve_alpha_source,
    write_alpha_rows,
)


def test_iter_alpha_rows_normalizes_dates(tmp_path):
    path = tmp_path / "alpha.jsonl"
    path.write_text(
        json.dumps({"date": "2024/01/02", "codes": ["A"]}) + "\n\n",
        encoding="utf-8",
    )

    rows = list(iter_alpha_rows(path))

    assert rows == [{"date": "2024-01-02", "codes": ["A"]}]


def test_iter_alpha_rows_accepts_utf8_bom(tmp_path):
    path = tmp_path / "alpha_bom.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf" + json.dumps({"date": "2024-01-02", "codes": ["A"]}).encode("utf-8")
    )

    rows = list(iter_alpha_rows(path))

    assert rows == [{"date": "2024-01-02", "codes": ["A"]}]


def test_load_alpha_dates_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="alpha file is empty"):
        load_alpha_dates(path)


def test_load_alpha_rows_rejects_duplicate_dates(tmp_path):
    path = tmp_path / "duplicate_dates.jsonl"
    write_alpha_rows(path, [
        {"date": "2024-01-02", "codes": ["A"]},
        {"date": "2024-01-02", "codes": ["B"]},
    ])

    with pytest.raises(ValueError, match="duplicate alpha date"):
        load_alpha_rows(path)


def test_load_alpha_rows_rejects_misaligned_or_duplicate_codes(tmp_path):
    mismatch = tmp_path / "mismatch.jsonl"
    duplicate = tmp_path / "duplicate_codes.jsonl"
    write_alpha_rows(mismatch, [
        {"date": "2024-01-02", "codes": ["A", "B"], "alpha": [1.0]},
    ])
    write_alpha_rows(duplicate, [
        {"date": "2024-01-02", "codes": ["A", "A"], "alpha": [1.0, 0.5]},
    ])

    with pytest.raises(ValueError, match="mismatched codes and alpha lengths"):
        load_alpha_rows(mismatch)
    with pytest.raises(ValueError, match="duplicate stock codes"):
        load_alpha_rows(duplicate)


def test_write_alpha_rows_creates_parent(tmp_path):
    output = tmp_path / "nested" / "alpha.jsonl"

    write_alpha_rows(output, [{"date": "2024-01-02", "codes": ["A"], "alpha": [1.0]}])

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["codes"] == ["A"]


def test_assert_same_date_accepts_timestamp_and_string():
    left = {"date": pd.Timestamp("2024-01-02")}
    right = {"date": "2024-01-02"}

    assert assert_same_date(left, right) == "2024-01-02"


def test_assert_same_date_raises_on_mismatch():
    with pytest.raises(ValueError, match="date mismatch"):
        assert_same_date({"date": "2024-01-02"}, {"date": "2024-01-03"})


def test_iter_aligned_alpha_rows_rejects_different_lengths(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    write_alpha_rows(left, [{"date": "2024-01-02"}, {"date": "2024-01-03"}])
    write_alpha_rows(right, [{"date": "2024-01-02"}])

    with pytest.raises(ValueError, match="different row counts"):
        list(iter_aligned_alpha_rows(left, right))


def test_load_alpha_manifest_resolves_relative_paths(tmp_path):
    alpha_dir = tmp_path / "signals"
    alpha_dir.mkdir()
    alpha_path = alpha_dir / "base.jsonl"
    write_alpha_rows(alpha_path, [{"date": "2024-01-02", "codes": ["A"]}])
    manifest_path = tmp_path / "alpha_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "default_alpha_jsonl": "signals/base.jsonl",
                "rules": [
                    {
                        "name": "small_capital",
                        "max_portfolio_value": 750000,
                        "alpha_jsonl": "signals/base.jsonl",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_alpha_manifest(manifest_path)

    assert manifest["manifest_path"] == str(manifest_path.resolve())
    assert manifest["default_alpha_jsonl"] == str(alpha_path.resolve())
    assert manifest["rules"][0]["alpha_jsonl"] == str(alpha_path.resolve())


def test_resolve_alpha_source_supports_direct_jsonl(tmp_path):
    alpha_path = tmp_path / "direct.jsonl"
    write_alpha_rows(alpha_path, [{"date": "2024-01-02", "codes": ["A"]}])

    resolved = resolve_alpha_source(alpha_jsonl=alpha_path)

    assert resolved.mode == "direct"
    assert resolved.request_path == str(alpha_path)
    assert resolved.alpha_jsonl == str(alpha_path.resolve())
    assert resolved.manifest_path == ""


def test_resolve_alpha_source_uses_matching_rule_and_default(tmp_path):
    default_alpha = tmp_path / "default.jsonl"
    large_alpha = tmp_path / "large.jsonl"
    write_alpha_rows(default_alpha, [{"date": "2024-01-02", "codes": ["A"]}])
    write_alpha_rows(large_alpha, [{"date": "2024-01-03", "codes": ["B"]}])
    manifest_path = tmp_path / "alpha_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "default_alpha_jsonl": "default.jsonl",
                "rules": [
                    {
                        "name": "large_only",
                        "min_portfolio_value": 1000000,
                        "alpha_jsonl": "large.jsonl",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    default_resolved = resolve_alpha_source(alpha_manifest=manifest_path, portfolio_value=500000)
    large_resolved = resolve_alpha_source(alpha_manifest=manifest_path, portfolio_value=1000000)

    assert default_resolved.rule_name == "default"
    assert default_resolved.alpha_jsonl == str(default_alpha.resolve())
    assert large_resolved.rule_name == "large_only"
    assert large_resolved.alpha_jsonl == str(large_alpha.resolve())
    assert large_resolved.min_portfolio_value == 1000000.0


def test_resolve_alpha_source_rejects_unmatched_manifest_without_default(tmp_path):
    alpha_path = tmp_path / "large.jsonl"
    write_alpha_rows(alpha_path, [{"date": "2024-01-03", "codes": ["B"]}])
    manifest_path = tmp_path / "alpha_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "name": "large_only",
                        "min_portfolio_value": 1000000,
                        "alpha_jsonl": "large.jsonl",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no rule for portfolio_value"):
        resolve_alpha_source(alpha_manifest=manifest_path, portfolio_value=500000)
