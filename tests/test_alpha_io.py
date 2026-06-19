import json

import pandas as pd
import pytest

from alpha.io import (
    assert_same_date,
    iter_aligned_alpha_rows,
    iter_alpha_rows,
    load_alpha_dates,
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
