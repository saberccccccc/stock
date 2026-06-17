"""Reusable helpers for open-price share-ledger backtests."""

import json
from pathlib import Path

import pandas as pd


def parse_float_list(raw):
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def load_alpha_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["date"] = pd.Timestamp(row["date"])
            rows.append(row)
    rows.sort(key=lambda row: row["date"])
    return rows
