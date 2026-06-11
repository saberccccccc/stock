#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small cache helpers for V9 alpha and price/volume matrices."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_alpha_rows_jsonl(alpha_rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in alpha_rows:
            out = dict(row)
            out["date"] = str(pd.Timestamp(out["date"]).date())
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def load_alpha_rows_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["date"] = pd.Timestamp(row["date"])
            rows.append(row)
    rows.sort(key=lambda r: pd.Timestamp(r["date"]))
    return rows


def save_universe_matrix_cache(path, price_mat, vol_mat, all_dates, all_codes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        price_mat=np.asarray(price_mat, dtype=np.float64),
        vol_mat=np.asarray(vol_mat, dtype=np.float64),
        all_dates=np.asarray([str(pd.Timestamp(d).date()) for d in all_dates], dtype=object),
        all_codes=np.asarray(list(all_codes), dtype=object),
    )


def load_universe_matrix_cache(path):
    data = np.load(Path(path), allow_pickle=True)
    price_mat = data["price_mat"].astype(np.float64, copy=False)
    vol_mat = data["vol_mat"].astype(np.float64, copy=False)
    all_dates = pd.DatetimeIndex(pd.to_datetime(data["all_dates"].astype(str)))
    all_codes = [str(x) for x in data["all_codes"].tolist()]
    code2idx = {code: i for i, code in enumerate(all_codes)}
    return price_mat, vol_mat, all_dates, code2idx, all_codes
