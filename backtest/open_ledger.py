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


def limit_new_names(
    selected,
    kept,
    row,
    max_new_names,
    current_selected,
    target_n,
    exit_hold_frac=None,
    switch_gap_frac=0.0,
):
    if max_new_names <= 0 or not current_selected:
        return selected
    codes = list(row.get("codes", []))
    rank_map = {code: rank for rank, code in enumerate(codes)}
    if exit_hold_frac is not None and exit_hold_frac > 0:
        exit_n = max(int(len(codes) * float(exit_hold_frac)), target_n)
        eligible_current = [
            code for code in current_selected
            if rank_map.get(code, len(codes) + 1) < exit_n
        ]
    else:
        eligible_current = [code for code in current_selected if code in rank_map]
    current_ranked = sorted(
        eligible_current,
        key=lambda code: rank_map[code],
    )
    target_n = max(int(target_n), 1)
    max_new_names = max(int(max_new_names), 0)
    min_old = max(target_n - max_new_names, 0)
    switch_gap = max(int(len(codes) * float(switch_gap_frac)), 0)
    limited = current_ranked[:min(len(current_ranked), min_old)]
    selected_set = set(limited)
    added = 0
    current_set = set(current_selected)
    replacement_old = current_ranked[len(limited):]
    replacement_slot = 0
    for code in codes:
        if len(limited) >= target_n:
            break
        if code in selected_set:
            continue
        if code in current_set:
            continue
        if added >= max_new_names:
            break
        if switch_gap > 0 and replacement_slot < len(replacement_old):
            old_code = replacement_old[replacement_slot]
            if rank_map[code] + switch_gap >= rank_map[old_code]:
                continue
        limited.append(code)
        selected_set.add(code)
        added += 1
        replacement_slot += 1
    for code in current_ranked:
        if len(limited) >= target_n:
            break
        if code not in selected_set:
            limited.append(code)
            selected_set.add(code)
    for code in codes:
        if len(limited) >= target_n:
            break
        if code not in selected_set:
            limited.append(code)
            selected_set.add(code)
    return limited
