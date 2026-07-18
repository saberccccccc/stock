"""Freeze a monthly rolling model config from an existing base config and v14 calendar."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.labels import label_end_offset
from experiments.recording import canonical_json_hash, sha256_file
from experiments.rolling import (
    FixedRollingSpec,
    assert_unique_oos_owners,
    build_fixed_windows,
    resolve_window_indices,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--v14-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--valid-months", type=int, default=6)
    parser.add_argument("--oos-months", type=int, default=1)
    parser.add_argument("--oos-start", default="2024-01-01")
    parser.add_argument("--oos-end", default="2025-12-31")
    return parser.parse_args(argv)


def build_config(base, meta, *, train_years, valid_months, oos_start, oos_end, oos_months=1):
    schedule = FixedRollingSpec(train_years, valid_months, oos_months, oos_start, oos_end)
    windows = build_fixed_windows(meta["all_dates"], schedule)
    offset = label_end_offset(base["data"]["label_family"], int(base["data"]["horizon_index"]))
    counts = []
    for window in windows:
        indices = resolve_window_indices(meta["all_dates"], window, offset)
        if not indices["train"] or not indices["valid"] or not indices["predict"]:
            raise ValueError(f"monthly window has empty label-safe segment: {window.name}")
        counts.append({"name": window.name, **{key: len(value) for key, value in indices.items()}})
    owners = assert_unique_oos_owners(meta["all_dates"], windows)
    config = dict(base)
    config["name"] = f"{base.get('name', 'rolling')}_rolling_{train_years}y_{valid_months}m_{oos_months}m"
    config["purpose"] = (
        "Predeclared walk-forward research arm: fixed history, label-tail purge, "
        "and one unique model owner per OOS date."
    )
    config["windows"] = [window.__dict__ for window in windows]
    config["monthly_schedule"] = {
        "train_years": int(train_years),
        "valid_months": int(valid_months),
        "oos_months": int(oos_months),
        "oos_start": oos_start,
        "oos_end": oos_end,
        "label_end_offset": int(offset),
        "window_count": len(windows),
        "unique_oos_dates": len(owners),
        "window_counts": counts,
    }
    return config


def main(argv=None):
    args = parse_args(argv)
    base_path = Path(args.base_config).resolve()
    meta_path = Path(args.v14_meta).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8-sig"))
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    config = build_config(
        base,
        meta,
        train_years=args.train_years,
        valid_months=args.valid_months,
        oos_months=args.oos_months,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
    )
    config["monthly_schedule"]["base_config_sha256"] = sha256_file(base_path)
    config["monthly_schedule"]["v14_meta_sha256"] = sha256_file(meta_path)
    config["monthly_schedule"]["config_sha256_before_self_field"] = canonical_json_hash(config)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "windows": config["monthly_schedule"]["window_count"],
                "unique_oos_dates": config["monthly_schedule"]["unique_oos_dates"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
