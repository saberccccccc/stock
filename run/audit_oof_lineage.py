"""Audit yearly rank diversity and coverage for a validated OOF blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha.io import load_alpha_rows
from experiments.oof_lineage import load_lineage_manifest


def _rows_by_date(path):
    return {
        pd.Timestamp(row["date"]).strftime("%Y-%m-%d"): row
        for row in load_alpha_rows(path)
    }


def _rank_corr(left, right):
    left_codes = list(left.get("codes", []))
    right_codes = list(right.get("codes", []))
    common = set(left_codes) & set(right_codes)
    if len(common) < 3:
        return np.nan
    left_rank = {code: rank for rank, code in enumerate(left_codes)}
    right_rank = {code: rank for rank, code in enumerate(right_codes)}
    x = np.asarray([left_rank[code] for code in common], dtype=float)
    y = np.asarray([right_rank[code] for code in common], dtype=float)
    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _top_overlap(left, right, top_n):
    top_left = set(list(left.get("codes", []))[:top_n])
    top_right = set(list(right.get("codes", []))[:top_n])
    if not top_left or not top_right:
        return np.nan
    return float(len(top_left & top_right) / min(len(top_left), len(top_right)))


def audit(lineage_path, blend_path, output_dir):
    lineage = load_lineage_manifest(lineage_path)
    components = {entry["component_id"]: entry for entry in lineage["components"]}
    if "compact" not in components or "v14_full" not in components:
        raise ValueError("historical audit requires compact and v14_full components")

    component_rows = {}
    for component_id in ("compact", "v14_full"):
        rows = {}
        for window in components[component_id]["windows"]:
            rows.update(_rows_by_date(window["alpha_path"]))
        component_rows[component_id] = rows
    blend_rows = _rows_by_date(blend_path)
    common_dates = sorted(set(component_rows["compact"]) & set(component_rows["v14_full"]) & set(blend_rows))
    if not common_dates:
        raise ValueError("no common dates among historical OOF inputs")

    daily = []
    for date in common_dates:
        compact = component_rows["compact"][date]
        v14 = component_rows["v14_full"][date]
        blend = blend_rows[date]
        daily.append(
            {
                "date": date,
                "year": int(date[:4]),
                "rank_corr_compact_v14": _rank_corr(compact, v14),
                "top30_overlap_compact_v14": _top_overlap(compact, v14, 30),
                "rank_corr_blend_compact": _rank_corr(blend, compact),
                "rank_corr_blend_v14": _rank_corr(blend, v14),
                "top30_overlap_blend_compact": _top_overlap(blend, compact, 30),
                "top30_overlap_blend_v14": _top_overlap(blend, v14, 30),
                "n_compact": len(compact.get("codes", [])),
                "n_v14": len(v14.get("codes", [])),
                "n_blend": len(blend.get("codes", [])),
            }
        )
    daily_frame = pd.DataFrame(daily)
    summary = (
        daily_frame.groupby("year", sort=True)
        .agg(
            signal_days=("date", "count"),
            rank_corr_compact_v14=("rank_corr_compact_v14", "mean"),
            top30_overlap_compact_v14=("top30_overlap_compact_v14", "mean"),
            rank_corr_blend_compact=("rank_corr_blend_compact", "mean"),
            rank_corr_blend_v14=("rank_corr_blend_v14", "mean"),
            top30_overlap_blend_compact=("top30_overlap_blend_compact", "mean"),
            top30_overlap_blend_v14=("top30_overlap_blend_v14", "mean"),
            avg_n_blend=("n_blend", "mean"),
        )
        .reset_index()
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    daily_frame.to_csv(output / "daily_oof_rank_diagnostics.csv", index=False)
    summary.to_csv(output / "yearly_oof_rank_diagnostics.csv", index=False)
    metadata = {
        "lineage_manifest": str(Path(lineage_path).resolve()),
        "blend_path": str(Path(blend_path).resolve()),
        "research_end": lineage["research_end"],
        "signal_start": common_dates[0],
        "signal_end": common_dates[-1],
        "dates": len(common_dates),
        "years": sorted(int(value) for value in summary["year"].tolist()),
    }
    (output / "oof_diagnostics_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    audit(args.lineage_manifest, args.blend, args.output_dir)


if __name__ == "__main__":
    main()
