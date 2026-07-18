"""Export top-k portfolio proposals into realistic open-ledger alpha files.

This is the bridge from a proposal-level portfolio-policy dataset to a
realistic open-price share-ledger training/evaluation dataset.  It preserves
the original alpha universe per day and only moves each proposal's selected
codes to the front of the ranking, so target/hold fractions remain comparable
across proposals.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)
os.chdir(ROOT)

from alpha.io import load_alpha_rows, write_alpha_rows


DEFAULT_STRESSES = ("normal", "lag1", "cost2x", "capacity_3pct")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topk-dataset", required=True)
    parser.add_argument("--source-alpha-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-data-date", required=True)
    parser.add_argument("--proposal", action="append", default=None)
    parser.add_argument("--portfolio-values", default="500000,1000000")
    parser.add_argument("--target-fracs", default="0.006")
    parser.add_argument("--hold-fracs", default="0.10")
    parser.add_argument("--preset", default="official_open_price_share_ledger")
    parser.add_argument("--execution-constraint-mode", default="realistic")
    parser.add_argument("--stresses", default=",".join(DEFAULT_STRESSES))
    parser.add_argument("--global-risk-features", default=None)
    return parser.parse_args(argv)


def normalize_date(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def parse_selected_codes(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [code for code in str(value).split(";") if code]


def load_topk_rows(path, split_name, proposals=None, start_date=None, end_date=None):
    frame = pd.read_parquet(path)
    if "date" not in frame.columns or "proposal" not in frame.columns or "selected_codes" not in frame.columns:
        raise ValueError("top-k dataset must contain date, proposal and selected_codes columns")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame[frame["split"].astype(str).eq(str(split_name))]
    if proposals:
        frame = frame[frame["proposal"].astype(str).isin(set(proposals))]
    if start_date:
        frame = frame[frame["date"].ge(pd.Timestamp(start_date).normalize())]
    if end_date:
        frame = frame[frame["date"].le(pd.Timestamp(end_date).normalize())]
    if frame.empty:
        raise ValueError("no top-k proposal rows remain after filters")
    duplicates = frame.duplicated(["date", "proposal"], keep=False)
    if duplicates.any():
        dup = frame.loc[duplicates, ["date", "proposal"]].head(5).to_dict("records")
        raise ValueError(f"duplicate date/proposal rows: {dup}")
    return frame.sort_values(["proposal", "date"]).reset_index(drop=True)


def source_rows_by_date(path):
    rows = load_alpha_rows(path, timestamp_dates=False)
    return {normalize_date(row["date"]): row for row in rows}


def reorder_universe(source_row, selected_codes):
    source_codes = [str(code) for code in source_row.get("codes", [])]
    source_set = set(source_codes)
    front = []
    seen = set()
    for code in selected_codes:
        code = str(code)
        if code in source_set and code not in seen:
            front.append(code)
            seen.add(code)
    reordered = front + [code for code in source_codes if code not in seen]
    n = len(reordered)
    # Keep alpha aligned with the exported ranking.  The open-ledger equal
    # weight path uses code order, while score-weighted tests can use alpha.
    alpha = np.linspace(1.0, 0.0, n, dtype=float).tolist() if n else []
    return reordered, alpha, len(front)


def build_proposal_alpha_rows(topk_frame, source_by_date, proposal):
    out = []
    proposal_frame = topk_frame[topk_frame["proposal"].astype(str).eq(str(proposal))]
    for row in proposal_frame.itertuples(index=False):
        date = normalize_date(getattr(row, "date"))
        if date not in source_by_date:
            raise ValueError(f"source alpha is missing date {date}")
        selected = parse_selected_codes(getattr(row, "selected_codes"))
        codes, alpha, selected_in_source = reorder_universe(source_by_date[date], selected)
        out.append(
            {
                "date": date,
                "codes": codes,
                "alpha": alpha,
                "n_stocks": int(len(codes)),
                "proposal": str(proposal),
                "selected_count": int(len(selected)),
                "selected_in_source": int(selected_in_source),
            }
        )
    return out


def backtest_command(alpha_path, out_dir, args, stress):
    cmd = [
        "python",
        "run/backtest_retention_open_ledger.py",
        "--alpha-jsonl",
        str(alpha_path),
        "--output-dir",
        str(out_dir),
        "--preset",
        args.preset,
        "--stress",
        stress,
        "--portfolio-values",
        args.portfolio_values,
        "--target-fracs",
        args.target_fracs,
        "--hold-fracs",
        args.hold_fracs,
        "--execution-constraint-mode",
        args.execution_constraint_mode,
        "--max-data-date",
        args.max_data_date,
    ]
    if args.start_date:
        cmd.extend(["--start-date", args.start_date])
    if args.end_date:
        cmd.extend(["--end-date", args.end_date])
    if args.global_risk_features:
        cmd.extend(["--global-risk-features", args.global_risk_features])
    return cmd


def write_command_manifest(output, alpha_files, args):
    stresses = [item.strip() for item in str(args.stresses).split(",") if item.strip()]
    commands = []
    for proposal, alpha_path in alpha_files.items():
        for stress in stresses:
            out_dir = output / "open_ledger" / proposal / stress
            commands.append(
                {
                    "proposal": proposal,
                    "stress": stress,
                    "alpha_jsonl": str(alpha_path),
                    "output_dir": str(out_dir),
                    "command": backtest_command(alpha_path, out_dir, args, stress),
                }
            )
    (output / "backtest_commands.json").write_text(
        json.dumps(commands, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output / "backtest_commands.ps1").open("w", encoding="utf-8") as handle:
        for item in commands:
            quoted = []
            for part in item["command"]:
                text = str(part)
                quoted.append(f'"{text}"' if any(ch.isspace() for ch in text) else text)
            handle.write(" ".join(quoted) + "\n")
    return commands


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    alpha_dir = output / "proposal_alpha"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    topk = load_topk_rows(
        args.topk_dataset,
        args.split_name,
        proposals=args.proposal,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    source = source_rows_by_date(args.source_alpha_jsonl)
    proposals = sorted(topk["proposal"].astype(str).unique().tolist())
    alpha_files = {}
    proposal_summaries = []
    for proposal in proposals:
        rows = build_proposal_alpha_rows(topk, source, proposal)
        path = alpha_dir / f"{proposal}.jsonl"
        write_alpha_rows(path, rows)
        alpha_files[proposal] = path
        proposal_summaries.append(
            {
                "proposal": proposal,
                "rows": int(len(rows)),
                "signal_start": rows[0]["date"] if rows else "",
                "signal_end": rows[-1]["date"] if rows else "",
                "mean_selected_count": float(np.mean([r["selected_count"] for r in rows])) if rows else np.nan,
                "mean_selected_in_source": float(np.mean([r["selected_in_source"] for r in rows])) if rows else np.nan,
                "alpha_jsonl": str(path),
            }
        )
    commands = write_command_manifest(output, alpha_files, args)
    meta = {
        "topk_dataset": str(args.topk_dataset),
        "source_alpha_jsonl": str(args.source_alpha_jsonl),
        "split_name": args.split_name,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "max_data_date": args.max_data_date,
        "selection_protocol": "Use 2024 validation and 2025 test only for selection. 2026 forward is observation-only.",
        "proposal_summaries": proposal_summaries,
        "command_count": int(len(commands)),
        "params": vars(args),
    }
    (output / "realistic_topk_ledger_dataset_v2_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
