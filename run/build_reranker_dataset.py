"""Build a candidate-level dataset for the M0 second-stage ranker."""

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backtest.engine import detect_regime
from core.research_protocol import assert_research_end_date
from run.backtest_v9_retention import load_v9_samples_and_predictor


HORIZON_INDICES = (0, 2, 4, 6)
HORIZON_WEIGHTS = np.asarray((0.15, 0.25, 0.35, 0.25), dtype=np.float64)
RELEVANCE_CUTOFFS = ((0.06, 4), (0.15, 3), (0.30, 2), (0.60, 1))
NON_FEATURE_COLS = {
    "split",
    "date",
    "code",
    "group_size",
    "candidate_position",
    "future_target",
    "future_h5_raw",
    "future_rank_pct",
    "relevance",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints_loss_ablation_M0_nomulti/epochs/epoch_006.pt",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2025-01-01")
    parser.add_argument("--candidate-frac", type=float, default=0.10)
    parser.add_argument("--predictor-mode", default="average")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-dates", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        default="reranker_data_20260614/m0_validation_2024",
    )
    parser.add_argument(
        "--alpha-output",
        default=None,
        help="Optional JSONL path for the full-universe M0 Alpha generated in the same pass",
    )
    return parser.parse_args()


def percentile_rank_desc(values):
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return ranks.astype(np.float64) / max(len(values) - 1, 1)


def relevance_from_rank(rank_pct):
    relevance = np.zeros(len(rank_pct), dtype=np.int8)
    for cutoff, grade in RELEVANCE_CUTOFFS:
        relevance[(rank_pct < cutoff) & (relevance == 0)] = grade
    return relevance


def alpha_history_features(code, alpha, rank_pct, alpha_history, rank_history):
    alpha_values = alpha_history[code]
    rank_values = rank_history[code]

    def lag(values, offset):
        return float(values[-offset]) if len(values) >= offset else np.nan

    alpha_lag1 = lag(alpha_values, 1)
    alpha_lag3 = lag(alpha_values, 3)
    rank_lag1 = lag(rank_values, 1)
    rank_lag3 = lag(rank_values, 3)
    alpha_ma3 = float(np.mean(alpha_values)) if alpha_values else np.nan
    return {
        "m0_alpha": float(alpha),
        "m0_rank_pct": float(rank_pct),
        "m0_alpha_change_1d": float(alpha - alpha_lag1)
        if np.isfinite(alpha_lag1)
        else np.nan,
        "m0_alpha_change_3d": float(alpha - alpha_lag3)
        if np.isfinite(alpha_lag3)
        else np.nan,
        "m0_rank_change_1d": float(rank_pct - rank_lag1)
        if np.isfinite(rank_lag1)
        else np.nan,
        "m0_rank_change_3d": float(rank_pct - rank_lag3)
        if np.isfinite(rank_lag3)
        else np.nan,
        "m0_alpha_ma3": alpha_ma3,
        "m0_alpha_vs_ma3": float(alpha - alpha_ma3)
        if np.isfinite(alpha_ma3)
        else np.nan,
    }


def generation_args(args):
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        split="val",
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        predictor_mode=args.predictor_mode,
        window=args.window,
        target_fracs="0.006",
        hold_fracs="0.10",
        market_timing_mode="legacy",
        market_min_mult=0.20,
        market_max_mult=1.00,
        weight_mode="equal",
        max_weight=0.05,
        commission_rate=0.0001,
        stamp_tax_rate=0.0005,
        slippage_rate=0.0005,
        index_file="hs300_index.csv",
        device=args.device,
        progress_every=args.progress_every,
        limit_dates=args.limit_dates,
        ablate_fundamental=False,
    )


def build_rows(samples, predictor, args):
    rows = []
    audits = []
    alpha_rows = []
    alpha_history = defaultdict(lambda: deque(maxlen=3))
    rank_history = defaultdict(lambda: deque(maxlen=3))

    for sample_idx, sample in enumerate(samples, start=1):
        date = pd.Timestamp(sample["date"])
        codes = np.asarray(sample["codes"], dtype=object)
        y_seq = np.asarray(sample["y_seq"], dtype=np.float64)
        raw_h5 = np.asarray(sample["raw_y"], dtype=np.float64)
        valid_label = np.isfinite(y_seq[:, HORIZON_INDICES]).all(axis=1)
        if valid_label.sum() < 100:
            continue

        valid_predict = np.ones(len(codes), dtype=bool)
        alpha = np.asarray(
            predictor.predict_alpha(sample, valid_predict, detect_regime(sample)),
            dtype=np.float64,
        )
        if len(alpha) != len(codes):
            raise RuntimeError(
                f"{date.date()} Alpha length {len(alpha)} != universe {len(codes)}"
            )

        valid = valid_label & np.isfinite(alpha)
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) < 100:
            continue

        weighted_target = (
            y_seq[valid_idx][:, HORIZON_INDICES] * HORIZON_WEIGHTS
        ).sum(axis=1)
        target_rank = percentile_rank_desc(weighted_target)
        alpha_valid = alpha[valid_idx]
        alpha_rank = percentile_rank_desc(alpha_valid)
        if args.alpha_output:
            alpha_order = np.argsort(-alpha_valid, kind="mergesort")
            ordered_codes = [str(codes[valid_idx[index]]) for index in alpha_order]
            ordered_alpha = alpha_valid[alpha_order].astype(float).tolist()
            alpha_rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "codes": ordered_codes,
                    "alpha": ordered_alpha,
                    "n_stocks": len(ordered_codes),
                }
            )

        candidate_count = max(30, int(np.ceil(len(valid_idx) * args.candidate_frac)))
        candidate_local = np.argsort(-alpha_valid, kind="mergesort")[:candidate_count]
        candidate_target_rank = percentile_rank_desc(weighted_target[candidate_local])
        candidate_relevance = relevance_from_rank(candidate_target_rank)

        top006 = target_rank < 0.006
        top02 = target_rank < 0.02
        candidate_mask = np.zeros(len(valid_idx), dtype=bool)
        candidate_mask[candidate_local] = True
        audits.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "universe_size": int(len(valid_idx)),
                "candidate_size": int(candidate_count),
                "top006_count": int(top006.sum()),
                "top02_count": int(top02.sum()),
                "top006_recall": float((candidate_mask & top006).sum() / max(top006.sum(), 1)),
                "top02_recall": float((candidate_mask & top02).sum() / max(top02.sum(), 1)),
                "candidate_relevance_mean": float(candidate_relevance.mean()),
            }
        )

        regime = detect_regime(sample)
        regime_code = {"panic": -1, "sideways": 0, "trend_up": 1}.get(regime, 0)
        for position, local_idx in enumerate(candidate_local):
            global_idx = int(valid_idx[local_idx])
            code = str(codes[global_idx])
            row = {
                "split": args.split,
                "date": date.strftime("%Y-%m-%d"),
                "code": code,
                "group_size": int(candidate_count),
                "candidate_position": int(position),
                "industry_id": int(sample["industry_ids"][global_idx]),
                "market_regime": int(regime_code),
                "future_target": float(weighted_target[local_idx]),
                "future_h5_raw": float(raw_h5[global_idx])
                if np.isfinite(raw_h5[global_idx])
                else np.nan,
                "future_rank_pct": float(target_rank[local_idx]),
                "relevance": int(candidate_relevance[position]),
            }
            row.update(
                alpha_history_features(
                    code,
                    alpha_valid[local_idx],
                    alpha_rank[local_idx],
                    alpha_history,
                    rank_history,
                )
            )
            for feature_idx, value in enumerate(sample["X"][global_idx]):
                row[f"x_{feature_idx:03d}"] = float(value)
            for risk_idx, value in enumerate(sample["risk"][global_idx]):
                row[f"risk_{risk_idx:03d}"] = float(value)
            rows.append(row)

        for local_idx, global_idx in enumerate(valid_idx):
            code = str(codes[int(global_idx)])
            alpha_history[code].append(float(alpha_valid[local_idx]))
            rank_history[code].append(float(alpha_rank[local_idx]))

        if args.progress_every and sample_idx % args.progress_every == 0:
            print(
                f"dataset {sample_idx}/{len(samples)} dates | rows={len(rows)}",
                flush=True,
            )

    if args.alpha_output:
        alpha_path = ROOT / args.alpha_output
        alpha_path.parent.mkdir(parents=True, exist_ok=True)
        with alpha_path.open("w", encoding="utf-8") as handle:
            for row in alpha_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return pd.DataFrame(rows), pd.DataFrame(audits)


def validate_dataset(frame):
    if frame.empty:
        raise ValueError("Reranker dataset is empty")
    duplicate_count = int(frame.duplicated(["date", "code"]).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate date/code rows")
    group_mismatch = (
        frame.groupby("date").size().astype(int)
        != frame.groupby("date")["group_size"].first().astype(int)
    )
    if group_mismatch.any():
        raise ValueError("Stored group_size does not match daily row count")
    feature_cols = [c for c in frame.columns if c not in NON_FEATURE_COLS]
    forbidden = [
        c
        for c in feature_cols
        if c.startswith("future_") or c == "relevance"
    ]
    if forbidden:
        raise ValueError(f"Future columns entered feature set: {forbidden}")
    return feature_cols


def main():
    args = parse_args()
    if not 0 < args.candidate_frac <= 1:
        raise ValueError("candidate-frac must be in (0, 1]")
    end_date = assert_research_end_date(
        args.end_date, context="reranker dataset"
    )
    if end_date <= pd.Timestamp(args.start_date):
        raise ValueError("end-date must be later than start-date")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _, samples, predictor = load_v9_samples_and_predictor(generation_args(args))
    frame, audit = build_rows(samples, predictor, args)
    feature_cols = validate_dataset(frame)

    dataset_path = out_dir / "reranker_dataset.parquet"
    audit_path = out_dir / "daily_audit.csv"
    frame.to_parquet(dataset_path, index=False)
    audit.to_csv(audit_path, index=False)

    label_distribution = {
        str(int(key)): int(value)
        for key, value in frame["relevance"].value_counts().sort_index().items()
    }
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "start_date": str(frame["date"].min()),
        "end_date": str(frame["date"].max()),
        "dates": int(frame["date"].nunique()),
        "rows": int(len(frame)),
        "feature_count": int(len(feature_cols)),
        "candidate_frac": float(args.candidate_frac),
        "label_distribution": label_distribution,
        "mean_top006_recall": float(audit["top006_recall"].mean()),
        "median_top006_recall": float(audit["top006_recall"].median()),
        "mean_top02_recall": float(audit["top02_recall"].mean()),
        "median_top02_recall": float(audit["top02_recall"].median()),
        "parquet_bytes": int(dataset_path.stat().st_size),
        "feature_columns": feature_cols,
        "non_feature_columns": sorted(NON_FEATURE_COLS),
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dataset_config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "feature_columns"}, indent=2))
    print(f"Saved dataset to {dataset_path}")


if __name__ == "__main__":
    main()
