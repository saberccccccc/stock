"""Train a light open-execution reranker and apply it to current V9 Alpha rows.

The model is trained on historical OOF reranker feature rows with a label that
matches the current execution direction more closely: next-open to +5-open
return.  It is then applied to the current V9 avgw3 Alpha files by rebuilding
feature rows from cached V9 samples.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backtest.runtime import build_v9_backtest_config
from alpha.io import load_alpha_rows as load_shared_alpha_rows
from alpha.io import write_alpha_rows
from core.research_protocol import assert_alpha_rows_within_research
from data.pipeline import build_cross_section_dataset, samples_from_precomputed_metadata


OOF_TRAINING_END = pd.Timestamp("2023-12-31")


def parse_args():
    parser = argparse.ArgumentParser(description="Train/apply current V9 open reranker")
    parser.add_argument("--output-dir", default="open_reranker_current_v9_20260617")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--train-glob", default="reranker_data_20260614/oof_F*/reranker_dataset.parquet")
    parser.add_argument("--val-alpha", required=True)
    parser.add_argument("--test-alpha", required=True)
    parser.add_argument("--candidate-top-n", type=int, default=300)
    parser.add_argument("--candidate-start-rank", type=int, default=0, help="Only rerank candidates from this 0-based rank")
    parser.add_argument("--alpha-weights", default="0.85,0.90,0.95")
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=450000)
    parser.add_argument("--existing-model", default=None, help="Reuse a saved open_reranker_lgb.pkl and only rescore alpha files")
    return parser.parse_args()


def load_alpha_rows(path):
    return load_shared_alpha_rows(path, timestamp_dates=True)


def write_alpha(path, rows):
    normalized = []
    for row in rows:
        out = dict(row)
        out["date"] = pd.Timestamp(out["date"]).strftime("%Y-%m-%d")
        normalized.append(out)
    return write_alpha_rows(path, normalized)


def feature_columns(frame):
    x_cols = [c for c in frame.columns if c.startswith("x_")]
    risk_cols = [c for c in frame.columns if c.startswith("risk_")]
    base = ["candidate_position", "m0_rank_pct", "m0_alpha"]
    return base + x_cols + risk_cols


def load_training_frame(pattern, max_rows):
    paths = sorted(Path(".").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No training parquet files match {pattern}")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame = frame[frame["candidate_position"] < 300].copy()
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    if data["date"].max() > OOF_TRAINING_END:
        raise ValueError(
            f"Open-reranker training rows exceed OOF cutoff {OOF_TRAINING_END.date()}"
        )
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=17).sort_values(["date", "candidate_position"])
    return data.reset_index(drop=True)


def add_open_label(frame, data_dir, horizon):
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("label horizon must be at least one trading day")
    labels = np.full(len(frame), np.nan, dtype=np.float32)
    by_code_indices = frame.groupby("code", sort=False).indices
    for code, indices in by_code_indices.items():
        path = Path(data_dir) / f"{code}.csv"
        if not path.exists():
            continue
        try:
            prices = pd.read_csv(path, usecols=["trade_date", "open"])
        except Exception:
            continue
        prices["trade_date"] = pd.to_datetime(prices["trade_date"], errors="coerce")
        prices["open"] = pd.to_numeric(prices["open"], errors="coerce")
        prices = prices.dropna().sort_values("trade_date")
        dates = pd.DatetimeIndex(prices["trade_date"])
        opens = prices["open"].to_numpy(dtype=np.float64)
        row_dates = pd.DatetimeIndex(pd.to_datetime(frame.iloc[indices]["date"]))
        pos = dates.searchsorted(row_dates, side="right")
        end = pos + horizon
        in_bounds = (pos < len(opens)) & (end < len(opens))
        vals = np.full(len(indices), np.nan, dtype=np.float32)
        bounded = np.flatnonzero(in_bounds)
        if len(bounded):
            entry = opens[pos[bounded]]
            exit_price = opens[end[bounded]]
            label_end = dates[end[bounded]]
            year_end = pd.DatetimeIndex(
                [pd.Timestamp(year=int(date.year), month=12, day=31) for date in row_dates[bounded]]
            )
            usable = (
                np.isfinite(entry)
                & np.isfinite(exit_price)
                & (entry > 0)
                & (label_end <= year_end)
            )
            target_rows = bounded[usable]
            vals[target_rows] = (
                exit_price[usable] / entry[usable] - 1.0
            ).astype(np.float32)
        labels[np.asarray(indices)] = vals
    frame = frame.copy()
    frame["open_h5_label"] = labels
    return frame[np.isfinite(frame["open_h5_label"])].copy()


def train_model(train, features, output_dir):
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": 17,
        "num_threads": 8,
    }
    dtrain = lgb.Dataset(train[features], label=train["open_h5_label"])
    model = lgb.train(params, dtrain, num_boost_round=240)
    with (Path(output_dir) / "open_reranker_lgb.pkl").open("wb") as handle:
        pickle.dump({"model": model, "feature_columns": features}, handle)
    return model


def load_current_samples(data_dir=None):
    cfg = build_v9_backtest_config()
    if data_dir is not None:
        cfg.data_dir = data_dir
    meta = build_cross_section_dataset(cfg, use_cache=True)
    if not isinstance(meta, dict):
        samples = meta[0] + meta[1]
    else:
        cfg.low_feat_dim = meta.get("low_agg_dim", getattr(cfg, "low_feat_dim", 14))
        eval_meta = dict(meta)
        eval_meta["val_indices"] = list(meta["train_indices"]) + list(meta["val_indices"])
        samples = samples_from_precomputed_metadata(eval_meta, "val")
    by_date = {pd.Timestamp(sample["date"]): sample for sample in samples}
    return by_date


def score_alpha_rows(rows, samples_by_date, model, features, candidate_top_n, candidate_start_rank, alpha_weight):
    output = []
    scored_dates = 0
    for row in rows:
        date = pd.Timestamp(row["date"])
        sample = samples_by_date.get(date)
        codes = [str(code) for code in row["codes"]]
        alphas = np.asarray(row.get("alpha", []), dtype=np.float64)
        if sample is None:
            output.append(row)
            continue
        sample_codes = np.asarray(sample["codes"], dtype=object).astype(str)
        code_to_i = {code: i for i, code in enumerate(sample_codes)}
        n = len(codes)
        start_rank = max(0, int(candidate_start_rank))
        top_n = min(int(candidate_top_n), n)
        if start_rank >= top_n:
            output.append(row)
            continue
        candidate_codes = [code for code in codes[start_rank:top_n] if code in code_to_i]
        if len(candidate_codes) < 10:
            output.append(row)
            continue
        idx = np.asarray([code_to_i[code] for code in candidate_codes], dtype=np.int64)
        frame = pd.DataFrame({
            "candidate_position": [codes.index(code) for code in candidate_codes],
            "m0_rank_pct": [codes.index(code) / max(n - 1, 1) for code in candidate_codes],
            "m0_alpha": [float(alphas[codes.index(code)]) if len(alphas) == n else 1.0 - codes.index(code) / max(n - 1, 1) for code in candidate_codes],
        })
        x = np.asarray(sample["X"], dtype=np.float32)[idx]
        risk = np.asarray(sample["risk"], dtype=np.float32)[idx]
        for j in range(x.shape[1]):
            frame[f"x_{j:03d}"] = x[:, j]
        for j in range(risk.shape[1]):
            frame[f"risk_{j:03d}"] = risk[:, j]
        for col in features:
            if col not in frame:
                frame[col] = np.nan
        pred = model.predict(frame[features])
        pred_rank = pd.Series(pred).rank(method="first", ascending=False).to_numpy()
        pred_pct = 1.0 - (pred_rank - 1.0) / max(len(pred_rank) - 1, 1)
        alpha_pct = 1.0 - frame["candidate_position"].to_numpy(dtype=np.float64) / max(n - 1, 1)
        score = float(alpha_weight) * alpha_pct + (1.0 - float(alpha_weight)) * pred_pct
        order = np.argsort(score)[::-1]
        ranked_candidates = [candidate_codes[i] for i in order]
        candidate_set = set(candidate_codes)
        final_codes = codes[:start_rank] + ranked_candidates + [code for code in codes[start_rank:] if code not in candidate_set]
        final_alpha = (1.0 - np.arange(len(final_codes), dtype=np.float64) / max(len(final_codes) - 1, 1)).tolist()
        output.append({
            "date": date,
            "codes": final_codes,
            "alpha": final_alpha,
            "n_stocks": len(final_codes),
            "open_reranker": {
                "candidate_top_n": int(candidate_top_n),
                "candidate_start_rank": int(candidate_start_rank),
                "alpha_weight": float(alpha_weight),
                "scored_candidates": int(len(candidate_codes)),
            },
        })
        scored_dates += 1
    return output, scored_dates


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = None
    if args.existing_model:
        with Path(args.existing_model).open("rb") as handle:
            bundle = pickle.load(handle)
        model = bundle["model"]
        features = bundle["feature_columns"]
    else:
        train = load_training_frame(args.train_glob, args.max_train_rows)
        features = feature_columns(train)
        train = add_open_label(train, args.data_dir, args.label_horizon)
        train_rows = int(len(train))
        model = train_model(train, features, out_dir)
        train[["date", "code", "candidate_position", "open_h5_label"]].to_parquet(
            out_dir / "training_label_sample.parquet", index=False
        )

    samples_by_date = load_current_samples(args.data_dir)
    for split, alpha_path in [("val", args.val_alpha), ("test", args.test_alpha)]:
        rows = load_alpha_rows(alpha_path)
        assert_alpha_rows_within_research(
            rows, context=f"open-reranker {split} scoring"
        )
        for weight_raw in args.alpha_weights.split(","):
            weight = float(weight_raw.strip())
            scored, scored_dates = score_alpha_rows(
                rows, samples_by_date, model, features, args.candidate_top_n, args.candidate_start_rank, weight
            )
            rank_suffix = ""
            if int(args.candidate_start_rank) != 0 or int(args.candidate_top_n) != 300:
                rank_suffix = f"_r{int(args.candidate_start_rank):03d}_{int(args.candidate_top_n):03d}"
            out_path = out_dir / f"{split}_openrerank_w{int(round(weight * 100)):03d}{rank_suffix}.jsonl"
            write_alpha(out_path, scored)
            print(f"{split} weight={weight:.2f} scored_dates={scored_dates}/{len(rows)} -> {out_path}", flush=True)

    meta = {
        "train_rows": train_rows,
        "features": features,
        "candidate_top_n": int(args.candidate_top_n),
        "candidate_start_rank": int(args.candidate_start_rank),
        "alpha_weights": args.alpha_weights,
        "label": f"next_open_to_plus_{args.label_horizon}_open",
        "existing_model": args.existing_model,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
