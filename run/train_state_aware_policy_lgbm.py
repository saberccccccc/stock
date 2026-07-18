"""Train a lightweight state-aware portfolio policy with LightGBM.

The policy is a marginal-fill reranker: for each signal day it scores only the
replaceable candidate slots prepared by ``build_state_aware_policy_dataset``.
Model selection is based on 2024/2025 evidence.  Forward rows are reported only
as observation and must not be used for parameter selection.
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


TARGET_COLUMNS = {
    "exec_target_raw",
    "exec_target",
    "edge_vs_baseline",
    "exec_base_return",
    "exec_delayed_return",
    "exec_return_1d",
    "exec_return_3d",
    "exec_return_5d",
    "exec_return_10d",
}
EXCLUDE_FEATURE_COLUMNS = {
    "split",
    "date",
    "code",
    "label_available",
    "beats_baseline",
    "baseline_target_raw",
    "edge_vs_baseline",
    "exec_target",
    "exec_target_raw",
    "exec_base_return",
    "exec_delayed_return",
    "exec_max_downside",
    "exec_signal_to_entry",
    "exec_blocked_buy",
    "exec_return_1d",
    "exec_return_3d",
    "exec_return_5d",
    "exec_return_10d",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--forward-dataset", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_COLUMNS),
        default="edge_vs_baseline",
        help="Training target. edge_vs_baseline is the default marginal-fill target.",
    )
    parser.add_argument(
        "--objective",
        choices=("regression", "lambdarank"),
        default="regression",
        help="Use regression on the raw target or listwise ranking within each signal day.",
    )
    parser.add_argument("--num-boost-round", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--min-data-in-leaf", type=int, default=120)
    parser.add_argument("--lambda-l2", type=float, default=5.0)
    parser.add_argument("--feature-fraction", type=float, default=0.90)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument(
        "--max-selected-per-day",
        type=int,
        default=None,
        help="Optional cap for reranked slots. Defaults to each day's rerank_slots.",
    )
    return parser.parse_args(argv)


def load_dataset(path):
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"empty dataset: {path}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


def infer_feature_columns(frame):
    features = []
    for column in frame.columns:
        if column in EXCLUDE_FEATURE_COLUMNS or column.startswith("exec_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            features.append(column)
    if not features:
        raise ValueError("no numeric feature columns found")
    return features


def eligible_labelled(frame, target):
    mask = frame["eligible"].eq(1) & frame["label_available"].eq(1) & frame[target].notna()
    out = frame.loc[mask].copy()
    if out.empty:
        raise ValueError(f"no eligible labelled rows for target={target}")
    return out


def build_matrix(frame, feature_columns, fill_values=None):
    x = frame[feature_columns].copy()
    for column in feature_columns:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    if fill_values is None:
        fill_values = {}
        for column in feature_columns:
            median = x[column].median()
            fill_values[column] = float(median) if np.isfinite(median) else 0.0
    x = x.fillna(fill_values)
    return x.to_numpy(dtype=np.float32), fill_values


def make_train_label(train_frame, target, objective):
    y = pd.to_numeric(train_frame[target], errors="coerce")
    if objective == "lambdarank":
        ranks = y.groupby(train_frame["date"]).rank(pct=True, method="first")
        return np.floor(ranks.fillna(0.0).clip(0.0, 1.0).to_numpy() * 31.0).astype(np.int32)
    return y.to_numpy(dtype=np.float32)


def train_model(train_frame, feature_columns, target, args):
    train_x, fill_values = build_matrix(train_frame, feature_columns)
    train_y = make_train_label(train_frame, target, args.objective)
    group = None
    if args.objective == "lambdarank":
        ordered = train_frame.sort_values(["date", "candidate_position"]).copy()
        train_x, fill_values = build_matrix(ordered, feature_columns)
        train_y = make_train_label(ordered, target, args.objective)
        group = ordered.groupby("date", sort=False).size().to_numpy(dtype=np.int32)
    dataset = lgb.Dataset(
        train_x,
        label=train_y,
        group=group,
        feature_name=feature_columns,
        free_raw_data=False,
    )
    params = {
        "objective": args.objective,
        "metric": "ndcg" if args.objective == "lambdarank" else "l2",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l2": args.lambda_l2,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "verbosity": -1,
        "seed": args.seed,
        "force_col_wise": True,
        "num_threads": 0,
    }
    if args.objective == "lambdarank":
        params["label_gain"] = list(range(32))
    model = lgb.train(params, dataset, num_boost_round=args.num_boost_round)
    return model, fill_values, params


def add_scores(frame, model, feature_columns, fill_values):
    scored = frame.copy()
    x, _ = build_matrix(scored, feature_columns, fill_values)
    scored["policy_score"] = model.predict(x)
    return scored


def _safe_mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def evaluate_policy(frame, model, feature_columns, fill_values, split_name, max_selected_per_day=None):
    labelled = eligible_labelled(frame, "exec_target_raw")
    scored = add_scores(labelled, model, feature_columns, fill_values)
    daily = []
    selected_rows = []
    for date, group in scored.groupby("date", sort=True):
        slots = int(group["rerank_slots"].max())
        if max_selected_per_day is not None:
            slots = min(slots, int(max_selected_per_day))
        slots = max(slots, 0)
        if slots == 0:
            continue
        choose_n = min(slots, len(group))
        chosen = group.sort_values(["policy_score", "candidate_position"], ascending=[False, True]).head(choose_n)
        baseline = group[group["baseline_fill"].eq(1) & group["protected_fill"].eq(0)].sort_values("candidate_position").head(choose_n)
        if baseline.empty:
            baseline = group.sort_values("candidate_position").head(choose_n)
        chosen_target = _safe_mean(chosen["exec_target_raw"])
        baseline_target = _safe_mean(baseline["exec_target_raw"])
        chosen_base = _safe_mean(chosen["exec_base_return"])
        baseline_base = _safe_mean(baseline["exec_base_return"])
        row = {
            "split": split_name,
            "date": date,
            "slots": choose_n,
            "chosen_target_raw": chosen_target,
            "baseline_slot_target_raw": baseline_target,
            "edge_raw": chosen_target - baseline_target if np.isfinite(chosen_target) and np.isfinite(baseline_target) else np.nan,
            "chosen_base_return": chosen_base,
            "baseline_slot_base_return": baseline_base,
            "edge_base_return": chosen_base - baseline_base if np.isfinite(chosen_base) and np.isfinite(baseline_base) else np.nan,
            "chosen_blocked_buy": _safe_mean(chosen["exec_blocked_buy"]),
            "baseline_blocked_buy": _safe_mean(baseline["exec_blocked_buy"]),
            "chosen_rank_pct": _safe_mean(chosen["candidate_rank_pct"]),
            "baseline_rank_pct": _safe_mean(baseline["candidate_rank_pct"]),
            "chosen_pressure": _safe_mean(chosen.get("global_us_hk_pressure", pd.Series(dtype=float))),
            "chosen_beta": _safe_mean(chosen.get("beta_60d", pd.Series(dtype=float))),
            "chosen_vol": _safe_mean(chosen.get("specific_vol_60d", pd.Series(dtype=float))),
            "chosen_ret20": _safe_mean(chosen.get("ret_20d", pd.Series(dtype=float))),
        }
        daily.append(row)
        selected_rows.append(chosen.assign(selection_date=date))
    daily_frame = pd.DataFrame(daily)
    selected_frame = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    if daily_frame.empty:
        summary = {
            "split": split_name,
            "days": 0,
            "mean_edge_raw": np.nan,
            "mean_edge_base_return": np.nan,
            "edge_win_rate": np.nan,
            "edge_t_stat": np.nan,
        }
        return daily_frame, selected_frame, summary
    edge = pd.to_numeric(daily_frame["edge_raw"], errors="coerce").dropna()
    base_edge = pd.to_numeric(daily_frame["edge_base_return"], errors="coerce").dropna()
    edge_std = float(edge.std(ddof=1)) if len(edge) > 1 else np.nan
    edge_t = float(edge.mean() / (edge_std / math.sqrt(len(edge)))) if np.isfinite(edge_std) and edge_std > 1e-12 else np.nan
    summary = {
        "split": split_name,
        "days": int(len(daily_frame)),
        "mean_slots": float(daily_frame["slots"].mean()),
        "mean_edge_raw": float(edge.mean()) if len(edge) else np.nan,
        "median_edge_raw": float(edge.median()) if len(edge) else np.nan,
        "mean_edge_base_return": float(base_edge.mean()) if len(base_edge) else np.nan,
        "edge_win_rate": float((edge > 0).mean()) if len(edge) else np.nan,
        "edge_t_stat": edge_t,
        "chosen_target_raw": float(daily_frame["chosen_target_raw"].mean()),
        "baseline_slot_target_raw": float(daily_frame["baseline_slot_target_raw"].mean()),
        "chosen_base_return": float(daily_frame["chosen_base_return"].mean()),
        "baseline_slot_base_return": float(daily_frame["baseline_slot_base_return"].mean()),
        "chosen_blocked_buy": float(daily_frame["chosen_blocked_buy"].mean()),
        "baseline_blocked_buy": float(daily_frame["baseline_blocked_buy"].mean()),
        "chosen_rank_pct": float(daily_frame["chosen_rank_pct"].mean()),
        "baseline_rank_pct": float(daily_frame["baseline_rank_pct"].mean()),
    }
    return daily_frame, selected_frame, summary


def feature_importance_frame(model, feature_columns):
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    return (
        pd.DataFrame({"feature": feature_columns, "gain": gain, "split": split})
        .sort_values(["gain", "split"], ascending=False)
        .reset_index(drop=True)
    )


def write_report(output, args, summaries, importance):
    lines = [
        "# 状态感知精排 LightGBM 训练报告",
        "",
        "## 协议",
        "",
        "- 训练：2024 val 数据集。",
        "- 选择/判断：2025 test 离线边际收益；2024 只作为训练内参考。",
        "- 2026 forward：只观察，不参与调参。",
        "- 目标不是 Alpha IC，而是可替换名额在 open-ledger 标签下相对原始填仓的边际改善。",
        "",
        "## 参数",
        "",
        f"- target: `{args.target}`",
        f"- objective: `{args.objective}`",
        f"- num_boost_round: `{args.num_boost_round}`",
        f"- learning_rate: `{args.learning_rate}`",
        f"- num_leaves: `{args.num_leaves}`",
        f"- min_data_in_leaf: `{args.min_data_in_leaf}`",
        f"- lambda_l2: `{args.lambda_l2}`",
        "",
        "## 结果摘要",
        "",
        "| split | days | mean edge raw | win rate | t-stat | edge base return | chosen blocked | baseline blocked |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {split} | {days} | {mean_edge_raw:.6f} | {edge_win_rate:.3f} | {edge_t_stat:.2f} | "
            "{mean_edge_base_return:.6f} | {chosen_blocked_buy:.4f} | {baseline_blocked_buy:.4f} |".format(
                **{
                    key: (0.0 if value is None or (isinstance(value, float) and not np.isfinite(value)) else value)
                    for key, value in item.items()
                }
            )
        )
    lines.extend(
        [
            "",
            "## 重要特征 Top 20",
            "",
            "| feature | gain | split |",
            "|---|---:|---:|",
        ]
    )
    for _, row in importance.head(20).iterrows():
        lines.append(f"| {row['feature']} | {row['gain']:.2f} | {int(row['split'])} |")
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "这一步只是离线精排验证：它证明模型是否能在候选替换名额中找到更好的股票。",
            "若 2025 test 的边际收益、胜率和交易约束不够好，就不应该写回 alpha 做正式 open-ledger。",
        ]
    )
    (output / "training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = load_dataset(args.train_dataset)
    test = load_dataset(args.test_dataset)
    forward = load_dataset(args.forward_dataset) if args.forward_dataset else None
    feature_columns = infer_feature_columns(train)
    train_rows = eligible_labelled(train, args.target)
    model, fill_values, params = train_model(train_rows, feature_columns, args.target, args)
    summaries = []
    daily_outputs = {}
    selected_outputs = {}
    for split_name, frame in [("train_2024", train), ("test_2025", test), ("forward_2026", forward)]:
        if frame is None:
            continue
        daily, selected, summary = evaluate_policy(
            frame,
            model,
            feature_columns,
            fill_values,
            split_name,
            max_selected_per_day=args.max_selected_per_day,
        )
        summaries.append(summary)
        daily_outputs[split_name] = daily
        selected_outputs[split_name] = selected
        daily.to_csv(output / f"{split_name}_daily_policy_eval.csv", index=False)
        if not selected.empty:
            selected.to_parquet(output / f"{split_name}_selected_rows.parquet", index=False)
    importance = feature_importance_frame(model, feature_columns)
    importance.to_csv(output / "feature_importance.csv", index=False)
    with (output / "model.pkl").open("wb") as fh:
        pickle.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "fill_values": fill_values,
                "params": params,
                "target": args.target,
            },
            fh,
        )
    metadata = {
        "train_dataset": str(args.train_dataset),
        "test_dataset": str(args.test_dataset),
        "forward_dataset": str(args.forward_dataset) if args.forward_dataset else None,
        "target": args.target,
        "objective": args.objective,
        "feature_columns": feature_columns,
        "params": params,
        "summaries": summaries,
        "selection_protocol": "Select only with 2024 train reference and 2025 test; 2026 is observational.",
    }
    (output / "training_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output, args, summaries, importance)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
