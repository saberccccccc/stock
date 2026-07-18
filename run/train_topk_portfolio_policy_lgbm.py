"""Train a lightweight selector over daily top-k portfolio proposals."""

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


EXCLUDE = {
    "split",
    "date",
    "proposal",
    "label_available",
    "portfolio_utility",
    "baseline_utility",
    "utility_delta_vs_baseline",
    "selected_codes",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--forward-dataset", default=None)
    parser.add_argument("--train-name", default="train_2024")
    parser.add_argument("--test-name", default="test_2025")
    parser.add_argument("--forward-name", default="forward_2026")
    parser.add_argument(
        "--selection-name",
        action="append",
        default=None,
        help="Split name allowed for model/rule selection. Repeat for multiple splits.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-col", default="utility_delta_vs_baseline")
    parser.add_argument("--objective", choices=("regression", "lambdarank"), default="regression")
    parser.add_argument("--num-boost-round", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=5)
    parser.add_argument("--min-data-in-leaf", type=int, default=40)
    parser.add_argument("--lambda-l2", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args(argv)


def load_dataset(path):
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


def feature_columns(frame):
    cols = []
    for col in frame.columns:
        if col in EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def build_matrix(frame, cols, fill_values=None):
    x = frame[cols].copy()
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if fill_values is None:
        fill_values = {}
        for col in cols:
            med = x[col].median()
            fill_values[col] = float(med) if np.isfinite(med) else 0.0
    x = x.fillna(fill_values)
    return x.to_numpy(dtype=np.float32), fill_values


def relevance_labels(frame, target_col, bins=5):
    labels = pd.Series(0, index=frame.index, dtype=np.int32)
    target = pd.to_numeric(frame[target_col], errors="coerce")
    for _, idx in frame.groupby("date", sort=False).groups.items():
        values = target.loc[idx]
        valid = values.notna()
        if int(valid.sum()) <= 1:
            continue
        ranks = values.loc[valid].rank(method="first", pct=True)
        labels.loc[ranks.index] = np.floor(ranks * bins).clip(0, bins - 1).astype(np.int32)
    return labels.to_numpy(dtype=np.int32)


def group_sizes_by_date(frame):
    return frame.groupby("date", sort=False).size().astype(int).tolist()


def train_model(train_frame, cols, args):
    train_frame = train_frame.sort_values("date").reset_index(drop=True)
    x, fill_values = build_matrix(train_frame, cols)
    if args.objective == "lambdarank":
        y = relevance_labels(train_frame, args.target_col)
        group = group_sizes_by_date(train_frame)
        metric = "ndcg"
    else:
        y = pd.to_numeric(train_frame[args.target_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        group = None
        metric = "l2"
    ds = lgb.Dataset(x, label=y, group=group, feature_name=cols, free_raw_data=False)
    params = {
        "objective": args.objective,
        "metric": metric,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l2": args.lambda_l2,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": args.seed,
        "force_col_wise": True,
        "num_threads": 0,
    }
    if args.objective == "lambdarank":
        params["label_gain"] = [0, 1, 2, 3, 4]
    model = lgb.train(params, ds, num_boost_round=args.num_boost_round)
    return model, fill_values, params


def score(frame, model, cols, fill_values):
    out = frame.copy()
    x, _ = build_matrix(out, cols, fill_values)
    out["policy_score"] = model.predict(x)
    return out


def evaluate(frame, model, cols, fill_values, split, target_col):
    scored = score(frame, model, cols, fill_values)
    rows = []
    for date, group in scored.groupby("date", sort=True):
        chosen = group.sort_values(["policy_score", "proposal"], ascending=[False, True]).head(1).iloc[0]
        oracle = group.sort_values([target_col, "proposal"], ascending=[False, True]).head(1).iloc[0]
        baseline = group[group["proposal"].eq("baseline")]
        baseline_delta = float(baseline[target_col].iloc[0]) if not baseline.empty else 0.0
        rows.append(
            {
                "split": split,
                "date": date,
                "chosen_proposal": chosen["proposal"],
                "chosen_delta": float(chosen[target_col]),
                "baseline_delta": baseline_delta,
                "oracle_proposal": oracle["proposal"],
                "oracle_delta": float(oracle[target_col]),
                "hit_oracle": int(chosen["proposal"] == oracle["proposal"]),
            }
        )
    daily = pd.DataFrame(rows)
    edge = daily["chosen_delta"]
    std = float(edge.std(ddof=1)) if len(edge) > 1 else np.nan
    t_stat = float(edge.mean() / (std / np.sqrt(len(edge)))) if np.isfinite(std) and std > 1e-12 else np.nan
    return daily, {
        "split": split,
        "days": int(len(daily)),
        "mean_chosen_delta": float(daily["chosen_delta"].mean()),
        "positive_rate": float((daily["chosen_delta"] > 0).mean()),
        "t_stat": t_stat,
        "oracle_hit_rate": float(daily["hit_oracle"].mean()),
        "mean_oracle_delta": float(daily["oracle_delta"].mean()),
        "proposal_counts": daily["chosen_proposal"].value_counts().to_dict(),
    }


def write_markdown(summaries, output_path, selection_splits):
    selection_text = ", ".join(selection_splits)
    observed = [row["split"] for row in summaries if row["split"] not in set(selection_splits)]
    observation_text = ", ".join(observed) if observed else "none"
    lines = [
        "# Top-k Portfolio Policy Selector",
        "",
        f"Selection splits: {selection_text}. Observation-only splits: {observation_text}.",
        "",
        "| split | days | mean delta | positive rate | t-stat | oracle hit | proposals |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['split']} | {row['days']} | {row['mean_chosen_delta']:.6f} | "
            f"{row['positive_rate']:.2%} | {row['t_stat']:.3f} | "
            f"{row['oracle_hit_rate']:.2%} | {row['proposal_counts']} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_frame = load_dataset(args.train_dataset)
    test_frame = load_dataset(args.test_dataset)
    forward_frame = load_dataset(args.forward_dataset) if args.forward_dataset else None
    selection_names = args.selection_name or [args.train_name, args.test_name]
    cols = feature_columns(train_frame)
    model, fill_values, params = train_model(train_frame, cols, args)
    summaries = []
    for split, frame in [(args.train_name, train_frame), (args.test_name, test_frame), (args.forward_name, forward_frame)]:
        if frame is None:
            continue
        daily, summary = evaluate(frame, model, cols, fill_values, split, args.target_col)
        daily.to_csv(output / f"{split}_daily.csv", index=False)
        summaries.append(summary)
    importance = pd.DataFrame(
        {
            "feature": cols,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    with (output / "model.pkl").open("wb") as fh:
        pickle.dump(
            {
                "model": model,
                "feature_columns": cols,
                "fill_values": fill_values,
                "params": params,
                "target": args.target_col,
                "objective": args.objective,
            },
            fh,
        )
    meta = {
        "train_dataset": str(args.train_dataset),
        "test_dataset": str(args.test_dataset),
        "forward_dataset": str(args.forward_dataset) if args.forward_dataset else None,
        "target": args.target_col,
        "objective": args.objective,
        "train_name": args.train_name,
        "test_name": args.test_name,
        "forward_name": args.forward_name if args.forward_dataset else None,
        "feature_columns": cols,
        "params": params,
        "summaries": summaries,
        "selection_names": selection_names,
        "selection_protocol": f"Use {', '.join(selection_names)} for selector development checks. "
        "Splits not listed there are observation-only.",
    }
    (output / "training_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summaries, output / "topk_portfolio_policy_summary.md", selection_names)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
