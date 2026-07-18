"""Apply the Torch V5 reranker to a candidate dataset and coarse Alpha."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "PyTorch is required. Use C:\\Users\\x\\miniconda3\\envs\\torch\\python.exe"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run.train_reranker_v5_torch import ListwiseReranker, apply_scaler
from run.validate_reranker_2024 import write_alpha


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--coarse-alpha", required=True)
    parser.add_argument("--model-dir", default="reranker_models_20260702/torch_listwise_v5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-frac", type=float, default=0.006)
    parser.add_argument("--hold-frac", type=float, default=0.10)
    parser.add_argument("--candidate-end", type=int, default=80)
    parser.add_argument("--max-reranked-fills", type=int, default=3)
    parser.add_argument("--gate", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_alpha_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row["date"] = pd.Timestamp(row["date"])
                row["codes"] = [str(code) for code in row["codes"]]
                rows.append(row)
    return rows


def load_model(model_dir, device):
    model_dir = Path(model_dir)
    with (model_dir / "preprocess.pkl").open("rb") as handle:
        preprocess = pickle.load(handle)
    features = preprocess["feature_columns"]
    hidden_dim = int(preprocess["args"].get("hidden_dim", 128))
    dropout = float(preprocess["args"].get("dropout", 0.15))
    model = ListwiseReranker(len(features), hidden_dim=hidden_dim, dropout=dropout)
    state = torch.load(model_dir / "model_best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    summary_path = model_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return model, preprocess, summary


@torch.no_grad()
def score_dataset(dataset, model, preprocess, device):
    features = preprocess["feature_columns"]
    missing = [column for column in features if column not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing model features: {missing[:10]}")
    scored_parts = []
    median = preprocess["median"]
    scale = preprocess["scale"]
    for date, group in dataset.groupby("date", sort=False):
        group = group.sort_values("candidate_position", kind="mergesort").copy()
        x_np = apply_scaler(group, features, median, scale)
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)
        mask = torch.ones(1, len(group), dtype=torch.bool, device=device)
        score, beat_logit = model(x, mask)
        group["v5_score"] = score.squeeze(0).detach().cpu().numpy()
        group["v5_win"] = torch.sigmoid(beat_logit.squeeze(0)).detach().cpu().numpy()
        scored_parts.append(group)
    return pd.concat(scored_parts, ignore_index=True)


def build_rows(scored, coarse_rows, args, gate):
    by_date = {
        pd.Timestamp(date): group.set_index("code", drop=False)
        for date, group in scored.groupby("date", sort=False)
    }
    current_selected = []
    holding_ages = {}
    output_rows = []
    audits = []
    for row in coarse_rows:
        date = pd.Timestamp(row["date"])
        original = [str(code) for code in row["codes"]]
        n = len(original)
        if date not in by_date:
            raise ValueError(f"No V5 candidates for {date.date()}")
        candidates_for_date = by_date[date]
        target_n = max(1, int(n * args.target_frac))
        hold_n = max(target_n, int(n * args.hold_frac))
        rank_map = {code: index for index, code in enumerate(original)}
        kept = [
            code for code in current_selected
            if rank_map.get(code, n + 1) < hold_n
        ]
        if len(kept) > target_n:
            kept = sorted(kept, key=lambda code: rank_map[code])[:target_n]
        vacancies = max(target_n - len(kept), 0)
        fill_candidates = [code for code in original if code not in set(kept)]
        baseline_fills = fill_candidates[:vacancies]
        protected_count = max(vacancies - int(args.max_reranked_fills), 0)
        protected = baseline_fills[:protected_count]
        baseline_rerank = baseline_fills[protected_count:]
        slots = min(vacancies, int(args.max_reranked_fills))
        eligible = [
            code
            for code in fill_candidates[protected_count:]
            if rank_map.get(code, n + 1) < int(args.candidate_end)
            and code in candidates_for_date.index
        ]
        active = False
        confidence = np.nan
        selected_rerank = baseline_rerank
        proposed = baseline_rerank
        if slots > 0 and len(eligible) >= slots:
            candidate_frame = candidates_for_date.loc[eligible].copy()
            proposed = (
                candidate_frame.sort_values(
                    ["v5_score", "m0_rank_pct", "candidate_position"],
                    ascending=[False, True, True],
                    kind="mergesort",
                )["code"]
                .astype(str)
                .head(slots)
                .tolist()
            )
            baseline_win = candidate_frame.loc[
                [code for code in baseline_rerank if code in candidate_frame.index],
                "v5_win",
            ]
            if len(baseline_win) == slots:
                confidence = float(
                    candidate_frame.set_index("code").loc[proposed, "v5_win"].mean()
                    - baseline_win.mean()
                )
                active = True if gate is None else confidence >= float(gate)
            else:
                active = gate is None
        if active:
            selected_rerank = proposed
        selected_fills = protected + selected_rerank
        selected = kept + selected_fills
        selected_fill_set = set(selected_fills)
        if active:
            priority = selected_fills + [
                code for code in original if code not in selected_fill_set
            ]
        else:
            priority = original
        alpha = (1.0 - np.arange(n, dtype=np.float64) / max(n - 1, 1)).tolist()
        output_rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "codes": priority,
                "alpha": alpha,
                "n_stocks": n,
                "reranker_v5": {
                    "active": bool(active),
                    "confidence": None if not np.isfinite(confidence) else confidence,
                    "gate": gate,
                    "slots": int(slots),
                },
            }
        )
        audits.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "vacancies": int(vacancies),
                "slots": int(slots),
                "active": bool(active),
                "confidence": confidence,
                "changed_fills": int(len(set(selected_fills) - set(baseline_fills))),
            }
        )
        live_set = set(selected)
        for code in list(holding_ages):
            if code not in live_set:
                holding_ages.pop(code, None)
        for code in selected:
            holding_ages[code] = holding_ages.get(code, 0) + 1
        current_selected = selected
    return output_rows, pd.DataFrame(audits)


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model, preprocess, summary = load_model(ROOT / args.model_dir, device)
    gate = args.gate
    if gate is None:
        gate_info = summary.get("gate_calibration_2023") or {}
        gate = gate_info.get("gate")
    dataset = pd.read_parquet(ROOT / args.dataset)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["code"] = dataset["code"].astype(str)
    dataset = dataset[dataset["v3_eligible"].eq(1)].copy()
    scored = score_dataset(dataset, model, preprocess, device)
    rows, audit = build_rows(scored, load_alpha_rows(ROOT / args.coarse_alpha), args, gate)

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_alpha(output / "alpha_v5_gated.jsonl", rows)
    scored.to_parquet(output / "candidate_scores.parquet", index=False)
    audit.to_csv(output / "v5_apply_audit.csv", index=False)
    apply_summary = {
        "rows": len(rows),
        "gate": gate,
        "active_share": float(audit["active"].mean()) if len(audit) else 0.0,
        "mean_changed_fills": float(audit["changed_fills"].mean()) if len(audit) else 0.0,
        "mean_confidence": float(audit["confidence"].mean()) if len(audit) else 0.0,
        "dataset": args.dataset,
        "coarse_alpha": args.coarse_alpha,
        "model_dir": args.model_dir,
    }
    (output / "v5_apply_summary.json").write_text(
        json.dumps(apply_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(apply_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
