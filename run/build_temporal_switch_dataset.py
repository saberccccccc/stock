"""Build switch-value samples from temporal alpha states.

Rows represent replacing current holding A with candidate B. Features only use
current/past alpha state and industry information. The label is future net h5
edge after an explicit round-trip switch cost proxy.
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.temporal_train_utils import TemporalMemmapDataset
from run.backtest_temporal_daily_top import valid_indices_for_item
from run.eval_temporal_full import (
    apply_checkpoint_horizons,
    build_cfg,
    build_model,
    forward_temporal_chunked,
    load_meta,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build temporal switch-value dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--output-dir", default="temporal_switch_value_data_20260531")
    parser.add_argument("--target-frac", type=float, default=0.20)
    parser.add_argument("--hold-frac", type=float, default=0.90)
    parser.add_argument("--candidate-frac", type=float, default=0.20)
    parser.add_argument("--horizon-idx", type=int, default=4, help="y_seq index, 4 means h5")
    parser.add_argument("--max-holdings-per-day", type=int, default=80)
    parser.add_argument("--max-candidates-per-holding", type=int, default=3)
    parser.add_argument("--temporal-chunk", type=int, default=512)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--commission-rate", type=float, default=0.0001)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--alpha-window", type=int, default=3)
    parser.add_argument("--limit-train-dates", type=int, default=None)
    parser.add_argument("--limit-val-dates", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260531)
    return parser.parse_args()


def rank_pct_desc(alpha):
    order = np.argsort(alpha)[::-1]
    out = np.empty(len(alpha), dtype=np.float64)
    out[order] = np.arange(len(alpha), dtype=np.float64)
    return out / max(len(alpha) - 1, 1)


def explicit_switch_cost(args):
    return 2.0 * float(args.commission_rate) + 2.0 * float(args.slippage_rate) + float(args.stamp_tax_rate)


def alpha_state_features(code, i, alpha, rank, alpha_hist, rank_hist, alpha_ma, prefix):
    prev_a = alpha_hist.get(code, {}).get("last", np.nan)
    prev3_a = alpha_hist.get(code, {}).get("lag3", np.nan)
    prev_r = rank_hist.get(code, {}).get("last", np.nan)
    prev3_r = rank_hist.get(code, {}).get("lag3", np.nan)
    ma = float(np.mean(alpha_ma[code])) if alpha_ma[code] else np.nan
    return {
        f"{prefix}_alpha": float(alpha[i]),
        f"{prefix}_rank_pct": float(rank[i]),
        f"{prefix}_alpha_change_1d": float(alpha[i] - prev_a) if np.isfinite(prev_a) else np.nan,
        f"{prefix}_alpha_change_3d": float(alpha[i] - prev3_a) if np.isfinite(prev3_a) else np.nan,
        f"{prefix}_rank_change_1d": float(rank[i] - prev_r) if np.isfinite(prev_r) else np.nan,
        f"{prefix}_rank_change_3d": float(rank[i] - prev3_r) if np.isfinite(prev3_r) else np.nan,
        f"{prefix}_alpha_ma3": ma,
        f"{prefix}_alpha_vs_ma3": float(alpha[i] - ma) if np.isfinite(ma) else np.nan,
    }


@torch.no_grad()
def predict_item_alpha(args, model, cfg, item, device):
    from core.train_utils import get_regime_dim

    n = int(item["X"].shape[0])
    X = item["X"].unsqueeze(0).to(device)
    X_seq = item["X_seq"].unsqueeze(0).to(device)
    risk = item["risk"].unsqueeze(0).to(device)
    industry_ids = item["industry_ids"].unsqueeze(0).to(device)
    mask = torch.ones(1, n, dtype=torch.bool, device=device)
    regime_dim = get_regime_dim(cfg)
    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
        alpha_raw, _ = forward_temporal_chunked(
            model,
            X,
            X_seq,
            risk[..., :regime_dim],
            mask,
            industry_ids,
            args.temporal_chunk,
        )
    return alpha_raw[0].float().cpu().numpy()


def choose_a_codes(current_selected, rank_map, max_holdings):
    ranked = [(rank_map.get(code, np.inf), code) for code in current_selected]
    ranked = [(r, c) for r, c in ranked if np.isfinite(r)]
    ranked.sort(reverse=True)
    return [c for _, c in ranked[:max_holdings]]


def build_split_rows(args, split, ds, model, cfg, device):
    rows = []
    current_selected = []
    holding_ages = {}
    alpha_hist = {}
    rank_hist = {}
    alpha_ma = defaultdict(lambda: deque(maxlen=args.alpha_window))
    switch_cost = explicit_switch_cost(args)

    limit = args.limit_train_dates if split == "train" else args.limit_val_dates
    n_dates = len(ds) if limit is None else min(len(ds), int(limit))
    for day_i in range(n_dates):
        if args.progress_every and day_i % args.progress_every == 0:
            print(f"[{split}] {day_i}/{n_dates} rows={len(rows)} holdings={len(current_selected)}", flush=True)
        item = ds[day_i]
        n = int(item["X"].shape[0])
        if n < 50 or args.horizon_idx >= item["y_seq"].shape[-1]:
            continue

        valid_idx = valid_indices_for_item(ds, item)
        codes = ds.all_codes[valid_idx].tolist()
        if len(codes) != n:
            raise RuntimeError(f"code count mismatch at {split} {day_i}: {len(codes)} vs {n}")
        alpha = predict_item_alpha(args, model, cfg, item, device)
        finite = np.isfinite(alpha)
        if np.count_nonzero(finite) < 50:
            continue
        alpha = alpha[finite]
        y_h = item["y_seq"].numpy()[finite, args.horizon_idx].astype(np.float64)
        industry = item["industry_ids"].numpy()[finite]
        codes = [codes[i] for i in np.flatnonzero(finite)]
        rank = rank_pct_desc(alpha)
        order = np.argsort(alpha)[::-1]
        code_to_i = {code: i for i, code in enumerate(codes)}
        rank_map = {code: int(np.where(order == i)[0][0]) for code, i in code_to_i.items()}

        holding_set = {c for c in current_selected if c in code_to_i}
        candidate_n = max(1, int(len(codes) * args.candidate_frac))
        candidate_codes = [codes[int(i)] for i in order[:candidate_n] if codes[int(i)] not in holding_set]
        a_codes = choose_a_codes(current_selected, rank_map, args.max_holdings_per_day)

        for a_code in a_codes:
            a_i = code_to_i.get(a_code)
            if a_i is None or not np.isfinite(y_h[a_i]):
                continue
            selected_b = []
            if candidate_codes:
                selected_b.append(candidate_codes[0])
            better = [c for c in candidate_codes if alpha[code_to_i[c]] > alpha[a_i]]
            if better:
                selected_b.append(min(better, key=lambda c: alpha[code_to_i[c]] - alpha[a_i]))
            same = [c for c in candidate_codes if int(industry[code_to_i[c]]) == int(industry[a_i])]
            if same:
                selected_b.append(same[0])
            for c in candidate_codes:
                if len(selected_b) >= args.max_candidates_per_holding:
                    break
                selected_b.append(c)

            seen = set()
            for b_code in selected_b:
                if b_code in seen:
                    continue
                seen.add(b_code)
                b_i = code_to_i.get(b_code)
                if b_i is None or not np.isfinite(y_h[b_i]):
                    continue
                row = {
                    "split": split,
                    "date": str(ds.all_dates[int(item["time_index"].item())]),
                    "A_code": a_code,
                    "B_code": b_code,
                    "same_industry": int(industry[a_i] == industry[b_i]),
                    "A_holding_days": int(holding_ages.get(a_code, 0)),
                    "switch_explicit_cost": switch_cost,
                    "switch_impact_cost": 0.0,
                    "switch_execution_risk_cost": 0.0,
                    "switch_full_cost": switch_cost,
                    "A_ret_fwd_h5": float(y_h[a_i]),
                    "B_ret_fwd_h5": float(y_h[b_i]),
                }
                row.update(alpha_state_features(a_code, a_i, alpha, rank, alpha_hist, rank_hist, alpha_ma, "A"))
                row.update(alpha_state_features(b_code, b_i, alpha, rank, alpha_hist, rank_hist, alpha_ma, "B"))
                row["alpha_diff"] = row["B_alpha"] - row["A_alpha"]
                row["rank_advantage"] = row["A_rank_pct"] - row["B_rank_pct"]
                row["switch_edge_raw_h5"] = row["B_ret_fwd_h5"] - row["A_ret_fwd_h5"]
                row["switch_edge_net_h5"] = row["switch_edge_raw_h5"] - switch_cost
                row["switch_success_h5"] = int(row["switch_edge_net_h5"] > 0.0)
                rows.append(row)

        target_n = max(1, int(len(codes) * args.target_frac))
        hold_n = max(target_n, int(len(codes) * args.hold_frac))
        kept = [code for code in current_selected if rank_map.get(code, n + 1) < hold_n]
        if len(kept) > target_n:
            kept = sorted(kept, key=lambda c: rank_map.get(c, n + 1))[:target_n]
        selected = list(kept)
        selected_set = set(selected)
        for idx in order:
            code = codes[int(idx)]
            if len(selected) >= target_n:
                break
            if code not in selected_set:
                selected.append(code)
                selected_set.add(code)
        for code in list(holding_ages):
            if code not in selected_set:
                holding_ages.pop(code, None)
        for code in selected:
            holding_ages[code] = holding_ages.get(code, 0) + 1
        current_selected = selected

        for i, code in enumerate(codes):
            hist_a = alpha_hist.setdefault(code, {})
            hist_r = rank_hist.setdefault(code, {})
            hist_a["lag3"] = hist_a.get("lag2", np.nan)
            hist_a["lag2"] = hist_a.get("last", np.nan)
            hist_a["last"] = float(alpha[i])
            hist_r["lag3"] = hist_r.get("lag2", np.nan)
            hist_r["lag2"] = hist_r.get("last", np.nan)
            hist_r["last"] = float(rank[i])
            alpha_ma[code].append(float(alpha[i]))

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta_path, meta = load_meta(args, checkpoint)
    cfg = apply_checkpoint_horizons(build_cfg(meta), checkpoint)
    model = build_model(meta, cfg, checkpoint, device)

    frames = []
    for split in ("train", "val"):
        ds = TemporalMemmapDataset(meta, split=split)
        frames.append(build_split_rows(args, split, ds, model, cfg, device))
    df = pd.concat(frames, axis=0, ignore_index=True)
    parquet_path = out_dir / "temporal_switch_value_dataset.parquet"
    csv_path = out_dir / "temporal_switch_value_dataset.csv"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        print(f"Failed to write parquet: {exc}", flush=True)
    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby("split")
        .agg(
            rows=("switch_edge_net_h5", "size"),
            edge_mean=("switch_edge_net_h5", "mean"),
            raw_edge_mean=("switch_edge_raw_h5", "mean"),
            success_rate=("switch_success_h5", "mean"),
            cost_mean=("switch_full_cost", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "temporal_switch_value_dataset_summary.csv", index=False)
    config = vars(args).copy()
    config["meta_path"] = meta_path
    (out_dir / "temporal_switch_value_dataset_config.json").write_text(
        pd.Series(config).to_json(indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Saved dataset to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()

