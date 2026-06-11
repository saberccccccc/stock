"""Full-stock validation for TemporalCrossAlphaModel.

This evaluates every valid stock in each validation cross-section. To keep the
result equivalent to a full forward pass, only the per-stock temporal encoder is
chunked; the cross-stock Transformer still sees the whole cross-section.
"""

import argparse
import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.config import DataConfig
from core.temporal_model import TemporalCrossAlphaModel
from core.temporal_train_utils import TemporalMemmapDataset
from core.train_utils import get_regime_dim, weighted_horizon_target
from data.market_features import N_MARKET


def parse_args():
    parser = argparse.ArgumentParser(description="Full-stock temporal validation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "trainval", "all"])
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--temporal-chunk", type=int, default=512)
    parser.add_argument("--limit-dates", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def load_meta(args, checkpoint):
    meta_path = args.meta or checkpoint.get("temporal_meta_path")
    if not meta_path:
        raise ValueError("Provide --meta or use a checkpoint with temporal_meta_path")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta_path, meta


def build_cfg(meta):
    cfg = DataConfig()
    cfg.horizon_weights = (0.10, 0.30, 0.30, 0.30)
    cfg.max_horizon = int(meta.get("max_horizon", cfg.max_horizon))
    cfg.target_horizon = int(meta.get("target_horizon", cfg.target_horizon))
    cfg.min_stocks_per_time = int(meta.get("min_stocks", cfg.min_stocks_per_time))
    cfg.use_macro_features = int(meta.get("risk_full_dim", 0)) > (6 + N_MARKET)
    return cfg


def apply_checkpoint_horizons(cfg, checkpoint):
    arch = checkpoint.get("arch_config", {})
    horizon_indices = checkpoint.get("horizon_indices", arch.get("horizon_indices"))
    horizon_weights = checkpoint.get("horizon_weights", arch.get("horizon_weights"))
    if horizon_indices:
        cfg.horizon_indices = tuple(int(x) for x in horizon_indices)
    if horizon_weights:
        cfg.horizon_weights = tuple(float(x) for x in horizon_weights)
    return cfg


def build_model(meta, cfg, checkpoint, device):
    arch = checkpoint.get("arch_config", {})
    agg_groups = arch.get("agg_groups")
    if not agg_groups:
        agg_groups = [(int(meta["high_feat_dim"]), 5, 0.1)]
        low_base_dim = int(meta.get("low_agg_dim", 0)) // 2
        if low_base_dim > 0:
            agg_groups.append((low_base_dim, 2, 0.0))
    agg_groups = [tuple(g) for g in agg_groups]

    model = TemporalCrossAlphaModel(
        input_dim=int(meta["x_dim"]),
        seq_dim=int(meta["seq_dim"]),
        agg_groups=agg_groups,
        hidden_dim=int(arch.get("hidden_dim", 256)),
        temporal_dim=int(arch.get("temporal_dim", 128)),
        n_horizons=len(cfg.horizon_indices),
        n_alpha=4,
        regime_dim=get_regime_dim(cfg),
        num_industries=int(meta["n_industries"]),
        dropout=getattr(cfg, "transformer_dropout", 0.35),
        temporal_residual_adapters=bool(arch.get("temporal_residual_adapters", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _rank_embed(model, X, mask):
    return model._build_rank_embed(X, mask)


@torch.no_grad()
def forward_temporal_chunked(model, X, X_seq, risk_cont, mask, industry_ids, temporal_chunk):
    """Equivalent to model.forward, but chunks only the independent TCN path."""
    cross_h = model.feature_grouper(X) + model.input_proj(X)

    temporal_parts = []
    n_stocks = X_seq.shape[1]
    for start in range(0, n_stocks, temporal_chunk):
        end = min(start + temporal_chunk, n_stocks)
        temporal_parts.append(model.temporal_encoder(X_seq[:, start:end]))
    temporal_h = torch.cat(temporal_parts, dim=1)

    h = model.temporal_fusion(cross_h, temporal_h)
    if getattr(model, "temporal_residual_adapters", False):
        h = model.temporal_adapter_pre(h, temporal_h)
    industry_ids_valid = torch.where(industry_ids >= 0, industry_ids, model.num_industries)
    industry_ids_valid = torch.clamp(industry_ids_valid, 0, model.num_industries)
    industry_emb = model.industry_embed(industry_ids_valid)
    h = h + model.industry_proj(torch.cat([h, industry_emb], dim=-1))
    h = h + _rank_embed(model, X, mask)
    h = h + model.trans_input_proj(X)

    mask_f = mask.float().unsqueeze(-1)
    regime_sum = (risk_cont * mask_f).sum(dim=1)
    regime_count = mask_f.sum(dim=1).clamp(min=1.0)
    regime_h = model.regime_proj(regime_sum / regime_count).unsqueeze(1)

    t1 = model.trans1(h, src_key_padding_mask=~mask) + regime_h
    if getattr(model, "temporal_residual_adapters", False):
        t1 = model.temporal_adapter_mid(t1, temporal_h)
    h_out = model.trans2(t1, src_key_padding_mask=~mask) + regime_h
    if getattr(model, "temporal_residual_adapters", False):
        h_out = model.temporal_adapter_out(h_out, temporal_h)

    alphas = torch.cat([head(h_out) for head in model.alpha_heads], dim=-1)
    gate = model.alpha_gate(h_out)
    alpha_raw = (alphas * gate).sum(dim=-1)
    horizon_preds = torch.cat([head(h_out) for head in model.horizon_heads], dim=-1)
    return alpha_raw, horizon_preds


def add_metric(results, key, value):
    if np.isfinite(value):
        results.setdefault(key, []).append(float(value))


def evaluate(args, model, ds, cfg, device):
    h_indices = list(cfg.horizon_indices)
    top_fracs = tuple(getattr(cfg, "eval_top_fracs", (0.05, 0.10)))
    min_eval_stocks = max(10, int(getattr(cfg, "min_stocks_per_time", 30)))
    regime_dim = get_regime_dim(cfg)
    results = {"alpha": []}
    daily_rows = []

    n_dates = len(ds) if args.limit_dates is None else min(len(ds), args.limit_dates)
    t0 = time.time()
    for i in range(n_dates):
        item = ds[i]
        n = int(item["X"].shape[0])
        if n < min_eval_stocks:
            continue
        X = item["X"].unsqueeze(0).to(device)
        X_seq = item["X_seq"].unsqueeze(0).to(device)
        risk = item["risk"].unsqueeze(0).to(device)
        industry_ids = item["industry_ids"].unsqueeze(0).to(device)
        y_seq = item["y_seq"].unsqueeze(0).to(device)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            alpha_raw, horizon_preds = forward_temporal_chunked(
                model,
                X,
                X_seq,
                risk[..., :regime_dim],
                mask,
                industry_ids,
                args.temporal_chunk,
            )
            target_weighted, _, _ = weighted_horizon_target(y_seq, cfg)

        pred = alpha_raw[0].float().cpu().numpy()
        target = target_weighted[0].float().cpu().numpy()
        y_np = y_seq[0].float().cpu().numpy()
        hp_np = horizon_preds[0].float().cpu().numpy()

        row = {
            "date": str(ds.all_dates[int(item["time_index"].item())]),
            "n_stocks": n,
        }
        ic = np.corrcoef(pred, target)[0, 1] if pred.size > 1 else np.nan
        add_metric(results, "alpha", ic)
        row["alpha"] = float(ic) if np.isfinite(ic) else None

        order = np.argsort(pred)
        k10 = max(5, int(n * 0.10))
        top10 = order[-k10:]
        bot10 = order[:k10]
        for j, h_idx in enumerate(h_indices):
            if h_idx >= y_np.shape[-1]:
                continue
            h_tag = f"h{h_idx + 1}"
            ret_h = y_np[:, h_idx]
            if j < hp_np.shape[-1]:
                ic_h = np.corrcoef(hp_np[:, j], ret_h)[0, 1] if n > 1 else np.nan
                add_metric(results, h_tag, ic_h)
                row[h_tag] = float(ic_h) if np.isfinite(ic_h) else None
            spread = ret_h[top10].mean() - ret_h[bot10].mean()
            add_metric(results, f"topbot_{h_tag}", spread)
            row[f"topbot_{h_tag}"] = float(spread) if np.isfinite(spread) else None
            for frac in top_fracs:
                tag = str(int(round(float(frac) * 100)))
                k = max(5, int(n * frac))
                top_idx = order[-k:]
                top_ret = ret_h[top_idx].mean()
                add_metric(results, f"topret_{h_tag}_top{tag}", top_ret)
                row[f"topret_{h_tag}_top{tag}"] = float(top_ret) if np.isfinite(top_ret) else None
                top_pred = pred[top_idx]
                top_y = ret_h[top_idx]
                top_ic = np.nan
                if k > 10 and np.std(top_pred) > 1e-8 and np.std(top_y) > 1e-8:
                    top_ic = np.corrcoef(top_pred, top_y)[0, 1]
                add_metric(results, f"topic_{h_tag}_top{tag}", top_ic)
                row[f"topic_{h_tag}_top{tag}"] = float(top_ic) if np.isfinite(top_ic) else None

        daily_rows.append(row)
        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            print(
                f"full eval {i + 1}/{n_dates} | n={n} | "
                f"alpha_mean={np.mean(results['alpha']):.4f} | "
                f"time={(time.time() - t0) / 60:.1f}m",
                flush=True,
            )

    summary = {key: (float(np.mean(vals)) if vals else 0.0) for key, vals in results.items()}
    counts = {key: len(vals) for key, vals in results.items()}
    return summary, counts, daily_rows


def write_report(args, checkpoint, meta_path, summary, counts, daily_rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem + f"_{args.split}_full_eval"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    daily_path = out_dir / f"{stem}_daily.jsonl"

    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "sampled_val_metrics_in_checkpoint": checkpoint.get("val_metrics", {}),
        "meta_path": str(meta_path),
        "split": args.split,
        "temporal_chunk": args.temporal_chunk,
        "amp": bool(args.amp),
        "summary": summary,
        "counts": counts,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with daily_path.open("w", encoding="utf-8") as f:
        for row in daily_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ordered = [
        "alpha",
        "h1", "h3", "h5", "h7",
        "topret_h5_top5", "topret_h5_top10",
        "topic_h5_top5", "topic_h5_top10",
        "topbot_h5",
        "topret_h7_top5", "topret_h7_top10",
        "topic_h7_top5", "topic_h7_top10",
        "topbot_h7",
    ]
    lines = [
        "# Temporal full-stock validation",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- checkpoint epoch: `{checkpoint.get('epoch')}`",
        f"- split: `{args.split}`",
        f"- temporal_chunk: `{args.temporal_chunk}`",
        f"- meta: `{meta_path}`",
        "",
        "## Summary",
        "",
        "| metric | value | count |",
        "|---|---:|---:|",
    ]
    seen = set()
    for key in ordered + sorted(summary):
        if key in seen or key not in summary:
            continue
        seen.add(key)
        lines.append(f"| {key} | {summary[key]:.6f} | {counts.get(key, 0)} |")
    lines.extend([
        "",
        "## Checkpoint Sampled Validation",
        "",
        "These are the metrics saved during capped validation in training, for comparison.",
        "",
        "```json",
        json.dumps(checkpoint.get("val_metrics", {}), ensure_ascii=False, indent=2),
        "```",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path, daily_path


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta_path, meta = load_meta(args, checkpoint)
    cfg = build_cfg(meta)
    cfg = apply_checkpoint_horizons(cfg, checkpoint)
    ds = TemporalMemmapDataset(meta, split=args.split)
    model = build_model(meta, cfg, checkpoint, device)

    print(
        f"Evaluating {args.split}: dates={len(ds)}, x_dim={meta['x_dim']}, "
        f"seq=({meta['seq_lookback']},{meta['seq_dim']}), device={device}, "
        f"amp={args.amp}",
        flush=True,
    )
    summary, counts, daily_rows = evaluate(args, model, ds, cfg, device)
    json_path, md_path, daily_path = write_report(
        args,
        checkpoint,
        meta_path,
        summary,
        counts,
        daily_rows,
        Path(args.output_dir),
    )
    print(f"Saved JSON: {json_path}", flush=True)
    print(f"Saved report: {md_path}", flush=True)
    print(f"Saved daily: {daily_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
