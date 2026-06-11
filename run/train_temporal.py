"""Train TemporalCrossAlphaModel without touching legacy checkpoints."""

import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.config import DataConfig
from core.research_protocol import RESEARCH_END_DATE
from core.temporal_model import TemporalCrossAlphaModel
from core.temporal_train_utils import TemporalMemmapDataset, collate_temporal_eval, collate_temporal_train
from core.train_utils import get_regime_dim, total_loss_v7, weighted_horizon_target
from data.market_features import N_MARKET
from data.temporal_pipeline import build_temporal_cross_section_dataset


def _parse_args():
    parser = argparse.ArgumentParser(description="Train temporal cross-section alpha model")
    parser.add_argument(
        "--preset",
        default="none",
        choices=["none", "v9_adapter"],
        help="Conservative preset for V10 experiments; v9_adapter keeps temporal signal as a small residual.",
    )
    parser.add_argument("--meta", default=None, help="Existing temporal metadata .pkl")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--val-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--history-calendar-days", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--keep-ratio", type=float, default=0.20)
    parser.add_argument("--max-train-stocks", type=int, default=800)
    parser.add_argument("--max-eval-stocks", type=int, default=1200)
    parser.add_argument("--full-eval", action="store_true",
                        help="Validate on every valid stock instead of capping the validation cross-section")
    parser.add_argument("--eval-temporal-chunk", type=int, default=512,
                        help="Stock chunk size for the independent temporal encoder during validation")
    parser.add_argument("--eval-limit-dates", type=int, default=None,
                        help="Limit validation dates for fast screening; omit for full validation split")
    parser.add_argument("--train-limit-dates", type=int, default=None,
                        help="Limit train dates for smoke tests; omit for full training split")
    parser.add_argument("--eval-progress-every", type=int, default=0,
                        help="Print validation progress every N dates; 0 disables progress logging")
    parser.add_argument("--eval-before-train", action="store_true",
                        help="Run validation once before epoch 1; useful for V9 warm-start sanity checks.")
    parser.add_argument("--train-sample-mode", default="random",
                        choices=["random", "target_top_bottom_random"])
    parser.add_argument("--sample-top-frac", type=float, default=0.30)
    parser.add_argument("--sample-bottom-frac", type=float, default=0.30)
    parser.add_argument("--sample-random-frac", type=float, default=0.40)
    parser.add_argument("--sample-target-horizon", type=int, default=4,
                        help="y_seq index for target-aware sampling; 4 means h5")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-path", default="checkpoints_exp/temporal_cross_alpha_best.pt")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-tech", action="store_true")
    parser.add_argument("--no-market", action="store_true")
    parser.add_argument("--macro", action="store_true")
    parser.add_argument(
        "--best-metric",
        default="alpha",
        help=(
            "Metric key or composite expression, e.g. "
            "composite:alpha=0.5,h5=0.3,topbot_h5=0.2"
        ),
    )
    parser.add_argument("--horizon-mode", default="multi", choices=["multi", "h5", "h3h5", "h3h5h7"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--temporal-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--temporal-dropout", type=float, default=0.10)
    parser.add_argument("--temporal-fusion-cross-gate-init", type=float, default=0.0)
    parser.add_argument("--temporal-fusion-mode", default="blend_norm", choices=["blend_norm", "residual_add"])
    parser.add_argument("--transformer-norm-first", action="store_true", default=None)
    parser.add_argument("--init-v9-checkpoint", default=None, help="Warm-start matching cross-section weights from a V9 checkpoint.")
    parser.add_argument("--freeze-v9-epochs", type=int, default=0, help="Freeze matched V9-initialized parameters for the first N epochs.")
    parser.add_argument("--temporal-residual-adapters", action="store_true")
    parser.add_argument("--temporal-adapter-dropout", type=float, default=0.10)
    parser.add_argument("--temporal-adapter-gate-init", type=float, default=-3.0)
    parser.add_argument("--spread-loss-weight", type=float, default=None)
    parser.add_argument("--spread-delay-epochs", type=int, default=None)
    parser.add_argument("--top-focus-loss-weight", type=float, default=None)
    parser.add_argument("--top-focus-temperature", type=float, default=None)
    parser.add_argument("--top-focus-delay-epochs", type=int, default=None)
    parser.add_argument("--pairwise-top-loss-weight", type=float, default=None)
    parser.add_argument("--pairwise-top-frac", type=float, default=None)
    parser.add_argument("--pairwise-num-pairs", type=int, default=None)
    parser.add_argument("--pairwise-model-top-weight", type=float, default=None)
    parser.add_argument("--pairwise-delay-epochs", type=int, default=None)
    args = parser.parse_args()
    _apply_preset(args)
    return args


def _apply_preset(args):
    if args.preset != "v9_adapter":
        return
    # This preset is intentionally conservative: keep V9-like cross-section
    # ranking dominant at initialization, use random sampling, and select by
    # long-only validation metrics instead of fast-falling train loss.
    args.horizon_mode = "h3h5h7"
    args.train_sample_mode = "random"
    args.sample_top_frac = 0.10
    args.sample_bottom_frac = 0.10
    args.sample_random_frac = 0.80
    args.hidden_dim = 256 if args.init_v9_checkpoint else 128
    args.temporal_dim = 64
    args.lr = 1e-5
    args.weight_decay = 3e-3
    args.dropout = 0.20
    args.temporal_dropout = 0.15
    args.temporal_residual_adapters = True
    args.temporal_adapter_dropout = 0.15
    args.temporal_adapter_gate_init = -5.0
    args.temporal_fusion_mode = "residual_add"
    args.temporal_fusion_cross_gate_init = -6.0
    args.transformer_norm_first = False
    args.full_eval = True
    args.eval_temporal_chunk = min(int(args.eval_temporal_chunk), 512)
    args.best_metric = "composite:alpha=0.4,topret_h5_top5=0.3,topret_h5_top10=0.2,topic_h5_top10=0.1"
    if args.spread_loss_weight is None:
        args.spread_loss_weight = 0.0005
    if args.spread_delay_epochs is None:
        args.spread_delay_epochs = 1
    if args.top_focus_loss_weight is None:
        args.top_focus_loss_weight = 0.001
    if args.top_focus_temperature is None:
        args.top_focus_temperature = 0.75
    if args.top_focus_delay_epochs is None:
        args.top_focus_delay_epochs = 0
    if args.pairwise_top_loss_weight is None:
        args.pairwise_top_loss_weight = 0.0005
    if args.pairwise_top_frac is None:
        args.pairwise_top_frac = 0.10
    if args.pairwise_num_pairs is None:
        args.pairwise_num_pairs = 512
    if args.pairwise_model_top_weight is None:
        args.pairwise_model_top_weight = 0.5
    if args.pairwise_delay_epochs is None:
        args.pairwise_delay_epochs = 1


def _load_v9_matching_weights(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = ckpt.get("model_state_dict", ckpt)
    target = model.state_dict()
    loaded = {}
    skipped = []
    for key, value in source.items():
        if key.startswith("horizon_heads."):
            skipped.append((key, "skip_horizon_head"))
            continue
        if key.startswith("gat_") or key.startswith("fusion_gate_") or key.startswith("cross_ind_attn_"):
            skipped.append((key, "skip_gat_branch"))
            continue
        if key in target and tuple(target[key].shape) == tuple(value.shape):
            loaded[key] = value
        else:
            skipped.append((key, "missing_or_shape"))
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    loaded_keys = set(loaded)
    print(
        f"Loaded V9 warm-start: path={checkpoint_path} matched={len(loaded)} "
        f"missing_after_partial={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}",
        flush=True,
    )
    return loaded_keys


def _set_loaded_params_trainable(model, loaded_keys, trainable):
    if not loaded_keys:
        return
    for name, param in model.named_parameters():
        if name in loaded_keys:
            param.requires_grad = bool(trainable)


def _rank_embed(model, X, mask):
    return model._build_rank_embed(X, mask)


@torch.no_grad()
def forward_temporal_chunked(model, X, X_seq, risk_cont, mask, industry_ids, temporal_chunk):
    """Equivalent to model.forward, but chunks only the per-stock temporal path."""
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


@torch.no_grad()
def evaluate_temporal(
    model,
    loader,
    cfg,
    device,
    temporal_chunk=None,
    amp=False,
    limit_dates=None,
    progress_every=0,
):
    model.eval()
    h_indices = list(cfg.horizon_indices)
    top_fracs = tuple(getattr(cfg, "eval_top_fracs", (0.05, 0.10)))
    min_eval_stocks = max(10, int(getattr(cfg, "min_stocks_per_time", 30)))
    results = {"alpha": []}
    for h_idx in h_indices:
        results[f"h{h_idx + 1}"] = []
        results[f"topbot_h{h_idx + 1}"] = []
        for frac in top_fracs:
            tag = str(int(round(float(frac) * 100)))
            results[f"topret_h{h_idx + 1}_top{tag}"] = []
            results[f"topic_h{h_idx + 1}_top{tag}"] = []

    regime_dim = get_regime_dim(cfg)
    for batch_idx, batch in enumerate(loader):
        if limit_dates is not None and batch_idx >= limit_dates:
            break
        if progress_every and batch_idx % progress_every == 0:
            print(f"    eval batch {batch_idx}/{len(loader)} | {_resource_status(device)}", flush=True)
        X = batch["X"].to(device)
        X_seq = batch["X_seq"].to(device)
        y_seq = batch["y_seq"].to(device)
        risk = batch["risk"].to(device)
        industry_ids = batch["industry_ids"].to(device)
        mask = batch["mask"].to(device)

        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            if temporal_chunk and X_seq.shape[1] > temporal_chunk:
                alpha_raw, horizon_preds = forward_temporal_chunked(
                    model,
                    X,
                    X_seq,
                    risk[..., :regime_dim],
                    mask,
                    industry_ids,
                    temporal_chunk,
                )
            else:
                alpha_raw, _, horizon_preds = model(X, X_seq, risk[..., :regime_dim], mask, industry_ids)
            target_weighted, _, _ = weighted_horizon_target(y_seq, cfg)

        alpha_cpu = alpha_raw.cpu()
        horizon_cpu = horizon_preds.cpu()
        target_cpu = target_weighted.cpu()
        y_seq_cpu = y_seq.cpu()
        mask_cpu = mask.cpu()

        for b in range(alpha_cpu.shape[0]):
            m = mask_cpu[b]
            n_valid = int(m.sum().item())
            if n_valid < min_eval_stocks:
                continue
            pred = alpha_cpu[b][m].numpy()
            target = target_cpu[b][m].numpy()
            ic = np.corrcoef(pred, target)[0, 1] if pred.size > 1 else 0.0
            if np.isfinite(ic):
                results["alpha"].append(float(ic))

            order = np.argsort(pred)
            k10 = max(5, int(n_valid * 0.10))
            top10 = order[-k10:]
            bot10 = order[:k10]
            for j, h_idx in enumerate(h_indices):
                if h_idx >= y_seq_cpu.shape[-1]:
                    continue
                ret_h = y_seq_cpu[b, m, h_idx].numpy()
                if j < horizon_cpu.shape[-1]:
                    hp = horizon_cpu[b, m, j].numpy()
                    ic_h = np.corrcoef(hp, ret_h)[0, 1] if hp.size > 1 else 0.0
                    if np.isfinite(ic_h):
                        results[f"h{h_idx + 1}"].append(float(ic_h))
                spread = ret_h[top10].mean() - ret_h[bot10].mean()
                if np.isfinite(spread):
                    results[f"topbot_h{h_idx + 1}"].append(float(spread))
                for frac in top_fracs:
                    tag = str(int(round(float(frac) * 100)))
                    k = max(5, int(n_valid * frac))
                    top_idx = order[-k:]
                    top_ret = ret_h[top_idx].mean()
                    if np.isfinite(top_ret):
                        results[f"topret_h{h_idx + 1}_top{tag}"].append(float(top_ret))
                    top_pred = pred[top_idx]
                    top_y = ret_h[top_idx]
                    if k > 10 and np.std(top_pred) > 1e-8 and np.std(top_y) > 1e-8:
                        top_ic = np.corrcoef(top_pred, top_y)[0, 1]
                        if np.isfinite(top_ic):
                            results[f"topic_h{h_idx + 1}_top{tag}"].append(float(top_ic))

    return {key: (float(np.mean(vals)) if vals else 0.0) for key, vals in results.items()}


def _load_or_build_meta(args, cfg):
    if args.meta:
        with open(args.meta, "rb") as f:
            return pickle.load(f)
    return build_temporal_cross_section_dataset(cfg, use_cache=True)


def _compute_best_score(val_metrics, best_metric):
    if not best_metric.startswith("composite:"):
        return val_metrics.get(best_metric, val_metrics.get("alpha", 0.0)), best_metric

    expr = best_metric.split(":", 1)[1]
    total = 0.0
    parts = []
    for raw_part in expr.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid composite metric part: {part!r}")
        name, weight_s = part.split("=", 1)
        name = name.strip()
        weight = float(weight_s.strip())
        value = float(val_metrics.get(name, 0.0))
        total += weight * value
        parts.append(f"{name}={value:.4f}*{weight:g}")
    label = "composite(" + ", ".join(parts) + ")"
    return float(total), label


def _resource_status(device):
    parts = []
    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserved_mb = torch.cuda.memory_reserved(device) / 1024 / 1024
        parts.append(f"cuda_alloc={alloc_mb:.0f}MB")
        parts.append(f"cuda_reserved={reserved_mb:.0f}MB")
    try:
        if os.name == "nt":
            import psutil
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        else:
            import resource
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        parts.append(f"rss={rss_mb:.0f}MB")
    except Exception:
        pass
    return " | ".join(parts)


def _set_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)


def main():
    args = _parse_args()
    _set_seed(args.seed)
    cfg = DataConfig()
    cfg.data_dir = args.data_dir
    cfg.seq_len = args.seq_len
    cfg.max_stocks = args.max_stocks
    cfg.force_rebuild = args.force_rebuild
    cfg.use_technical_features = not args.no_tech
    cfg.use_market_features = not args.no_market
    cfg.use_macro_features = args.macro
    if args.horizon_mode == "h5":
        cfg.horizon_indices = (4,)
        cfg.horizon_weights = (1.0,)
    elif args.horizon_mode == "h3h5":
        cfg.horizon_indices = (2, 4)
        cfg.horizon_weights = (0.35, 0.65)
    elif args.horizon_mode == "h3h5h7":
        cfg.horizon_indices = (2, 4, 6)
        cfg.horizon_weights = (0.25, 0.45, 0.30)
    else:
        cfg.horizon_weights = (0.10, 0.30, 0.30, 0.30)
    cfg.temporal_lookback = args.lookback
    if args.history_calendar_days is not None:
        cfg.temporal_history_calendar_days = args.history_calendar_days
    cfg.temporal_train_start = args.train_start
    cfg.temporal_val_start = args.val_start
    cfg.temporal_test_start = args.test_start
    cfg.temporal_end_date = str(RESEARCH_END_DATE.date())
    cfg.best_val_metric = args.best_metric
    if args.spread_loss_weight is not None:
        cfg.spread_loss_weight = float(args.spread_loss_weight)
    if args.spread_delay_epochs is not None:
        cfg.spread_delay_epochs = int(args.spread_delay_epochs)
    if args.top_focus_loss_weight is not None:
        cfg.top_focus_loss_weight = float(args.top_focus_loss_weight)
    if args.top_focus_temperature is not None:
        cfg.top_focus_temperature = float(args.top_focus_temperature)
    if args.top_focus_delay_epochs is not None:
        cfg.top_focus_delay_epochs = int(args.top_focus_delay_epochs)
    if args.pairwise_top_loss_weight is not None:
        cfg.pairwise_top_loss_weight = float(args.pairwise_top_loss_weight)
    if args.pairwise_top_frac is not None:
        cfg.pairwise_top_frac = float(args.pairwise_top_frac)
    if args.pairwise_num_pairs is not None:
        cfg.pairwise_num_pairs = int(args.pairwise_num_pairs)
    if args.pairwise_model_top_weight is not None:
        cfg.pairwise_model_top_weight = float(args.pairwise_model_top_weight)
    if args.pairwise_delay_epochs is not None:
        cfg.pairwise_delay_epochs = int(args.pairwise_delay_epochs)

    meta = _load_or_build_meta(args, cfg)
    cfg.max_horizon = int(meta.get("max_horizon", cfg.max_horizon))
    cfg.target_horizon = int(meta.get("target_horizon", cfg.target_horizon))
    cfg.min_stocks_per_time = int(meta.get("min_stocks", cfg.min_stocks_per_time))
    cfg.use_macro_features = int(meta.get("risk_full_dim", 0)) > (6 + N_MARKET)

    train_ds = TemporalMemmapDataset(meta, split="train")
    val_ds = TemporalMemmapDataset(meta, split="val")
    if args.train_limit_dates is not None:
        train_ds = Subset(train_ds, list(range(min(int(args.train_limit_dates), len(train_ds)))))
    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(args.seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_temporal_train(
            b,
            keep_ratio=args.keep_ratio,
            max_stocks=args.max_train_stocks,
            sample_mode=args.train_sample_mode,
            top_frac=args.sample_top_frac,
            bottom_frac=args.sample_bottom_frac,
            random_frac=args.sample_random_frac,
            target_horizon=args.sample_target_horizon,
        ),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
        generator=loader_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_temporal_eval(
            b,
            max_stocks=None if args.full_eval else args.max_eval_stocks,
        ),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regime_dim = get_regime_dim(cfg)
    agg_groups = [(int(meta["high_feat_dim"]), 5, 0.1)]
    low_base_dim = int(meta.get("low_agg_dim", 0)) // 2
    if low_base_dim > 0:
        agg_groups.append((low_base_dim, 2, 0.0))
    model = TemporalCrossAlphaModel(
        input_dim=int(meta["x_dim"]),
        seq_dim=int(meta["seq_dim"]),
        agg_groups=agg_groups,
        hidden_dim=args.hidden_dim,
        temporal_dim=args.temporal_dim,
        n_horizons=len(cfg.horizon_indices),
        n_alpha=4,
        regime_dim=regime_dim,
        num_industries=int(meta["n_industries"]),
        dropout=args.dropout if args.dropout is not None else getattr(cfg, "transformer_dropout", 0.35),
        temporal_dropout=args.temporal_dropout,
        temporal_fusion_cross_gate_init=args.temporal_fusion_cross_gate_init,
        temporal_fusion_mode=args.temporal_fusion_mode,
        transformer_norm_first=bool(args.transformer_norm_first) if args.transformer_norm_first is not None else True,
        temporal_residual_adapters=args.temporal_residual_adapters,
        temporal_adapter_dropout=args.temporal_adapter_dropout,
        temporal_adapter_gate_init=args.temporal_adapter_gate_init,
    ).to(device)

    loaded_v9_keys = set()
    if args.init_v9_checkpoint:
        loaded_v9_keys = _load_v9_matching_weights(model, args.init_v9_checkpoint, device)
        if args.freeze_v9_epochs > 0:
            _set_loaded_params_trainable(model, loaded_v9_keys, False)
            print(f"Frozen {len(loaded_v9_keys)} V9-warm-start parameters for {args.freeze_v9_epochs} epochs", flush=True)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    start_epoch = 0
    best_loss = float("inf")
    if args.resume and os.path.exists(args.save_path):
        ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_loss = float(ckpt.get("val_loss", best_loss))
        print(f"Resumed {args.save_path} from epoch {start_epoch}")

    print(
        f"Temporal training: train_dates={len(train_ds)}, val_dates={len(val_ds)}, "
        f"x_dim={meta['x_dim']}, seq=({meta['seq_lookback']},{meta['seq_dim']}), "
        f"keep_ratio={args.keep_ratio}, max_train_stocks={args.max_train_stocks}, "
        f"max_eval_stocks={'full' if args.full_eval else args.max_eval_stocks}, "
        f"eval_temporal_chunk={args.eval_temporal_chunk}, sample_mode={args.train_sample_mode}, "
        f"horizon_mode={args.horizon_mode}, temporal_adapters={args.temporal_residual_adapters}, "
        f"preset={args.preset}, dropout={args.dropout}, temporal_dropout={args.temporal_dropout}, "
        f"fusion_mode={args.temporal_fusion_mode}, fusion_gate_init={args.temporal_fusion_cross_gate_init}, "
        f"transformer_norm_first={args.transformer_norm_first}, adapter_gate_init={args.temporal_adapter_gate_init}, "
        f"init_v9={args.init_v9_checkpoint}, freeze_v9_epochs={args.freeze_v9_epochs}, "
        f"hidden_dim={args.hidden_dim}, temporal_dim={args.temporal_dim}, accum_steps={args.accum_steps}, "
        f"seed={args.seed}, device={device}",
        flush=True,
    )

    if args.eval_before_train:
        val_metrics = evaluate_temporal(
            model,
            val_loader,
            cfg,
            device,
            temporal_chunk=args.eval_temporal_chunk if args.full_eval else None,
            amp=args.amp,
            limit_dates=args.eval_limit_dates,
            progress_every=args.eval_progress_every,
        )
        score, score_label = _compute_best_score(val_metrics, args.best_metric)
        print(
            f"Pretrain eval {score_label}={score:.4f} alpha={val_metrics.get('alpha', 0.0):.4f} "
            f"metrics={val_metrics}",
            flush=True,
        )
        val_loss = -score
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": 0,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "best_metric": args.best_metric,
                    "best_score": score,
                    "val_metrics": val_metrics,
                    "arch_config": model.arch_config(),
                    "horizon_indices": tuple(cfg.horizon_indices),
                    "horizon_weights": tuple(cfg.horizon_weights),
                    "temporal_meta_path": meta.get("meta_path"),
                    "full_eval": bool(args.full_eval),
                    "eval_temporal_chunk": int(args.eval_temporal_chunk),
                    "eval_limit_dates": args.eval_limit_dates,
                    "seed": int(args.seed),
                    "pretrain_eval": True,
                },
                args.save_path,
            )
            print(f"  saved pretrain best temporal checkpoint: {args.save_path}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if args.freeze_v9_epochs > 0 and epoch == args.freeze_v9_epochs and loaded_v9_keys:
            _set_loaded_params_trainable(model, loaded_v9_keys, True)
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch), eta_min=1e-5)
            print(f"Unfroze V9-warm-start parameters at epoch {epoch + 1}", flush=True)
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        total_loss_meter = 0.0

        for step, batch in enumerate(train_loader):
            if step % max(1, args.log_every) == 0:
                n_valid = int(batch["mask"].sum().item())
                print(
                    f"  Epoch {epoch + 1}/{args.epochs} | batch {step}/{len(train_loader)} "
                    f"| valid_stocks={n_valid} | loss_so_far={total_loss_meter / max(step, 1):.4f} "
                    f"| {_resource_status(device)}",
                    flush=True,
                )
            X = batch["X"].to(device)
            X_seq = batch["X_seq"].to(device)
            y = batch["y"].to(device)
            y_seq = batch["y_seq"].to(device)
            risk = batch["risk"].to(device)
            industry_ids = batch["industry_ids"].to(device)
            mask = batch["mask"].to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                alpha_raw, alphas, horizon_preds = model(X, X_seq, risk[..., :regime_dim], mask, industry_ids)
                loss, _ = total_loss_v7(
                    alpha_raw,
                    alphas,
                    horizon_preds,
                    y,
                    y_seq,
                    mask,
                    cfg,
                    industry_ids=industry_ids,
                    spread_enabled=epoch >= getattr(cfg, "spread_delay_epochs", 5),
                    top_focus_enabled=epoch >= getattr(cfg, "top_focus_delay_epochs", 5),
                    pairwise_enabled=epoch >= getattr(cfg, "pairwise_delay_epochs", 5),
                )
                loss = loss / args.accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss_meter += float(loss.item()) * args.accum_steps

        if len(train_loader) % args.accum_steps != 0:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        val_metrics = evaluate_temporal(
            model,
            val_loader,
            cfg,
            device,
            temporal_chunk=args.eval_temporal_chunk if args.full_eval else None,
            amp=args.amp,
            limit_dates=args.eval_limit_dates,
            progress_every=args.eval_progress_every,
        )
        score, score_label = _compute_best_score(val_metrics, args.best_metric)
        val_loss = -score
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train={total_loss_meter / max(len(train_loader), 1):.4f} "
            f"{score_label}={score:.4f} alpha={val_metrics.get('alpha', 0.0):.4f} "
            f"time={(time.time() - t0) / 60:.1f}m",
            flush=True,
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "best_metric": args.best_metric,
                    "best_score": score,
                    "val_metrics": val_metrics,
                    "arch_config": model.arch_config(),
                    "horizon_indices": tuple(cfg.horizon_indices),
                    "horizon_weights": tuple(cfg.horizon_weights),
                    "temporal_meta_path": meta.get("meta_path"),
                    "full_eval": bool(args.full_eval),
                    "eval_temporal_chunk": int(args.eval_temporal_chunk),
                    "eval_limit_dates": args.eval_limit_dates,
                    "seed": int(args.seed),
                },
                args.save_path,
            )
            print(f"  saved best temporal checkpoint: {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
