# train.py - V7 多周期联合训练脚本
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
# V9: tqdm removed - too verbose, use simple prints instead
import pickle
import math
import time
import warnings

warnings.filterwarnings('ignore')

from data.market_features import N_MARKET
from data.pipeline import MACRO_COLS

# regime维度 = 3个股票级风险因子 + 市场整体属性 + 可选宏观/资金流特征
REGIME_BASE_DIM = 3 + N_MARKET
MACRO_REGIME_DIM = len(MACRO_COLS)
REGIME_DIM = REGIME_BASE_DIM


def get_regime_dim(cfg):
    return REGIME_BASE_DIM + (MACRO_REGIME_DIM if getattr(cfg, 'use_macro_features', False) else 0)


# ============================ Dataset ============================
class CrossSectionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "X": torch.from_numpy(s["X"]).float(),
            "y": torch.from_numpy(s["y"]).float(),
            "y_seq": torch.from_numpy(s["y_seq"]).float(),
            "risk": torch.from_numpy(s["risk"]).float(),
            "industry_ids": torch.from_numpy(s["industry_ids"]).long(),
        }


# ============================ Collate ============================
def collate_fn_eval(batch):
    B = len(batch)
    max_N = max(item["X"].shape[0] for item in batch)
    F = batch[0]["X"].shape[1]
    H = batch[0]["y_seq"].shape[1]
    R = batch[0]["risk"].shape[1]
    X = torch.zeros(B, max_N, F)
    y = torch.zeros(B, max_N)
    y_seq = torch.zeros(B, max_N, H)
    risk = torch.zeros(B, max_N, R)
    industry_ids = torch.full((B, max_N), -1, dtype=torch.long)
    mask = torch.zeros(B, max_N, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["X"].shape[0]
        X[i, :n] = item["X"]
        y[i, :n] = item["y"]
        y_seq[i, :n] = item["y_seq"]
        risk[i, :n] = item["risk"]
        industry_ids[i, :n] = item["industry_ids"]
        mask[i, :n] = 1
    return {"X": X, "y": y, "y_seq": y_seq, "risk": risk, "industry_ids": industry_ids, "mask": mask}


def collate_fn(batch, keep_ratio=0.7, min_keep=20):
    """动态子采样：每截面随机保留 keep_ratio 股票"""
    B = len(batch)
    X_list, y_list, yseq_list, risk_list, ind_list = [], [], [], [], []
    for item in batch:
        N = item["X"].shape[0]
        keep_n = max(min_keep, int(N * keep_ratio))
        keep_n = min(keep_n, N)
        idx = torch.randperm(N)[:keep_n]
        X_list.append(item["X"][idx])
        y_list.append(item["y"][idx])
        yseq_list.append(item["y_seq"][idx])
        risk_list.append(item["risk"][idx])
        ind_list.append(item["industry_ids"][idx])

    max_N = max(x.shape[0] for x in X_list)
    F = X_list[0].shape[1]
    H = yseq_list[0].shape[1]
    R = risk_list[0].shape[1]

    X = torch.zeros(B, max_N, F)
    y = torch.zeros(B, max_N)
    y_seq = torch.zeros(B, max_N, H)
    risk = torch.zeros(B, max_N, R)
    industry_ids = torch.full((B, max_N), -1, dtype=torch.long)
    mask = torch.zeros(B, max_N, dtype=torch.bool)

    for i in range(B):
        n = X_list[i].shape[0]
        X[i, :n] = X_list[i]
        y[i, :n] = y_list[i]
        y_seq[i, :n] = yseq_list[i]
        risk[i, :n] = risk_list[i]
        industry_ids[i, :n] = ind_list[i]
        mask[i, :n] = 1
    return {"X": X, "y": y, "y_seq": y_seq, "risk": risk, "industry_ids": industry_ids, "mask": mask}


# ============================ 损失函数 ============================
def _masked_corr_loss_1d(pred, target, mask):
    pred_p = pred[mask]
    target_p = target[mask]
    if pred_p.numel() < 10:
        return None
    pred_z = (pred_p - pred_p.mean()) / (pred_p.std() + 1e-8)
    target_z = (target_p - target_p.mean()) / (target_p.std() + 1e-8)
    return -(pred_z * target_z).mean()


def correlation_ic_loss(pred, target, mask):
    """Compute negative Pearson IC loss"""
    if pred.dim() == 1:
        loss = _masked_corr_loss_1d(pred, target, mask)
        return loss if loss is not None else torch.tensor(0.0, device=pred.device)

    losses = []
    for b in range(pred.shape[0]):
        loss = _masked_corr_loss_1d(pred[b], target[b], mask[b])
        if loss is not None:
            losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(losses).mean()


def alpha_diversity_loss(alphas, mask):
    losses = []
    for b in range(alphas.shape[0]):
        valid = mask[b]
        if valid.sum() < 10:
            continue
        a = alphas[b, valid]
        a = (a - a.mean(dim=0, keepdim=True)) / (a.std(dim=0, keepdim=True) + 1e-8)
        corr = (a.T @ a) / max(a.shape[0] - 1, 1)
        off_diag = corr - torch.diag(torch.diag(corr))
        losses.append((off_diag ** 2).mean())
    if not losses:
        return torch.tensor(0.0, device=alphas.device)
    return torch.stack(losses).mean()


def weighted_horizon_target(y_seq, cfg):
    h_indices = list(cfg.horizon_indices)
    h_weights = list(cfg.horizon_weights)
    max_h = y_seq.shape[-1]
    valid_indices = [i for i in h_indices if i < max_h]
    valid_weights = [h_weights[j] for j, i in enumerate(h_indices) if i < max_h]
    w_sum = sum(valid_weights) + 1e-8
    norm_weights = [w / w_sum for w in valid_weights]

    target_weighted = torch.zeros_like(y_seq[..., 0])
    for idx, w in zip(valid_indices, norm_weights):
        target_weighted = target_weighted + w * y_seq[..., idx]
    return target_weighted, valid_indices, norm_weights


def _masked_corr_within_industry_1d(pred, target, mask, industry_ids, min_stocks=5):
    """单截面：每个行业内算 Pearson IC，等权平均所有行业"""
    valid_mask = mask & (industry_ids >= 0)
    if valid_mask.sum() < 20:
        return _masked_corr_loss_1d(pred, target, mask)

    unique_inds = torch.unique(industry_ids[valid_mask])
    within_corrs = []
    for ind in unique_inds:
        ind_mask = valid_mask & (industry_ids == ind)
        if ind_mask.sum() < min_stocks:
            continue
        pred_i = pred[ind_mask]
        target_i = target[ind_mask]
        pred_z = (pred_i - pred_i.mean()) / (pred_i.std() + 1e-8)
        target_z = (target_i - target_i.mean()) / (target_i.std() + 1e-8)
        within_corrs.append((pred_z * target_z).mean())

    if not within_corrs:
        return _masked_corr_loss_1d(pred, target, mask)
    return -torch.stack(within_corrs).mean()


def correlation_ic_loss_within_industry(pred, target, mask, industry_ids):
    """批量版：逐截面计算行业内平均IC，再取batch平均"""
    losses = []
    for b in range(pred.shape[0]):
        loss = _masked_corr_within_industry_1d(pred[b], target[b], mask[b], industry_ids[b])
        if loss is not None:
            losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(losses).mean()


def total_loss_v7(alpha_raw, alphas, horizon_preds, y, y_seq, mask, cfg, industry_ids=None, spread_enabled=True):
    """
    V7 多周期联合损失
    Returns (total_loss, components) where components is a dict with per-component scalars.
    """
    comp = {}  # loss components for logging

    target_weighted, valid_indices, norm_weights = weighted_horizon_target(y_seq, cfg)

    # 主loss：全局IC + 行业内IC
    w_ind = getattr(cfg, 'industry_loss_weight', 0.0)
    main_global = correlation_ic_loss(alpha_raw, target_weighted, mask)
    comp['global_ic'] = main_global.detach()
    if w_ind > 0 and industry_ids is not None:
        main_industry = correlation_ic_loss_within_industry(alpha_raw, target_weighted, mask, industry_ids)
        comp['within_ic'] = main_industry.detach()
        main = (1 - w_ind) * main_global + w_ind * main_industry
    else:
        main = main_global

    # 多周期loss
    multi = 0.0
    for j, (idx, w) in enumerate(zip(valid_indices, norm_weights)):
        if j < horizon_preds.shape[-1]:
            multi = multi + w * correlation_ic_loss(
                horizon_preds[..., j], y_seq[..., idx], mask
            )
    comp['multi'] = multi.detach() if isinstance(multi, torch.Tensor) else torch.tensor(multi)

    # 多样性正则：惩罚alpha头之间的相关性，避免多头塌缩
    div = alpha_diversity_loss(alphas, mask)
    comp['div'] = div.detach()

    # Top-bottom spread loss：per-horizon 等权平均，避免 h5 主导
    w_spread = getattr(cfg, 'spread_loss_weight', 0.0)
    spread = torch.tensor(0.0, device=alpha_raw.device)
    if w_spread > 0 and spread_enabled:
        temp = getattr(cfg, 'spread_temperature', 0.5)
        n_h = 0
        for h_idx in valid_indices:
            if h_idx < y_seq.shape[-1]:
                spread = spread + top_bottom_spread_loss(alpha_raw, y_seq[..., h_idx], mask, temperature=temp)
                n_h += 1
        if n_h > 0:
            spread = spread / n_h
    comp['spread'] = spread.detach()

    total = main + 0.3 * multi + 0.05 * div + w_spread * spread
    return total, comp


def top_bottom_spread_loss(alpha_raw, ret, mask, min_stocks=40, temperature=0.5):
    """可微 Top-Bottom spread：softmax 加权多头-空头收益差"""
    losses = []
    for b in range(alpha_raw.shape[0]):
        valid_mask = mask[b]
        if valid_mask.sum() < min_stocks:
            continue
        a = alpha_raw[b][valid_mask]
        r = ret[b][valid_mask]
        r_z = (r - r.mean()) / (r.std() + 1e-8)
        long_w = torch.softmax(a / temperature, dim=0)
        short_w = torch.softmax(-a / temperature, dim=0)
        spread_val = (long_w * r_z).sum() - (short_w * r_z).sum()
        losses.append(-spread_val)
    if not losses:
        return torch.tensor(0.0, device=alpha_raw.device)
    return torch.stack(losses).mean()


# ============================ 评估 ============================
def _is_oom_error(error):
    msg = str(error).lower()
    return "out of memory" in msg


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    """Compute validation set Pearson IC per horizon + top/bottom spread metrics."""
    model.eval()
    h_indices = list(cfg.horizon_indices)
    all_ics = {f"h{h_indices[i]+1}": [] for i in range(len(h_indices))}
    all_ics["alpha"] = []
    top_spreads = {f"topbot_h{h_indices[i]+1}": [] for i in range(len(h_indices))}

    regime_dim = get_regime_dim(cfg)

    for bi, batch in enumerate(loader):
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        y_seq = batch["y_seq"].to(device)
        risk = batch["risk"].to(device)
        industry_ids = batch["industry_ids"].to(device)
        mask = batch["mask"].to(device)

        try:
            alpha_raw, _, horizon_preds = model(X, risk[..., :regime_dim], mask, industry_ids)
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            print(f"  [WARN] 验证 batch {bi} OOM: {e}, 跳过")
            del X, y, y_seq, risk, industry_ids, mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        target_weighted, _, _ = weighted_horizon_target(y_seq, cfg)

        # 立即移到CPU释放显存
        alpha_cpu = alpha_raw.cpu()
        horizon_cpu = horizon_preds.cpu()
        y_cpu = y.cpu()
        y_seq_cpu = y_seq.cpu()
        target_cpu = target_weighted.cpu()
        mask_cpu = mask.cpu()

        # 释放GPU tensor
        del X, y, y_seq, risk, industry_ids, mask, alpha_raw, horizon_preds, target_weighted
        if device.type == "cuda" and bi % 100 == 0:
            torch.cuda.empty_cache()

        for b in range(alpha_cpu.shape[0]):
            m = mask_cpu[b]
            n_valid = m.sum().item()
            if n_valid < 50:
                continue

            # Alpha IC
            pred_np = alpha_cpu[b][m].numpy()
            target_np = target_cpu[b][m].numpy()
            ic = np.corrcoef(pred_np, target_np)[0, 1] if len(pred_np) > 1 else 0
            if np.isfinite(ic):
                all_ics["alpha"].append(ic)

            # 各周期IC
            for j, h_idx in enumerate(h_indices):
                if h_idx < y_seq_cpu.shape[-1] and j < horizon_cpu.shape[-1]:
                    hp_np = horizon_cpu[b, m, j].numpy()
                    yh_np = y_seq_cpu[b, m, h_idx].numpy()
                    ic_h = np.corrcoef(hp_np, yh_np)[0, 1] if len(hp_np) > 1 else 0
                    if np.isfinite(ic_h):
                        all_ics[f"h{h_idx+1}"].append(ic_h)

            # Top-bottom spread: top 10% vs bottom 10% by alpha, per horizon
            k = max(5, int(n_valid * 0.10))
            top_idx = np.argpartition(pred_np, -k)[-k:]
            bot_idx = np.argpartition(pred_np, k)[:k]
            for h_idx in h_indices:
                if h_idx < y_seq_cpu.shape[-1]:
                    ret_h = y_seq_cpu[b, m, h_idx].numpy()
                    top_ret = ret_h[top_idx].mean()
                    bot_ret = ret_h[bot_idx].mean()
                    if np.isfinite(top_ret) and np.isfinite(bot_ret):
                        top_spreads[f"topbot_h{h_idx+1}"].append(top_ret - bot_ret)

        del alpha_cpu, horizon_cpu, y_cpu, y_seq_cpu, target_cpu, mask_cpu

    results = {}
    for k, v in all_ics.items():
        results[k] = np.mean(v) if v else 0.0
    for k, v in top_spreads.items():
        results[k] = np.mean(v) if v else 0.0
    return results


# ============================ 训练主函数============================
def train_model(train_loader, val_loader, input_dim, base_feat_dim, cfg,
                n_aggs=5, n_alpha=4, n_horizons=4, epochs=30,
                lr=3e-4, weight_decay=2e-3, accum_steps=1, grad_clip=0.3,
                use_amp=False, resume=True, num_industries=83, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("使用CPU训练")

    regime_dim = get_regime_dim(cfg)
    from core.model import UltimateV7Model
    model = UltimateV7Model(
        input_dim, base_feat_dim, n_aggs=n_aggs,
        hidden_dim=256, n_heads=8, n_layers=4,
        n_horizons=n_horizons, n_alpha=n_alpha,
        use_gat=getattr(cfg, 'use_gat', False),
        regime_dim=regime_dim,
        num_industries=num_industries,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"UltimateV7Model 参数量: {n_params:,}")

    best_model_path = save_path or "checkpoints/ultimate_v7_best.pt"
    start_epoch = 0
    best_val_loss = float('inf')

    current_arch = {
        'input_dim': input_dim, 'base_feat_dim': base_feat_dim,
        'n_aggs': n_aggs, 'hidden_dim': 256, 'n_heads': 8, 'n_layers': 4,
        'n_horizons': n_horizons, 'n_alpha': n_alpha,
        'use_gat': getattr(cfg, 'use_gat', False),
        'regime_dim': regime_dim, 'num_industries': num_industries,
    }

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup = getattr(cfg, 'lr_warmup_epochs', 0)
    if warmup > 0 and warmup < epochs:
        decay_epochs = epochs - warmup
        eta_min = 1e-5
        def lr_lambda(e):
            if e < warmup:
                return 1.0
            progress = (e - warmup) / max(decay_epochs, 1)
            return eta_min/lr + 0.5 * (1 - eta_min/lr) * (1 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # AMP：自动混合精度，节省GPU显存40%
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    amp_str = "AMP ON" if scaler else "AMP OFF"
    print(f"混合精度(AMP): {amp_str}")

    if resume and os.path.exists(best_model_path):
        print(f"loading existing checkpoint {best_model_path}, continuing training...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        arch_config = checkpoint.get('arch_config', None)

        to_load = True
        if arch_config is not None:
            for k, v in current_arch.items():
                if k in arch_config and arch_config[k] != v:
                    print(f"  架构不匹配 {k}={arch_config[k]} (当前={v})")
                    to_load = False
            if not to_load:
                print("[FIXED]")

        if to_load:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except (RuntimeError, KeyError) as e:
                print(f"  检查点结构不兼容，忽略旧模型并从头训练: {e}")
                to_load = False

        if to_load:
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 0)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"restored optimizer state")
            else:
                print(f"  (检查点无optimizer状态，使用全新优化器)")
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"restored scheduler state")
            print(f"从epoch {start_epoch} 恢复, best val_loss={best_val_loss:.4f}")

    print(f"\n{'='*60}")
    print(f"开始训练| epochs={epochs} | lr={lr} | accum={accum_steps}")
    print(f"horizon_indices={cfg.horizon_indices} | weights={cfg.horizon_weights}")
    print(f"{'='*60}\n")

    epoch_times = []
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()

        print(f"\n--- Epoch {epoch+1} TRAINING start ---")
        train_loss = 0.0
        train_comps = {}
        optimizer.zero_grad()

        n_batches = len(train_loader)
        report_every = max(1, n_batches // 4)  # report 4 times per epoch
        for i, batch in enumerate(train_loader):
            if i % report_every == 0:
                print(f"  Epoch {epoch+1}/{epochs} | batch {i}/{n_batches} | loss so far: {train_loss/max(i,1):.4f}")
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            y_seq = batch["y_seq"].to(device)
            risk = batch["risk"].to(device)
            industry_ids = batch["industry_ids"].to(device)
            mask = batch["mask"].to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                alpha_raw, alphas, horizon_preds = model(X, risk[..., :regime_dim], mask, industry_ids)
                spread_on = epoch >= getattr(cfg, 'spread_delay_epochs', 5)
                total, comp = total_loss_v7(alpha_raw, alphas, horizon_preds, y, y_seq, mask, cfg, industry_ids, spread_enabled=spread_on)
            loss = total / accum_steps

            # 释放模型输出tensor，这些在loss中不再需要
            del alpha_raw, alphas, horizon_preds

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 提前提取标量值用于统计，然后释放loss tensor
            loss_val = loss.item() * accum_steps
            # 累积各分量用于epoch级别打印
            for k, v in comp.items():
                train_comps[k] = train_comps.get(k, 0.0) + v.item()
            del loss, total, comp

            if (i + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            else:
                # 未累积到步时立即释放所有输入tensor
                del X, y, y_seq, risk, industry_ids, mask

            train_loss += loss_val

            # 每200批清理一次GPU缓存，防止碎片累积
            if device.type == "cuda" and i > 0 and i % 200 == 0:
                torch.cuda.empty_cache()

        # 安全清理循环中残留的tensor（最后一步可能已释放部分）
        try:
            del X, y, y_seq, risk, industry_ids, mask
        except NameError:
            pass
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 处理剩余梯度
        if len(train_loader) % accum_steps != 0:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(train_loader)
        print(f"--- Epoch {epoch+1} TRAINING done in {time.time()-epoch_start:.1f}s ---")

        # 清理GPU内存碎片，为验证腾出空间
        if device.type == "cuda":
            torch.cuda.empty_cache()
            import gc; gc.collect()

        # 验证
        t_val = time.time()
        val_ics = evaluate(model, val_loader, cfg, device)
        print(f"--- VAL done in {time.time()-t_val:.1f}s ---")
        val_loss = -val_ics["alpha"]  # negative alpha IC as early-stopping criterion

        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = float(np.mean(epoch_times))
        remaining_epochs = epochs - epoch - 1
        eta_seconds = avg_epoch_time * remaining_epochs
        print(
            f"Epoch {epoch+1} 用时: {epoch_time/60:.1f} min | "
            f"平均: {avg_epoch_time/60:.1f} min/epoch | "
            f"预计剩余: {eta_seconds/3600:.2f} h"
        )
        current_lr = optimizer.param_groups[0]['lr']

        # 构建loss分量字符串
        comp_strs = []
        for k in ['global_ic', 'within_ic', 'spread']:
            if k in train_comps:
                comp_strs.append(f"{k}={train_comps[k]/len(train_loader):.4f}")
        comp_str = "  [" + " | ".join(comp_strs) + "]" if comp_strs else ""

        # 分开 IC 和 top-bottom spread
        ic_items = {k: v for k, v in val_ics.items() if not k.startswith('topbot')}
        spread_items = {k: v for k, v in val_ics.items() if k.startswith('topbot')}
        ic_str = " | ".join([f"{k}: {v:.4f}" for k, v in ic_items.items()])
        spread_str = " | ".join([f"{k}: {v:.4f}" for k, v in spread_items.items()])

        print(f"Epoch {epoch+1} | LR: {current_lr:.2e} | Train: {train_loss:.4f}{comp_str}")
        print(f"         Val IC  -> {ic_str}")
        if spread_str:
            print(f"         Val TB  -> {spread_str}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_ics': val_ics,
                'arch_config': current_arch,
            }, best_model_path)
            print(f"  >>> 保存最优模型(val_alpha_IC={val_ics['alpha']:.4f})")

    # 加载最优模型
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n训练完毕，已加载最优模型(epoch {checkpoint['epoch']}, "
              f"alpha_IC={checkpoint.get('val_ics', {}).get('alpha', 'N/A')})")
    return model


# ============================ 主程序入口============================
if __name__ == "__main__":
    from core.config import DataConfig
    from data.pipeline import build_cross_section_dataset

    cfg = DataConfig()
    cfg.use_technical_features = True
    cfg.use_macro_features = True
    cfg.min_stocks_per_time = 30
    cfg.target_horizon = 5
    cfg.seq_len = 40
    cfg.max_horizon = 10

    # 命令行参数支持
    if "--test" in sys.argv:
        cfg.test_mode = True
        cfg.test_stocks = int(sys.argv[sys.argv.index("--test") + 1]) if len(sys.argv) > sys.argv.index("--test") + 1 and sys.argv[sys.argv.index("--test") + 1].isdigit() else 1000

    if "--full" in sys.argv:
        cfg.test_mode = False
        print("full training mode")
    if "--gat" in sys.argv:
        cfg.use_gat = True
        print("启用 GAT 分支")

    print(f"配置: target_horizon={cfg.target_horizon}, seq_len={cfg.seq_len}, "
          f"horizons={cfg.horizon_indices}, weights={cfg.horizon_weights}, "
          f"market={cfg.use_market_features}")

    # 加载数据
    print("\n构建数据集...")
    train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)

    input_dim = train_samples[0]["X"].shape[1]
    horizon = train_samples[0]["y_seq"].shape[1]
    print(f"Input dim: {input_dim}, Horizon labels: {horizon}")
    print(f"训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}")

    # 计算 base_feat_dim（聚合特征的基础维度）
    from data.pipeline import N_AGGS, INDUSTRY_REL_FEATURES
    industry_rel_dim = len(INDUSTRY_REL_FEATURES)
    total_agg = (input_dim - industry_rel_dim) // 2
    base_feat_dim = total_agg // N_AGGS
    print(f"base_feat_dim={base_feat_dim}, n_aggs={N_AGGS}")

    train_ds = CrossSectionDataset(train_samples)
    val_ds = CrossSectionDataset(val_samples)

    # 根据显存调整 batch size 与累积步数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda' and torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
        batch_size = 4
        accum_steps = 8       # 等效 32
    else:
        batch_size = 16
        accum_steps = 2       # 等效 32

    val_batch_size = batch_size

    print(f"Batch size: {batch_size} (val: {val_batch_size}), Accum steps: {accum_steps}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False,
        collate_fn=collate_fn_eval, num_workers=0, pin_memory=False,
    )

    model = train_model(
        train_loader, val_loader, input_dim, base_feat_dim, cfg,
        n_aggs=N_AGGS, n_alpha=4, n_horizons=len(cfg.horizon_indices),
        epochs=30, lr=3e-4, accum_steps=accum_steps, resume=False,
    )
    print("训练完成")
