# train.py 鈥?V7 澶氬懆鏈熻仈鍚堣?缁冭剼鏈?
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
# V9: tqdm removed 鈥?too verbose, use simple prints instead
import pickle
import time
import warnings

warnings.filterwarnings('ignore')

from data.market_features import N_MARKET
from data.pipeline import MACRO_COLS

# regime缁村害 = 3涓?偂绁ㄧ骇椋庨櫓鍥犲瓙 + 甯傚満鏁翠綋灞炴EUR?+ 鍙?EUR夊畯瑙?璧勯噾娴佺壒寰?
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
    """鍔ㄦEUR佸瓙閲囨牱锛氭瘡鎴?潰闅忔満淇濈暀 keep_ratio 鑲＄エ"""
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


# ============================ 鎹熷け鍑芥暟 ============================
def _masked_corr_loss_1d(pred, target, mask):
    pred_p = pred[mask]
    target_p = target[mask]
    if pred_p.numel() < 10:
        return None
    pred_z = (pred_p - pred_p.mean()) / (pred_p.std() + 1e-8)
    target_z = (target_p - target_p.mean()) / (target_p.std() + 1e-8)
    return -(pred_z * target_z).mean()


def correlation_rank_loss(pred, target, mask):
    """Compute Pearson correlation loss"""
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


def total_loss_v7(alpha_raw, alphas, horizon_preds, y, y_seq, mask, cfg):
    """
    V7 多周期联合损失
    - main: alpha_raw vs 主标签(加权多周期目标)
    - multi: 各horizon独立预测 vs 对应标签
    - div: alpha头多样性正则
    """

    target_weighted, valid_indices, norm_weights = weighted_horizon_target(y_seq, cfg)

    # 涓?loss
    main = correlation_rank_loss(alpha_raw, target_weighted, mask)

    # 澶氬懆鏈?loss
    multi = 0.0
    for j, (idx, w) in enumerate(zip(valid_indices, norm_weights)):
        if j < horizon_preds.shape[-1]:
            multi = multi + w * correlation_rank_loss(
                horizon_preds[..., j], y_seq[..., idx], mask
            )

    # 澶氭牱鎬ф?鍒欙細鎯╃綒alpha澶翠箣闂寸殑鐩稿叧鎬э紝閬垮厤澶氬ご濉岀缉
    div = alpha_diversity_loss(alphas, mask)

    return main + 0.3 * multi + 0.05 * div


# ============================ 璇勪及 ============================
@torch.no_grad()
def evaluate(model, loader, cfg, device):
    """Compute validation set Rank IC per horizon (chunked processing, memory cleanup)"""
    model.eval()
    h_indices = list(cfg.horizon_indices)
    all_ics = {f"h{h_indices[i]+1}": [] for i in range(len(h_indices))}
    all_ics["alpha"] = []

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
            msg = str(e).lower()
            if "out of memory" not in msg and "cuda" not in msg:
                raise
            print(f"  [WARN] 楠岃瘉 batch {bi} OOM: {e}, 璺宠繃")
            # OOM鏃朵篃瑕佹竻鐞嗗凡鍒嗛厤鐨勮緭鍏?ensor
            del X, y, y_seq, risk, industry_ids, mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        target_weighted, _, _ = weighted_horizon_target(y_seq, cfg)

        # 绔嬪嵆绉诲埌CPU閲婃斁鏄惧瓨
        alpha_cpu = alpha_raw.cpu()
        horizon_cpu = horizon_preds.cpu()
        y_cpu = y.cpu()
        y_seq_cpu = y_seq.cpu()
        target_cpu = target_weighted.cpu()
        mask_cpu = mask.cpu()

        # 閲婃斁GPU tensor
        del X, y, y_seq, risk, industry_ids, mask, alpha_raw, horizon_preds, target_weighted
        if device.type == "cuda" and bi % 100 == 0:
            torch.cuda.empty_cache()

        for b in range(alpha_cpu.shape[0]):
            m = mask_cpu[b]
            if m.sum() < 10:
                continue

            # Alpha IC
            pred_np = alpha_cpu[b][m].numpy()
            target_np = target_cpu[b][m].numpy()
            ic = np.corrcoef(pred_np, target_np)[0, 1] if len(pred_np) > 1 else 0
            if np.isfinite(ic):
                all_ics["alpha"].append(ic)

            # 鍚勫懆鏈?IC
            for j, h_idx in enumerate(h_indices):
                if h_idx < y_seq_cpu.shape[-1] and j < horizon_cpu.shape[-1]:
                    hp_np = horizon_cpu[b, m, j].numpy()
                    yh_np = y_seq_cpu[b, m, h_idx].numpy()
                    ic_h = np.corrcoef(hp_np, yh_np)[0, 1] if len(hp_np) > 1 else 0
                    if np.isfinite(ic_h):
                        all_ics[f"h{h_idx+1}"].append(ic_h)

        del alpha_cpu, horizon_cpu, y_cpu, y_seq_cpu, target_cpu, mask_cpu

    results = {}
    for k, v in all_ics.items():
        results[k] = np.mean(v) if v else 0.0
    return results


# ============================ 璁?粌涓诲嚱鏁?============================
def train_model(train_loader, val_loader, input_dim, base_feat_dim, cfg,
                n_aggs=5, n_alpha=4, n_horizons=4, epochs=30,
                lr=3e-4, weight_decay=2e-3, accum_steps=1, grad_clip=0.3,
                use_amp=True, resume=True, num_industries=83, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"鏄惧瓨: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("浣跨敤CPU璁?粌")

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
    print(f"UltimateV7Model 鍙傛暟閲? {n_params:,}")

    best_model_path = save_path or "ultimate_v7_best.pt"
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # AMP锛氳嚜鍔ㄦ贩鍚堢簿搴︼紝鑺傜渷GPU鏄惧瓨40%
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    amp_str = "AMP ON" if scaler else "AMP OFF"
    print(f"娣峰悎绮惧害(AMP): {amp_str}")

    if resume and os.path.exists(best_model_path):
        print(f"loading existing checkpoint {best_model_path}, continuing training...")
        checkpoint = torch.load(best_model_path, map_location=device)
        arch_config = checkpoint.get('arch_config', None)

        to_load = True
        if arch_config is not None:
            for k, v in current_arch.items():
                if k in arch_config and arch_config[k] != v:
                    print(f"  鏋舵瀯涓嶅尮閰? {k}={arch_config[k]} (褰撳墠={v})")
                    to_load = False
            if not to_load:
                print("[FIXED]")

        if to_load:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except (RuntimeError, KeyError) as e:
                print(f"  妫EUR鏌ョ偣缁撴瀯涓嶅吋瀹癸紝蹇界暐鏃фā鍨嬪苟浠庡ご璁?粌: {e}")
                to_load = False

        if to_load:
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 0)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"restored optimizer state")
            else:
                print(f"  (妫EUR鏌ョ偣鏃爋ptimizer鐘舵EUR侊紝浣跨敤鍏ㄦ柊浼樺寲鍣?")
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"restored optimizer state")
            print(f"浠?epoch {start_epoch} 鎭㈠?, best val_loss={best_val_loss:.4f}")

    print(f"\n{'='*60}")
    print(f"寮EUR濮嬭?缁?| epochs={epochs} | lr={lr} | accum={accum_steps}")
    print(f"horizon_indices={cfg.horizon_indices} | weights={cfg.horizon_weights}")
    print(f"{'='*60}\n")

    epoch_times = []
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()

        print(f"\n--- Epoch {epoch+1} TRAINING start ---")
        train_loss = 0.0
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
                loss = total_loss_v7(alpha_raw, alphas, horizon_preds, y, y_seq, mask, cfg)
            loss = loss / accum_steps

            # 閲婃斁妯"瀷杈撳嚭tensor锛岃繖浜涘湪loss涓?笉鍐嶉渶瑕?
            del alpha_raw, alphas, horizon_preds

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 鎻愬墠鎻愬彇鏍囬噺鍊肩敤浜庣粺璁★紝鐒跺悗閲婃斁loss tensor
            loss_val = loss.item() * accum_steps
            del loss

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
                # 鏈?疮绉?埌姝ユ椂绔嬪嵆閲婃斁鎵EUR鏈夎緭鍏?ensor
                del X, y, y_seq, risk, industry_ids, mask

            train_loss += loss_val

            # 姣?00鎵规竻鐞嗕竴娆?PU缂撳瓨锛岄槻姝?鐗囩疮绉?
            if device.type == "cuda" and i > 0 and i % 200 == 0:
                torch.cuda.empty_cache()

        # 瀹夊叏娓呯悊寰?幆涓?畫鐣欑殑tensor锛堟渶鍚庝竴姝ュ彲鑳藉凡閲婃斁閮ㄥ垎锛?
        try:
            del X, y, y_seq, risk, industry_ids, mask
        except NameError:
            pass
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 澶勭悊鍓╀綑姊?害
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

        # 娓呯悊GPU鍐呭瓨纰庣墖锛屼负楠岃瘉鑵惧嚭绌洪棿
        if device.type == "cuda":
            torch.cuda.empty_cache()
            import gc; gc.collect()

        # 楠岃瘉
        t_val = time.time()
        val_ics = evaluate(model, val_loader, cfg, device)
        print(f"--- VAL done in {time.time()-t_val:.1f}s ---")
        val_loss = -val_ics["alpha"]  # 鐢?-RankIC 浣滀负鏃╁仠鏍囧噯

        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = float(np.mean(epoch_times))
        remaining_epochs = epochs - epoch - 1
        eta_seconds = avg_epoch_time * remaining_epochs
        print(
            f"Epoch {epoch+1} 鐢ㄦ椂: {epoch_time/60:.1f} min | "
            f"骞冲潎: {avg_epoch_time/60:.1f} min/epoch | "
            f"棰勮?鍓╀綑: {eta_seconds/3600:.2f} h"
        )
        ic_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_ics.items()])
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val IC 鈫?{ic_str}")

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
            print(f"  >>> 淇濆瓨鏈EUR浼樻ā鍨?(val_alpha_IC={val_ics['alpha']:.4f})")

    # 鍔犺浇鏈EUR浼樻ā鍨?
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n璁?粌瀹屾瘯锛屽凡鍔犺浇鏈EUR浼樻ā鍨?(epoch {checkpoint['epoch']}, "
              f"alpha_IC={checkpoint.get('val_ics', {}).get('alpha', 'N/A')})")
    return model


# ============================ 涓荤▼搴忓叆鍙?============================
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

    # 鍛戒护琛屽弬鏁版敮鎸?
    if "--test" in sys.argv:
        cfg.test_mode = True
        cfg.test_stocks = int(sys.argv[sys.argv.index("--test") + 1]) if len(sys.argv) > sys.argv.index("--test") + 1 and sys.argv[sys.argv.index("--test") + 1].isdigit() else 1000

    if "--full" in sys.argv:
        cfg.test_mode = False
        print("full training mode")
    if "--gat" in sys.argv:
        cfg.use_gat = True
        print("鍚?敤 GAT 鍒嗘敮")

    print(f"閰嶇疆: target_horizon={cfg.target_horizon}, seq_len={cfg.seq_len}, "
          f"horizons={cfg.horizon_indices}, weights={cfg.horizon_weights}, "
          f"market={cfg.use_market_features}")

    # 鍔犺浇鏁版嵁
    print("\n鏋勫缓鏁版嵁闆?..")
    train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)

    input_dim = train_samples[0]["X"].shape[1]
    horizon = train_samples[0]["y_seq"].shape[1]
    print(f"Input dim: {input_dim}, Horizon labels: {horizon}")
    print(f"璁?粌鏍锋湰: {len(train_samples)}, 楠岃瘉鏍锋湰: {len(val_samples)}")

    # 璁$畻 base_feat_dim锛堣仛鍚堢壒寰佺殑鍩虹?缁村害锛?
    from data.pipeline import N_AGGS, INDUSTRY_REL_FEATURES
    industry_rel_dim = len(INDUSTRY_REL_FEATURES)
    total_agg = (input_dim - industry_rel_dim) // 2
    base_feat_dim = total_agg // N_AGGS
    print(f"base_feat_dim={base_feat_dim}, n_aggs={N_AGGS}")

    train_ds = CrossSectionDataset(train_samples)
    val_ds = CrossSectionDataset(val_samples)

    # 鏍规嵁鏄惧瓨璋冩暣 batch size 涓庣疮绉??鏁?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda' and torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
        batch_size = 4
        accum_steps = 8       # 绛夋晥 32
    else:
        batch_size = 16
        accum_steps = 2       # 绛夋晥 32

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
    print("璁?粌瀹屾垚")
