"""蹇?€熸祴璇旼AT鍗昩atch鍓嶅悜+鍙嶅悜锛岀湅鏄惧瓨鍗犵敤"""
import sys, os, gc, time, warnings
os.chdir(r"F:\stock_prediction\deepseek_optimized")
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore')

import torch
import numpy as np
from core.config import DataConfig
from data.pipeline import build_cross_section_dataset, N_AGGS, INDUSTRY_REL_FEATURES
from core.train_utils import CrossSectionDataset, collate_fn, get_regime_dim
from core.model import UltimateV7Model

cfg = DataConfig()
cfg.use_technical_features = True
cfg.use_gat = True
cfg.min_stocks_per_time = 30
cfg.target_horizon = 5
cfg.seq_len = 40
cfg.max_horizon = 10
cfg.use_market_features = True
cfg.use_macro_features = True
cfg.test_mode = False

device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats()
print(f"Device: {torch.cuda.get_device_name(0)}, Total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

print("Loading dataset...")
train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)
regime_dim = get_regime_dim(cfg)
for s in train_samples + val_samples:
    s["risk"] = s["risk"][:, :regime_dim].copy()
gc.collect()

input_dim = train_samples[0]["X"].shape[1]
industry_rel_dim = len(INDUSTRY_REL_FEATURES)
total_agg = (input_dim - industry_rel_dim) // 2
base_feat_dim = total_agg // N_AGGS
all_ids = np.concatenate([s["industry_ids"] for s in train_samples])
num_industries = int(all_ids.max()) + 1
print(f"input_dim={input_dim}, base_feat_dim={base_feat_dim}, num_industries={num_industries}")

train_ds = CrossSectionDataset(train_samples[:10])  # 10 samples for testing

# Build batch (batch_size=4)
batch = collate_fn([train_ds[i] for i in range(4)], keep_ratio=0.5, min_keep=30)
print(f"Batch: X={batch['X'].shape}, stocks={batch['mask'].sum(dim=1).tolist()}")

model = UltimateV7Model(
    input_dim, base_feat_dim, n_aggs=N_AGGS,
    hidden_dim=256, n_heads=8, n_layers=4,
    n_horizons=len(cfg.horizon_indices), n_alpha=4,
    use_gat=True, regime_dim=regime_dim,
    num_industries=num_industries,
).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

X = batch["X"].to(device)
risk = batch["risk"][..., :regime_dim].to(device)
mask = batch["mask"].to(device)
industry_ids = batch["industry_ids"].to(device)

torch.cuda.synchronize()
before = torch.cuda.memory_allocated()

print("\n=== Forward ===")
t0 = time.time()
alpha_raw, alphas, horizon_preds = model(X, risk, mask, industry_ids)
torch.cuda.synchronize()
t1 = time.time()
print(f"Time: {t1-t0:.3f}s")
print(f"alpha_raw: {alpha_raw.shape}")
print(f"Forward alloc: {(torch.cuda.memory_allocated()-before)/1024**2:.1f}MB")
print(f"Peak alloc: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

print("\n=== Backward ===")
loss = -(alpha_raw * batch["y"].to(device)).mean()
t0 = time.time()
loss.backward()
torch.cuda.synchronize()
t1 = time.time()
print(f"Time: {t1-t0:.3f}s")
print(f"After backward alloc: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
print(f"Peak alloc (after backward): {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

del X, risk, mask, industry_ids, alpha_raw, alphas, horizon_preds, loss
gc.collect()
torch.cuda.empty_cache()
print(f"\nAfter cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

print("\n=== Batches 4 (bs=4, accum=4) ===")
torch.cuda.reset_peak_memory_stats()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
accum_steps = 4
peak = 0
for step in range(4):
    batch = collate_fn([train_ds[i] for i in range(4)], keep_ratio=0.5, min_keep=30)
    X = batch["X"].to(device)
    y = batch["y"].to(device)
    y_seq = batch["y_seq"].to(device)
    risk = batch["risk"][..., :regime_dim].to(device)
    mask = batch["mask"].to(device)
    industry_ids = batch["industry_ids"].to(device)

    alpha_raw, alphas, horizon_preds = model(X, risk, mask, industry_ids)
    loss = -(alpha_raw * y).mean() / accum_steps
    del alpha_raw, alphas, horizon_preds
    loss.backward()
    loss_val = loss.item()
    del loss

    if (step + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
        optimizer.step()
        optimizer.zero_grad()
    else:
        del X, y, y_seq, risk, industry_ids, mask

    current = torch.cuda.memory_allocated()
    peak = max(peak, torch.cuda.max_memory_allocated())
    torch.cuda.synchronize()
    print(f"  Step {step}: alloc={current/1024**2:.1f}MB, peak={torch.cuda.max_memory_allocated()/1024**2:.1f}MB, loss={loss_val:.4f}")

print(f"\nOverall peak: {peak/1024**2:.1f}MB ({peak/1024**3:.2f}GB)")
print(f"Free: {(torch.cuda.get_device_properties(0).total_memory-peak)/1024**2:.1f}MB")
