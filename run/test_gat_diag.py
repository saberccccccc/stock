"""GAT鎻愰€熻瘖鏂?細鐢ㄥ凡鏈夊叏閲廲ache锛岄檺鍒舵牱鏈?暟鏉ユ祴train/val鑰楁椂"""
import sys, os, gc, time, warnings
os.chdir(r"F:\stock_prediction\deepseek_optimized")
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore')

import torch
import numpy as np
from core.config import DataConfig
from data.pipeline import build_cross_section_dataset, N_AGGS, INDUSTRY_REL_FEATURES
from core.train_utils import CrossSectionDataset, collate_fn, collate_fn_eval, get_regime_dim
from core.model import UltimateV7Model
from torch.utils.data import DataLoader

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
print(f"GPU: {torch.cuda.get_device_name(0)}")

print("Loading full cache...")
t0 = time.time()
train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)
print(f"  Done in {time.time()-t0:.1f}s | Train: {len(train_samples)}, Val: {len(val_samples)}")
print(f"  Stocks/train sample: min={min(s['X'].shape[0] for s in train_samples):,} max={max(s['X'].shape[0] for s in train_samples):,}")

regime_dim = get_regime_dim(cfg)
for s in train_samples + val_samples:
    s['risk'] = s['risk'][:, :regime_dim].copy()
gc.collect()

input_dim = train_samples[0]["X"].shape[1]
industry_rel_dim = len(INDUSTRY_REL_FEATURES)
total_agg = (input_dim - industry_rel_dim) // 2
base_feat_dim = total_agg // N_AGGS
all_ids = np.concatenate([s['industry_ids'] for s in train_samples])
num_industries = int(all_ids.max()) + 1
print(f"  input_dim={input_dim}, base_feat_dim={base_feat_dim}, num_industries={num_industries}")

N_TRAIN_SAMPLES = 32
N_VAL_SAMPLES = 8
BS = 4

train_subset = train_samples[:N_TRAIN_SAMPLES]
val_subset = val_samples[:N_VAL_SAMPLES]
print(f"\nUsing {len(train_subset)} train + {len(val_subset)} val samples, BS={BS}")

train_ds = CrossSectionDataset(train_subset)
val_ds = CrossSectionDataset(val_subset)

train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,
    collate_fn=lambda b: collate_fn(b, keep_ratio=1.0, min_keep=30), num_workers=0, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False,
    collate_fn=collate_fn_eval, num_workers=0, pin_memory=False)

model = UltimateV7Model(input_dim, base_feat_dim, n_aggs=N_AGGS,
    hidden_dim=256, n_heads=8, n_layers=4,
    n_horizons=len(cfg.horizon_indices), n_alpha=4,
    use_gat=True, regime_dim=regime_dim, num_industries=num_industries).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# warmup
dummy = torch.randn(1, 50, input_dim, device=device)
_ = model(dummy, dummy[..., :regime_dim], torch.ones(1, 50, dtype=torch.bool, device=device),
          torch.zeros(1, 50, dtype=torch.long, device=device))
torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()

# === TRAIN timing ===
print("\n=== TRAIN (keep_ratio=1.0, fwd+bwd) ===")
model.train()
train_times = []
for i, batch in enumerate(train_loader):
    X = batch["X"].to(device)
    y = batch["y"].to(device)
    y_seq = batch["y_seq"].to(device)
    risk = batch["risk"][..., :regime_dim].to(device)
    mask = batch["mask"].to(device)
    industry_ids = batch["industry_ids"].to(device)
    N = mask.sum().item()

    torch.cuda.synchronize()
    t0 = time.time()
    alpha_raw, alphas, horizon_preds = model(X, risk, mask, industry_ids)
    loss = -(alpha_raw * y).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t1 = time.time()
    train_times.append((N, t1 - t0))

    del X, y, y_seq, risk, mask, industry_ids, alpha_raw, alphas, horizon_preds, loss
    stocks_str = ",".join([str(x.item()) for x in batch["mask"].sum(dim=1)])
    print(f"  batch {i}: stocks=[{stocks_str}] batch_N={N:5d} | {t1-t0:.3f}s | peak={torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# === VAL timing (cold cache, first pass through these val batches) ===
print("\n=== VAL (cold cache) ===")
model.eval()
val_times = []
for i, batch in enumerate(val_loader):
    X = batch["X"].to(device)
    y = batch["y"].to(device)
    y_seq = batch["y_seq"].to(device)
    risk = batch["risk"][..., :regime_dim].to(device)
    mask = batch["mask"].to(device)
    industry_ids = batch["industry_ids"].to(device)
    N = mask.sum().item()

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        alpha_raw, _, horizon_preds = model(X, risk, mask, industry_ids)
    torch.cuda.synchronize()
    t1 = time.time()
    val_times.append((N, t1 - t0))

    del X, y, y_seq, risk, mask, industry_ids, alpha_raw, horizon_preds
    stocks_str = ",".join([str(x.item()) for x in batch["mask"].sum(dim=1)])
    print(f"  batch {i}: stocks=[{stocks_str}] batch_N={N:5d} | {t1-t0:.3f}s | peak={torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

print(f"\n{'='*60}")
print(f"TRAIN avg: {np.mean([t[1] for t in train_times]):.3f}s/batch")
print(f"VAL   avg: {np.mean([t[1] for t in val_times]):.3f}s/batch")
print(f"Ratio (val/train): {np.mean([t[1] for t in val_times]) / max(np.mean([t[1] for t in train_times]), 1e-8):.2f}x")
print(f"{'='*60}")
