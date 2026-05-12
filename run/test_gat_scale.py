"""鍏ㄩ噺瑙勬ā GAT 鍗昩atch鑰楁椂+鏄惧瓨娴嬭瘯"""

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

print(f"input_dim={input_dim}, base_feat_dim={base_feat_dim}")



# Use later samples (more stocks) for realistic test

print(f"Total samples: {len(train_samples)}")

stocks_per_sample = [s["X"].shape[0] for s in train_samples]

print(f"Stocks per sample: min={min(stocks_per_sample)}, max={max(stocks_per_sample)}, median={np.median(stocks_per_sample):.0f}")

# Pick samples near the 75th percentile for a realistic max

p75 = int(np.percentile(stocks_per_sample, 75))

print(f"75th percentile: {p75} stocks (after keep_ratio=0.5: ~{p75//2})")



late_idx = [i for i in range(len(train_samples)) if stocks_per_sample[i] > p75][:20]

print(f"Late samples (>{p75} stocks): {len(late_idx)}")



train_ds = CrossSectionDataset([train_samples[i] for i in late_idx])

sample_sizes = [train_ds[i]["X"].shape[0] for i in range(len(train_ds))]

print(f"Test sample stocks: {sample_sizes}")



# Test batch_size=4

print("\n=== batch_size=4 ===")

batch = collate_fn([train_ds[i] for i in range(4)], keep_ratio=0.5, min_keep=30)

print(f"Batch after keep_ratio=0.5: X={batch['X'].shape}, stocks={batch['mask'].sum(dim=1).tolist()}")



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

t0 = time.time()

alpha_raw, alphas, horizon_preds = model(X, risk, mask, industry_ids)

torch.cuda.synchronize()

t1 = time.time()

print(f"Forward: {t1-t0:.3f}s, alloc={torch.cuda.memory_allocated()/1024**2:.1f}MB, peak={torch.cuda.max_memory_allocated()/1024**2:.1f}MB")



loss = -(alpha_raw * batch["y"].to(device)).mean()

loss.backward()

torch.cuda.synchronize()

t2 = time.time()

print(f"Forward+Backward: {t2-t0:.3f}s, peak={torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

del X, risk, mask, industry_ids, alpha_raw, alphas, horizon_preds, loss

gc.collect(); torch.cuda.empty_cache()



# Test batch_size=8

print("\n=== batch_size=8 ===")

batch_big = collate_fn([train_ds[i] for i in range(min(8, len(train_ds)))], keep_ratio=0.5, min_keep=30)

print(f"Batch after keep_ratio=0.5: X={batch_big['X'].shape}, stocks={batch_big['mask'].sum(dim=1).tolist()}")



X = batch_big["X"].to(device)

risk = batch_big["risk"][..., :regime_dim].to(device)

mask = batch_big["mask"].to(device)

industry_ids = batch_big["industry_ids"].to(device)



torch.cuda.synchronize()

t0 = time.time()

alpha_raw, alphas, horizon_preds = model(X, risk, mask, industry_ids)

torch.cuda.synchronize()

t1 = time.time()

print(f"Forward: {t1-t0:.3f}s, peak={torch.cuda.max_memory_allocated()/1024**2:.1f}MB")



loss = -(alpha_raw * batch_big["y"].to(device)).mean()

loss.backward()

torch.cuda.synchronize()

t2 = time.time()

print(f"Forward+Backward: {t2-t0:.3f}s, peak={torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

del X, mask, industry_ids, alpha_raw, alphas, horizon_preds, loss

gc.collect(); torch.cuda.empty_cache()

print(f"After cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.max_memory_allocated()

print(f"Free memory: {free/1024**2:.1f}MB")

