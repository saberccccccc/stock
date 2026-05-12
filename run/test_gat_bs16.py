"""娴婫AT BS=16鏄惧瓨鍗犵敤"""

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



print("Loading full cache...")

train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)

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



# Use later samples (more stocks) for worst-case test

stocks = [s["X"].shape[0] for s in train_samples]

p75 = int(np.percentile(stocks, 75))

late = [i for i in range(len(train_samples)) if stocks[i] > p75][:32]

print(f"Using {len(late)} high-stock samples (>{p75} stocks)")



train_ds = CrossSectionDataset([train_samples[i] for i in late])



for BS in [8, 16]:

    torch.cuda.reset_peak_memory_stats()

    batch = collate_fn([train_ds[i] for i in range(min(BS, len(train_ds)))], keep_ratio=0.5, min_keep=30)

    N = batch["mask"].sum().item()

    print(f"\n=== BS={BS} (stocks={N}) ===")



    model = UltimateV7Model(input_dim, base_feat_dim, n_aggs=N_AGGS,

        hidden_dim=256, n_heads=8, n_layers=4,

        n_horizons=len(cfg.horizon_indices), n_alpha=4,

        use_gat=True, regime_dim=regime_dim, num_industries=num_industries).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)



    X = batch["X"].to(device)

    y = batch["y"].to(device)

    risk = batch["risk"][..., :regime_dim].to(device)

    mask = batch["mask"].to(device)

    industry_ids = batch["industry_ids"].to(device)



    torch.cuda.synchronize()

    t0 = time.time()

    alpha_raw, alphas, horizon_preds = model(X, risk, mask, industry_ids)

    loss = -(alpha_raw * y).mean()

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    torch.cuda.synchronize()

    t1 = time.time()



    peak = torch.cuda.max_memory_allocated()

    print(f"  Forward+Backward: {t1-t0:.3f}s | Peak: {peak/1024**2:.1f}MB | Free: {(6*1024-peak/1024**2):.1f}MB")



    del X, y, risk, mask, industry_ids, alpha_raw, alphas, horizon_preds, loss, model, optimizer

    gc.collect()

    torch.cuda.empty_cache()

