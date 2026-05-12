"""Test validation phase: GAT eval OOM detection"""
import sys, os, gc, time, warnings
os.chdir(r"F:\stock_prediction\deepseek_optimized")
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore')

import torch
import numpy as np
from core.config import DataConfig
from data.pipeline import build_cross_section_dataset, N_AGGS, INDUSTRY_REL_FEATURES
from core.train_utils import CrossSectionDataset, collate_fn, collate_fn_eval, evaluate, get_regime_dim
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
print(f"GPU: {torch.cuda.get_device_name(0)}, Mem: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

print("Loading dataset...")
train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)

# test smaller val set
val_samples = val_samples[:20]
print(f"Using {len(val_samples)} val samples")

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

# Check val stock counts
stocks_per_val = [s["X"].shape[0] for s in val_samples]
print(f"Val stocks: min={min(stocks_per_val)}, max={max(stocks_per_val)}, median={np.median(stocks_per_val):.0f}")

val_ds = CrossSectionDataset(val_samples)
val_loader = DataLoader(
    val_ds, batch_size=4, shuffle=False,
    collate_fn=collate_fn_eval, num_workers=0, pin_memory=False,
)

model = UltimateV7Model(
    input_dim, base_feat_dim, n_aggs=N_AGGS,
    hidden_dim=256, n_heads=8, n_layers=4,
    n_horizons=len(cfg.horizon_indices), n_alpha=4,
    use_gat=True, regime_dim=regime_dim,
    num_industries=num_industries,
).to(device)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

print("\n=== Testing evaluate (20 val samples, bs=4) ===")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
try:
    val_ics = evaluate(model, val_loader, cfg, device)
    t1 = time.time()
    print(f"Evaluate OK in {t1-t0:.1f}s")
    print(f"Val ICs: {val_ics}")
    print(f"Peak mem: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")
except Exception as e:
    t1 = time.time()
    print(f"Evaluate FAILED after {t1-t0:.1f}s: {e}")
    import traceback
    traceback.print_exc()
