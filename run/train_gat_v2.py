"""GAT training (no Tee version, avoid pipe blocking)"""
import sys, os
from datetime import datetime
os.chdir(r"F:\stock_prediction\deepseek_optimized")
sys.path.insert(0, os.getcwd())

import locale
if sys.platform == 'win32':
    import _locale
    _locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])

# 杈撳嚭鍙?啓log文件，不pipe到stdout锛堥槻姝?indows pipe闃诲?锛?log_file = open("train_gat.log", "w", encoding="utf-8")

def log(msg):
    log_file.write(msg + "\n")
    log_file.flush()

# 重定向stdout到log文件，避免Windows pipe闃诲?bash捕获
sys.stdout = log_file
sys.stderr = log_file

log("GAT training (V9 + industry graph attention)")
log("GAT training (V9 + industry graph attention)")
log("=" * 60)
log("改进:")
log("  - FeatureGrouper + Transformer (cross-stock attention)")
log("  - GATConv industry subgraph message passing")
log("  - FusionGate: 鑷?EUR傚簲铻嶅悎Transformer+GAT")
log("  - 行业embedding + rank embedding")
log("=" * 60)

from core.config import DataConfig
from data.pipeline import build_cross_section_dataset, N_AGGS, INDUSTRY_REL_FEATURES
from data.market_features import N_MARKET
from core.train_utils import CrossSectionDataset, collate_fn, collate_fn_eval, train_model, get_regime_dim
import numpy as np
import torch
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

log(f"配置: target_horizon={cfg.target_horizon}, seq_len={cfg.seq_len}, "
    f"horizons={cfg.horizon_indices}, weights={cfg.horizon_weights}, "
    f"market={cfg.use_market_features}, macro={cfg.use_macro_features}, "
    f"use_gat={cfg.use_gat}")

log("\n鏋勫缓鏁版嵁闆?..")
train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)

regime_dim = get_regime_dim(cfg)
for s in train_samples + val_samples:
    s['risk'] = s['risk'][:, :regime_dim].copy()
import gc
gc.collect()

input_dim = train_samples[0]["X"].shape[1]
horizon = train_samples[0]["y_seq"].shape[1]
log(f"Input dim: {input_dim}, Horizon labels: {horizon}")
log(f"璁?粌鏍锋湰: {len(train_samples)}, 验证样本: {len(val_samples)}")

industry_rel_dim = len(INDUSTRY_REL_FEATURES)
total_agg = (input_dim - industry_rel_dim) // 2
base_feat_dim = total_agg // N_AGGS

risk_dim = train_samples[0]["risk"].shape[1]
regime_dim = get_regime_dim(cfg)
all_ids = np.concatenate([s['industry_ids'] for s in train_samples])
num_industries = int(all_ids.max()) + 1
log(f"  input_dim={input_dim}, base_feat_dim={base_feat_dim}, n_aggs={N_AGGS}")
log(f"  regime_dim={regime_dim}, risk_dim={risk_dim}, industries={num_industries}")
log(f"  GAT: {num_industries}涓??涓氬叏杩炴帴鍥?+ {base_feat_dim}涓?熀纭EUR特征")

train_ds = CrossSectionDataset(train_samples)
val_ds = CrossSectionDataset(val_samples)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    batch_size, accum_steps = 8, 2
else:
    batch_size, accum_steps = 2, 8

val_batch_size = 4
log(f"Batch size: {batch_size} (val: {val_batch_size}), Accum steps: {accum_steps}")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    collate_fn=lambda b: collate_fn(b, keep_ratio=0.5, min_keep=30),
    num_workers=0, pin_memory=False,
)
val_loader = DataLoader(
    val_ds, batch_size=val_batch_size, shuffle=False,
    collate_fn=collate_fn_eval, num_workers=0, pin_memory=False,
)

best_model_path = "ultimate_v7_gat_best.pt"
if os.path.exists(best_model_path):
    os.remove(best_model_path)

import traceback
try:
    model = train_model(
        train_loader, val_loader, input_dim, base_feat_dim, cfg,
        n_aggs=N_AGGS, n_alpha=4, n_horizons=len(cfg.horizon_indices),
        epochs=25, lr=3e-4, weight_decay=2e-3, accum_steps=accum_steps,
        grad_clip=0.2, use_amp=False, resume=False, num_industries=num_industries,
        save_path=best_model_path,
    )
    log("GAT璁?粌瀹屾垚")
except Exception as e:
    log(f"\nFATAL ERROR: {e}")
    traceback.print_exc(file=log_file)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

log_file.close()
