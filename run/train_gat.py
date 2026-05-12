"""GAT training: V9 + industry graph attention fusion version"""
import sys
import os
from datetime import datetime
os.chdir(r"F:\stock_prediction\deepseek_optimized")
sys.path.insert(0, os.getcwd())

class LogWriter:
    """Write-only log file, no pipe to stdout (prevent Windows pipe blocking)"""
    def __init__(self, logfile):
        self.logfile = logfile
    def write(self, obj):
        if '\r' not in obj and not self.logfile.closed:
            self.logfile.write(obj)
            if '\n' in obj:
                self.logfile.flush()
    def flush(self):
        if not self.logfile.closed:
            self.logfile.flush()

import locale
if sys.platform == 'win32':
    import _locale
    _locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])

log_file = open("train_gat.log", "w", encoding="utf-8")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = LogWriter(log_file)
sys.stderr = LogWriter(log_file)

print("GAT training (V9 + industry graph attention)")
print("GAT training (V9 + industry graph attention)")
print("=" * 60)
print("改进:")
print("  - FeatureGrouper 鈫?Transformer锛堣法鑲$エ锛?")
print("  - GATConv锛?灞傦紝琛屼笟鍏ㄨ繛鎺ュ浘锛?")
print("  - FusionGate: 鑷?EUR傚簲铻嶅悎Transformer+GAT")
print("  - 行业embedding + rank embedding")
print("=" * 60)

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
# cfg.test_stocks = 100

print(f"配置: target_horizon={cfg.target_horizon}, seq_len={cfg.seq_len}, "
      f"horizons={cfg.horizon_indices}, weights={cfg.horizon_weights}, "
      f"market={cfg.use_market_features}, macro={cfg.use_macro_features}, "
      f"use_gat={cfg.use_gat}")

print("\n鏋勫缓鏁版嵁闆?..")
train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)

# 鎴?柇risk到regime_dim锛堝幓鎺?3缁磋?业one-hot，节省~49% risk鍐呭瓨锛?regime_dim = get_regime_dim(cfg)
for s in train_samples + val_samples:
    s['risk'] = s['risk'][:, :regime_dim].copy()
import gc
gc.collect()

input_dim = train_samples[0]["X"].shape[1]
horizon = train_samples[0]["y_seq"].shape[1]
print(f"Input dim: {input_dim}, Horizon labels: {horizon}")
print(f"璁?粌鏍锋湰: {len(train_samples)}, 验证样本: {len(val_samples)}")

industry_rel_dim = len(INDUSTRY_REL_FEATURES)
total_agg = (input_dim - industry_rel_dim) // 2
base_feat_dim = total_agg // N_AGGS

print(f"\n特征维度:")
risk_dim = train_samples[0]["risk"].shape[1]
all_ids = np.concatenate([s['industry_ids'] for s in train_samples])
num_industries = int(all_ids.max()) + 1
print(f"  input_dim={input_dim}, base_feat_dim={base_feat_dim}, n_aggs={N_AGGS}")
print(f"  regime_dim={regime_dim}, risk_dim={risk_dim}, industries={num_industries}")
print(f"  GAT: {num_industries}涓??涓氬叏杩炴帴鍥?+ {base_feat_dim}涓?熀纭EUR特征")

train_ds = CrossSectionDataset(train_samples)
val_ds = CrossSectionDataset(val_samples)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_mem < 8:
        # GAT: batch_size=8, 宄板EUR紐1.85GB, RTX 2060 6GB富余4.3GB
        batch_size, accum_steps = 8, 2   # 等效batch 16
    else:
        batch_size, accum_steps = 4, 4   # 澶ф樉瀛? 等效batch 16
else:
    batch_size, accum_steps = 2, 8

val_batch_size = 4
print(f"Batch size: {batch_size} (val: {val_batch_size}), Accum steps: {accum_steps}")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    collate_fn=lambda b: collate_fn(b, keep_ratio=0.9, min_keep=30),
    num_workers=0, pin_memory=False,
)
val_loader = DataLoader(
    val_ds, batch_size=val_batch_size, shuffle=False,
    collate_fn=collate_fn_eval, num_workers=0, pin_memory=False,
)

best_model_path = "ultimate_v7_gat_best.pt"
import traceback

try:
    model = train_model(
        train_loader, val_loader, input_dim, base_feat_dim, cfg,
        n_aggs=N_AGGS, n_alpha=4, n_horizons=len(cfg.horizon_indices),
        epochs=25, lr=3e-4, weight_decay=2e-3, accum_steps=accum_steps,
        grad_clip=0.2, use_amp=False, num_industries=num_industries,
        save_path=best_model_path,
    )
    print("GAT璁?粌瀹屾垚")
except Exception as e:
    print(f"\nFATAL ERROR: {e}")
    traceback.print_exc()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

sys.stdout = original_stdout
sys.stderr = original_stderr
log_file.close()
