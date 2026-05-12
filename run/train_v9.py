"""V9 training script: strong regularization version"""
import sys
import os
from datetime import datetime
os.chdir(r"F:\stock_prediction\deepseek_optimized")
sys.path.insert(0, os.getcwd())

class Tee:
    """Tee stdout/stderr to a log file. Suppresses tqdm \r progress lines from log."""
    def __init__(self, original, logfile):
        self.original = original
        self.logfile = logfile

    def write(self, obj):
        self.original.write(obj)
        # Skip tqdm-style \r progress lines from the log file
        if '\r' not in obj and not self.logfile.closed:
            self.logfile.write(obj)
            if '\n' in obj:
                self.logfile.flush()

    def flush(self):
        self.original.flush()
        if not self.logfile.closed:
            self.logfile.flush()

# Windows涓嬪己鍒禪TF-8杈撳嚭
import locale
if sys.platform == 'win32':
    import _locale
    _locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])

log_file = open("train_v9.log", "w", encoding="utf-8")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = Tee(original_stdout, log_file)
sys.stderr = Tee(original_stderr, log_file)

print("=" * 60)
print("V9 璁?粌鍚?姩锛堝己姝ｅ垯鍖栫増鏈?級")
print("=" * 60)
print("V9 vs V8 鏀硅繘:")
print("  - Transformer dropout: 0.3 鈫?0.35")
print("  - weight_decay: 5e-4 鈫?2e-3 (4x)")
print("  - gradient clip: 0.3 鈫?0.2")
print("  - fresh start (no checkpoint reuse)")
print("  - fresh start (no checkpoint reuse)")
print("=" * 60)

from core.config import DataConfig
from data.pipeline import build_cross_section_dataset, N_AGGS, INDUSTRY_REL_FEATURES
from data.market_features import N_MARKET
from core.train_utils import CrossSectionDataset, collate_fn, collate_fn_eval, train_model
import torch
from torch.utils.data import DataLoader

cfg = DataConfig()
cfg.use_technical_features = True
cfg.min_stocks_per_time = 30
cfg.target_horizon = 5
cfg.seq_len = 40
cfg.max_horizon = 10
cfg.use_market_features = True
cfg.use_macro_features = True
cfg.test_mode = False          # 浣跨敤鍏ㄩ儴鑲$エ

print(f"閰嶇疆: target_horizon={cfg.target_horizon}, seq_len={cfg.seq_len}, "
      f"horizons={cfg.horizon_indices}, weights={cfg.horizon_weights}, "
      f"market={cfg.use_market_features}, macro={cfg.use_macro_features}")

print("\n鏋勫缓鏁版嵁闆?..")
train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)

input_dim = train_samples[0]["X"].shape[1]
horizon = train_samples[0]["y_seq"].shape[1]
print(f"Input dim: {input_dim}, Horizon labels: {horizon}")
print(f"璁?粌鏍锋湰: {len(train_samples)}, 楠岃瘉鏍锋湰: {len(val_samples)}")

industry_rel_dim = len(INDUSTRY_REL_FEATURES)
total_agg = (input_dim - industry_rel_dim) // 2
base_feat_dim = total_agg // N_AGGS
print(f"base_feat_dim={base_feat_dim}, n_aggs={N_AGGS}")

# 鎵撳嵃璇︾粏缁村害淇℃伅
print("\n" + "="*60)
print("鐗瑰緛缁村害璇︽儏:")
print("="*60)
agg_dim = base_feat_dim * N_AGGS
rank_dim = agg_dim
print(f"X鐗瑰緛 (input_dim={input_dim}):")
print(f"  - 鑱氬悎鐗瑰緛: {agg_dim}缁?({base_feat_dim}涓?熀纭EUR鐗瑰緛 脳 {N_AGGS}绉嶈仛鍚?")
print(f"  - Rank鐗瑰緛: {rank_dim}缁?(鎴?潰鎺掑簭)")
print(f"  - 琛屼笟鐩稿?: {industry_rel_dim}缁?")
print(f"  - 鍚堣?: {agg_dim} + {rank_dim} + {industry_rel_dim} = {input_dim}缁?")

risk_dim = train_samples[0]["risk"].shape[1]
from core.train_utils import get_regime_dim
stock_risk_dim = 3
market_feat_dim = N_MARKET
macro_feat_dim = 3 if cfg.use_macro_features else 0
regime_dim = get_regime_dim(cfg)
industry_onehot_dim = risk_dim - regime_dim
num_industries = industry_onehot_dim
print(f"\nRisk鐗瑰緛 (total={risk_dim}):")
print(f"  - 鑲$エ绾ч?闄? {stock_risk_dim}缁?(size, vol, mom)")
print(f"  - 甯傚満鐗瑰緛: {market_feat_dim}缁?(16瀹藉熀+3瀹藉害+31琛屼笟鏀剁泭+31琛屼笟鍙?敤鎬?")
print(f"  - 瀹忚?/璧勯噾娴? {macro_feat_dim}缁?(north, margin, PMI)")
print(f"  - 琛屼笟one-hot: {industry_onehot_dim}缁?")
print(f"  - 琛屼笟embedding: {num_industries}涓?湡瀹炶?涓?+ 1涓?湭鐭ヨ?涓?")
print(f"  - full: {regime_dim}dim (stock {stock_risk_dim} + market {market_feat_dim} + macro {macro_feat_dim})")
print("="*60 + "\n")

train_ds = CrossSectionDataset(train_samples)
val_ds = CrossSectionDataset(val_samples)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_mem < 8:
        batch_size, accum_steps = 2, 8      # 绛夋晥 batch 16
    else:
        batch_size, accum_steps = 4, 4
else:
    batch_size, accum_steps, gpu_mem = 4, 4, 99

val_batch_size = 1 if (device.type == "cuda" and gpu_mem < 8) else batch_size
print(f"Batch size: {batch_size} (val: {val_batch_size}), Accum steps: {accum_steps}")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn, num_workers=0, pin_memory=False,
)
val_loader = DataLoader(
    val_ds, batch_size=val_batch_size, shuffle=False,
    collate_fn=collate_fn_eval, num_workers=0, pin_memory=False,
)

import traceback

try:
    model = train_model(
        train_loader, val_loader, input_dim, base_feat_dim, cfg,
        n_aggs=N_AGGS, n_alpha=4, n_horizons=len(cfg.horizon_indices),
        epochs=25, lr=3e-4, weight_decay=2e-3, accum_steps=accum_steps,
        grad_clip=0.2, use_amp=False, num_industries=num_industries,
    )
    print("璁?粌瀹屾垚")
except Exception as e:
    print(f"\nFATAL ERROR: {e}")
    traceback.print_exc()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_info = (torch.cuda.memory_allocated()/1024**3,
                     torch.cuda.memory_reserved()/1024**3)
        print(f"GPU mem after crash: alloc={mem_info[0]:.2f}GB, cache={mem_info[1]:.2f}GB")

sys.stdout = original_stdout
sys.stderr = original_stderr
log_file.close()
