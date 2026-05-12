"""璁?粌鍚?姩鑴氭湰锛氬惈鏃ュ織杈撳嚭"""
import sys
import os
os.chdir(r"F:\stock_prediction\deepseek_optimized")
sys.path.insert(0, os.getcwd())

# 鍚屾椂杈撳嚭鍒版枃浠跺拰鎺у埗鍙?
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

log_file = open("train.log", "w", encoding="utf-8")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = Tee(original_stdout, log_file)
sys.stderr = Tee(original_stderr, log_file)

print("=" * 60)
print("V9 璁?粌鍚?姩锛堝惈甯傚満鏁翠綋灞炴€?+ 澧炲己Alpha澶达級")
print("=" * 60)

from core.config import DataConfig
from data.pipeline import build_cross_section_dataset, N_AGGS, INDUSTRY_REL_FEATURES
from core.train_utils import CrossSectionDataset, collate_fn, collate_fn_eval, train_model
import torch
from torch.utils.data import DataLoader

cfg = DataConfig()
cfg.use_technical_features = True
cfg.use_macro_features = True
cfg.min_stocks_per_time = 30
cfg.target_horizon = 5
cfg.seq_len = 40
cfg.max_horizon = 10
cfg.use_market_features = True
cfg.test_mode = False          # V9: 鍏ㄩ噺璁?粌锛?332鍙?偂绁?級

print(f"閰嶇疆: target_horizon={cfg.target_horizon}, seq_len={cfg.seq_len}, "
      f"horizons={cfg.horizon_indices}, weights={cfg.horizon_weights}, "
      f"market={cfg.use_market_features}")

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

from core.train_utils import get_regime_dim
num_industries = train_samples[0]["risk"].shape[1] - get_regime_dim(cfg)
print(f"num_industries={num_industries} (+1 unknown)")

train_ds = CrossSectionDataset(train_samples)
val_ds = CrossSectionDataset(val_samples)

sample_n = train_samples[0]["X"].shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_mem < 8:
        batch_size, accum_steps = 2, 4
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
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU mem after crash: alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB, "
              f"cache={torch.cuda.memory_reserved()/1024**3:.2f}GB")

sys.stdout = original_stdout
sys.stderr = original_stderr
log_file.close()
