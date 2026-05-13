"""GAT training: V9 + industry graph attention fusion version."""
import gc
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.config import DataConfig
from core.train_utils import CrossSectionDataset, collate_fn, collate_fn_eval, get_regime_dim, train_model
from data.pipeline import INDUSTRY_REL_FEATURES, N_AGGS, build_cross_section_dataset


class LogWriter:
    """Write-only log file, no pipe to stdout."""

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


def configure_windows_utf8():
    os.environ.setdefault("PYTHONUTF8", "1")


def build_config():
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
    return cfg


def prepare_data(cfg):
    print("\n构建数据集...")
    train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)

    regime_dim = get_regime_dim(cfg)
    for sample in train_samples + val_samples:
        sample['risk'] = sample['risk'][:, :regime_dim].copy()
    gc.collect()

    return train_samples, val_samples, regime_dim


def infer_num_industries(train_samples):
    all_ids = np.concatenate([s['industry_ids'] for s in train_samples])
    known_ids = all_ids[all_ids >= 0]
    return int(known_ids.max()) + 1 if known_ids.size else 1


def main():
    os.chdir(PROJECT_ROOT)
    configure_windows_utf8()

    log_file = open(PROJECT_ROOT / "train_gat.log", "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = LogWriter(log_file)
    sys.stderr = LogWriter(log_file)

    try:
        print("GAT training (V9 + industry graph attention)")
        print("=" * 60)
        print("改进:")
        print("  - FeatureGrouper -> Transformer（跨股票）")
        print("  - GATConv（行业子图消息传播）")
        print("  - FusionGate: 自适应融合Transformer+GAT")
        print("  - 行业embedding + rank embedding")
        print("=" * 60)

        cfg = build_config()
        print(f"配置: target_horizon={cfg.target_horizon}, seq_len={cfg.seq_len}, "
              f"horizons={cfg.horizon_indices}, weights={cfg.horizon_weights}, "
              f"market={cfg.use_market_features}, macro={cfg.use_macro_features}, "
              f"use_gat={cfg.use_gat}")

        train_samples, val_samples, regime_dim = prepare_data(cfg)

        input_dim = train_samples[0]["X"].shape[1]
        horizon = train_samples[0]["y_seq"].shape[1]
        print(f"Input dim: {input_dim}, Horizon labels: {horizon}")
        print(f"训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}")

        industry_rel_dim = len(INDUSTRY_REL_FEATURES)
        total_agg = (input_dim - industry_rel_dim) // 2
        base_feat_dim = total_agg // N_AGGS
        risk_dim = train_samples[0]["risk"].shape[1]
        num_industries = infer_num_industries(train_samples)
        print("\n特征维度:")
        print(f"  input_dim={input_dim}, base_feat_dim={base_feat_dim}, n_aggs={N_AGGS}")
        print(f"  regime_dim={regime_dim}, risk_dim={risk_dim}, industries={num_industries}")
        print(f"  GAT: {num_industries}个行业子图 + {base_feat_dim}个基础特征")

        train_ds = CrossSectionDataset(train_samples)
        val_ds = CrossSectionDataset(val_samples)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if gpu_mem < 8:
                batch_size, accum_steps = 8, 2
            else:
                batch_size, accum_steps = 4, 4
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

        try:
            train_model(
                train_loader, val_loader, input_dim, base_feat_dim, cfg,
                n_aggs=N_AGGS, n_alpha=4, n_horizons=len(cfg.horizon_indices),
                epochs=25, lr=3e-4, weight_decay=2e-3, accum_steps=accum_steps,
                grad_clip=0.2, use_amp=False, num_industries=num_industries,
                save_path=str(PROJECT_ROOT / "ultimate_v7_gat_best.pt"),
            )
            print("GAT训练完成")
        except Exception as e:
            print(f"\nFATAL ERROR: {e}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
