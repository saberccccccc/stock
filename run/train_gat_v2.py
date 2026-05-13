"""GAT training (no Tee version, avoid pipe blocking)."""
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


def configure_windows_utf8():
    os.environ.setdefault("PYTHONUTF8", "1")


def log(msg, log_file):
    log_file.write(msg + "\n")
    log_file.flush()


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
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        log("GAT training (V9 + industry graph attention)", log_file)
        log("=" * 60, log_file)
        log("改进:", log_file)
        log("  - FeatureGrouper + Transformer (cross-stock attention)", log_file)
        log("  - GATConv industry subgraph message passing", log_file)
        log("  - FusionGate: 自适应融合Transformer+GAT", log_file)
        log("  - 行业embedding + rank embedding", log_file)
        log("=" * 60, log_file)

        cfg = build_config()
        log(f"配置: target_horizon={cfg.target_horizon}, seq_len={cfg.seq_len}, "
            f"horizons={cfg.horizon_indices}, weights={cfg.horizon_weights}, "
            f"market={cfg.use_market_features}, macro={cfg.use_macro_features}, "
            f"use_gat={cfg.use_gat}", log_file)

        log("\n构建数据集...", log_file)
        train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)

        regime_dim = get_regime_dim(cfg)
        for sample in train_samples + val_samples:
            sample['risk'] = sample['risk'][:, :regime_dim].copy()
        gc.collect()

        input_dim = train_samples[0]["X"].shape[1]
        horizon = train_samples[0]["y_seq"].shape[1]
        log(f"Input dim: {input_dim}, Horizon labels: {horizon}", log_file)
        log(f"训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}", log_file)

        industry_rel_dim = len(INDUSTRY_REL_FEATURES)
        total_agg = (input_dim - industry_rel_dim) // 2
        base_feat_dim = total_agg // N_AGGS
        risk_dim = train_samples[0]["risk"].shape[1]
        num_industries = infer_num_industries(train_samples)
        log(f"  input_dim={input_dim}, base_feat_dim={base_feat_dim}, n_aggs={N_AGGS}", log_file)
        log(f"  regime_dim={regime_dim}, risk_dim={risk_dim}, industries={num_industries}", log_file)
        log(f"  GAT: {num_industries}个行业子图 + {base_feat_dim}个基础特征", log_file)

        train_ds = CrossSectionDataset(train_samples)
        val_ds = CrossSectionDataset(val_samples)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            batch_size, accum_steps = 8, 2
        else:
            batch_size, accum_steps = 2, 8

        val_batch_size = 4
        log(f"Batch size: {batch_size} (val: {val_batch_size}), Accum steps: {accum_steps}", log_file)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b, keep_ratio=0.5, min_keep=30),
            num_workers=0, pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=val_batch_size, shuffle=False,
            collate_fn=collate_fn_eval, num_workers=0, pin_memory=False,
        )

        best_model_path = PROJECT_ROOT / "ultimate_v7_gat_best.pt"
        if best_model_path.exists():
            best_model_path.unlink()

        try:
            train_model(
                train_loader, val_loader, input_dim, base_feat_dim, cfg,
                n_aggs=N_AGGS, n_alpha=4, n_horizons=len(cfg.horizon_indices),
                epochs=25, lr=3e-4, weight_decay=2e-3, accum_steps=accum_steps,
                grad_clip=0.2, use_amp=False, resume=False, num_industries=num_industries,
                save_path=str(best_model_path),
            )
            log("GAT训练完成", log_file)
        except Exception as e:
            log(f"\nFATAL ERROR: {e}", log_file)
            traceback.print_exc(file=log_file)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
