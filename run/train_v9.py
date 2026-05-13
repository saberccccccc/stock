"""V9 training script: strong regularization version"""
import gc
import locale
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from core.config import DataConfig
from core.train_utils import CrossSectionDataset, collate_fn, collate_fn_eval, get_regime_dim, train_model
from data.market_features import N_MARKET
from data.pipeline import INDUSTRY_REL_FEATURES, N_AGGS, build_cross_section_dataset


class Tee:
    """Tee stdout/stderr to a log file. Suppresses tqdm \r progress lines from log."""

    def __init__(self, original, logfile):
        self.original = original
        self.logfile = logfile

    def write(self, obj):
        self.original.write(obj)
        if '\r' not in obj and not self.logfile.closed:
            self.logfile.write(obj)
            if '\n' in obj:
                self.logfile.flush()

    def flush(self):
        self.original.flush()
        if not self.logfile.closed:
            self.logfile.flush()


def configure_windows_utf8():
    os.environ.setdefault("PYTHONUTF8", "1")


def build_config():
    cfg = DataConfig()
    cfg.use_technical_features = True
    cfg.min_stocks_per_time = 30
    cfg.target_horizon = 5
    cfg.seq_len = 40
    cfg.max_horizon = 10
    cfg.use_market_features = True
    cfg.use_macro_features = True
    cfg.test_mode = False
    return cfg


def print_feature_summary(train_samples, input_dim, base_feat_dim, cfg):
    industry_rel_dim = len(INDUSTRY_REL_FEATURES)
    agg_dim = base_feat_dim * N_AGGS
    rank_dim = agg_dim
    print("\n" + "=" * 60)
    print("特征维度详情:")
    print("=" * 60)
    print(f"X特征 (input_dim={input_dim}):")
    print(f"  - 聚合特征: {agg_dim}维 ({base_feat_dim}个基础特征 × {N_AGGS}种聚合)")
    print(f"  - Rank特征: {rank_dim}维 (截面排序)")
    print(f"  - 行业相对: {industry_rel_dim}维")
    print(f"  - 合计: {agg_dim} + {rank_dim} + {industry_rel_dim} = {input_dim}维")

    risk_dim = train_samples[0]["risk"].shape[1]
    stock_risk_dim = 3
    market_feat_dim = N_MARKET
    macro_feat_dim = 3 if cfg.use_macro_features else 0
    regime_dim = get_regime_dim(cfg)
    industry_onehot_dim = risk_dim - regime_dim
    print(f"\nRisk特征 (total={risk_dim}):")
    print(f"  - 股票级风险 {stock_risk_dim}维 (size, vol, mom)")
    print(f"  - 市场特征: {market_feat_dim}维 (16宽基+3宽度+31行业收益+31行业可用性mask)")
    print(f"  - 宏观/资金流 {macro_feat_dim}维 (north, margin, PMI)")
    print(f"  - 行业one-hot: {industry_onehot_dim}维")
    print(f"  - 行业embedding: {industry_onehot_dim}个真实行业 + 1个未知行业")
    print(f"  - regime输入: {regime_dim}维")
    print("=" * 60 + "\n")
    return industry_onehot_dim


def main():
    os.chdir(PROJECT_ROOT)
    configure_windows_utf8()

    log_file = open(PROJECT_ROOT / "train_v9.log", "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        print("=" * 60)
        print("V9 训练启动（强正则化版本）")
        print("=" * 60)
        print("V9 vs V8 改进:")
        print("  - Transformer dropout: 0.3 -> 0.35")
        print("  - weight_decay: 5e-4 -> 2e-3 (4x)")
        print("  - gradient clip: 0.3 -> 0.2")
        print("  - resume compatible checkpoint by default")
        print("=" * 60)

        cfg = build_config()
        print(f"配置: target_horizon={cfg.target_horizon}, seq_len={cfg.seq_len}, "
              f"horizons={cfg.horizon_indices}, weights={cfg.horizon_weights}, "
              f"market={cfg.use_market_features}, macro={cfg.use_macro_features}")

        print("\n构建数据集...")
        train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)

        input_dim = train_samples[0]["X"].shape[1]
        horizon = train_samples[0]["y_seq"].shape[1]
        print(f"Input dim: {input_dim}, Horizon labels: {horizon}")
        print(f"训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}")

        industry_rel_dim = len(INDUSTRY_REL_FEATURES)
        total_agg = (input_dim - industry_rel_dim) // 2
        base_feat_dim = total_agg // N_AGGS
        print(f"base_feat_dim={base_feat_dim}, n_aggs={N_AGGS}")
        num_industries = print_feature_summary(train_samples, input_dim, base_feat_dim, cfg)

        train_ds = CrossSectionDataset(train_samples)
        val_ds = CrossSectionDataset(val_samples)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if gpu_mem < 8:
                batch_size, accum_steps = 2, 8
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

        try:
            train_model(
                train_loader, val_loader, input_dim, base_feat_dim, cfg,
                n_aggs=N_AGGS, n_alpha=4, n_horizons=len(cfg.horizon_indices),
                epochs=25, lr=3e-4, weight_decay=2e-3, accum_steps=accum_steps,
                grad_clip=0.2, use_amp=False, num_industries=num_industries,
                save_path=str(PROJECT_ROOT / "ultimate_v7_best.pt"),
            )
            print("训练完成")
        except Exception as e:
            print(f"\nFATAL ERROR: {e}")
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem_info = (
                    torch.cuda.memory_allocated() / 1024 ** 3,
                    torch.cuda.memory_reserved() / 1024 ** 3,
                )
                print(f"GPU mem after crash: alloc={mem_info[0]:.2f}GB, cache={mem_info[1]:.2f}GB")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
