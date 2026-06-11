"""Unified training entry: python run/train.py --model {v9,gat,gat_v2,legacy}"""
import argparse
import gc
import os
import shutil
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
from core.train_utils import (CrossSectionDataset, MemmapDataset,
                               PrecomputedMemmapDataset, collate_fn,
                               collate_fn_eval, get_regime_dim, train_model)
from data.market_features import N_MARKET
from data.pipeline import (INDUSTRY_REL_FEATURES, N_AGGS, build_cross_section_dataset,
                           _open_memmap)

# ── Logging utilities ─────────────────────────────────────
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


def infer_num_industries(train_samples):
    all_ids = np.concatenate([s['industry_ids'] for s in train_samples])
    known_ids = all_ids[all_ids >= 0]
    return int(known_ids.max()) + 1 if known_ids.size else 1


# ── Model dispatch tables ─────────────────────────────────
MODEL_CONFIGS = {
    "v9": {
        "use_gat": False, "keep_ratio": None, "resume": True,
        "ckpt_suffix": "best", "log_style": "tee", "risk_trim": False,
        "header": "V9 训练（exp-004）",
        "v9_banner": True,
    },
    "gat": {
        "use_gat": True, "keep_ratio": None, "resume": True,
        "ckpt_suffix": "gat_best", "log_style": "writer", "risk_trim": True,
        "header": "GAT training (V9 + industry graph attention)",
    },
    "gat_v2": {
        "use_gat": True, "keep_ratio": 0.5, "resume": False,
        "ckpt_suffix": "gat_v2_best", "log_style": "raw", "risk_trim": True,
        "backup_ckpt": True,
        "header": "GAT v2 training (V9 + GAT + experimental)",
    },
    "legacy": {
        "use_gat": False, "keep_ratio": None, "resume": True,
        "ckpt_suffix": "legacy_best", "log_style": "tee", "risk_trim": False,
        "no_utf8_fix": True,
        "header": "Legacy V9 训练启动（含市场整体属性 + 增强Alpha头）",
    },
}

BATCH_CONFIGS = {
    "v9":     {"lt8": (4, 4), "ge8": (8, 2), "cpu": (4, 4), "val_lt8": 4},
    "gat":    {"lt8": (2, 8), "ge8": (8, 2), "cpu": (2, 8), "val_lt8": 2},
    "gat_v2": {"lt8": (8, 2), "ge8": (8, 2), "cpu": (2, 8), "val_lt8": 4},
    "legacy": {"lt8": (2, 4), "ge8": (4, 4), "cpu": (4, 4), "val_lt8": 1},
}


# ── CLI ───────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Unified training entry for V9/GAT/Legacy")
    parser.add_argument("--model", choices=["v9", "gat", "gat_v2", "legacy"], required=True)
    parser.add_argument("--test-stocks", type=int, default=None, help="Limit stocks for smoke test")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--data-dir", default=None, help="Override data directory (e.g. ../deepseek_optimized/data/raw)")
    parser.add_argument("--top-focus-loss-weight", type=float, default=None, help="Long-only top-focus auxiliary loss weight.")
    parser.add_argument("--top-focus-temperature", type=float, default=None, help="Softmax temperature for top-focus loss.")
    parser.add_argument("--top-focus-delay-epochs", type=int, default=None, help="Epoch delay before enabling top-focus loss.")
    parser.add_argument("--pairwise-top-loss-weight", type=float, default=None, help="Top-area pairwise ranking loss weight.")
    parser.add_argument("--pairwise-top-frac", type=float, default=None, help="Top fraction used for pairwise sampling.")
    parser.add_argument("--pairwise-num-pairs", type=int, default=None, help="Sampled pairs per cross-section and horizon.")
    parser.add_argument("--pairwise-model-top-weight", type=float, default=None, help="Weight for pairs sampled inside model top bucket.")
    parser.add_argument("--pairwise-delay-epochs", type=int, default=None, help="Epoch delay before enabling pairwise loss.")
    parser.add_argument("--best-val-metric", default=None, help="Validation metric used for checkpoint selection.")
    parser.add_argument("--eval-top-fracs", default=None, help="Comma-separated top fractions for validation metrics, e.g. 0.05,0.10.")
    return parser.parse_args()


# ── Core helpers ──────────────────────────────────────────
def build_config(mc, data_dir=None, args=None):
    cfg = DataConfig()
    if data_dir:
        cfg.data_dir = data_dir
    cfg.use_technical_features = True
    cfg.min_stocks_per_time = 30
    cfg.target_horizon = 5
    cfg.seq_len = 40
    cfg.max_horizon = 10
    cfg.use_market_features = True
    cfg.use_macro_features = True
    cfg.use_fundamental_features = True
    cfg.use_shareholder_features = True
    cfg.use_restricted_features = True
    cfg.test_mode = False
    if args is not None:
        if args.top_focus_loss_weight is not None:
            cfg.top_focus_loss_weight = args.top_focus_loss_weight
        if args.top_focus_temperature is not None:
            cfg.top_focus_temperature = args.top_focus_temperature
        if args.top_focus_delay_epochs is not None:
            cfg.top_focus_delay_epochs = args.top_focus_delay_epochs
        if args.pairwise_top_loss_weight is not None:
            cfg.pairwise_top_loss_weight = args.pairwise_top_loss_weight
        if args.pairwise_top_frac is not None:
            cfg.pairwise_top_frac = args.pairwise_top_frac
        if args.pairwise_num_pairs is not None:
            cfg.pairwise_num_pairs = args.pairwise_num_pairs
        if args.pairwise_model_top_weight is not None:
            cfg.pairwise_model_top_weight = args.pairwise_model_top_weight
        if args.pairwise_delay_epochs is not None:
            cfg.pairwise_delay_epochs = args.pairwise_delay_epochs
        if args.best_val_metric is not None:
            cfg.best_val_metric = args.best_val_metric
        if args.eval_top_fracs is not None:
            cfg.eval_top_fracs = tuple(float(x.strip()) for x in args.eval_top_fracs.split(",") if x.strip())
    if mc["use_gat"]:
        cfg.use_gat = True
    return cfg


def resolve_batch(model, device):
    bc = BATCH_CONFIGS[model]
    if device.type == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if gpu_mem < 8:
            return bc["lt8"][0], bc["lt8"][1], bc["val_lt8"], gpu_mem
        else:
            return bc["ge8"][0], bc["ge8"][1], bc["ge8"][0], gpu_mem
    return bc["cpu"][0], bc["cpu"][1], bc["cpu"][0], 99


def setup_logging(model, mc):
    log_name = f"train_{'gat' if model in ('gat', 'gat_v2') else model}.log"
    log_file = open(PROJECT_ROOT / log_name, "w", encoding="utf-8")
    style = mc["log_style"]
    if style == "tee":
        return log_file, Tee(sys.stdout, log_file), Tee(sys.stderr, log_file)
    if style == "writer":
        return log_file, LogWriter(log_file), LogWriter(log_file)
    # raw: direct file handle replacement (gat_v2)
    return log_file, log_file, log_file


def log_print(*args, **kwargs):
    print(*args, **kwargs)


def print_banner(mc):
    print("=" * 60)
    print(mc["header"])
    print("=" * 60)


def print_feature_summary(train_samples, input_dim, base_feat_dim, cfg):
    industry_rel_dim = len(INDUSTRY_REL_FEATURES)
    agg_dim = base_feat_dim * N_AGGS
    rank_dim = agg_dim
    print("\n" + "=" * 60)
    print("特征维度详情:")
    print("=" * 60)
    print(f"X特征 (input_dim={input_dim}):")
    print(f"  - 聚合特征: {agg_dim}维 ({base_feat_dim}个基础特征 × {N_AGGS}种聚合)  [旧路径参考]")
    print(f"  - Rank特征: {rank_dim}维 (截面排序)")
    print(f"  - 行业相对: {industry_rel_dim}维")
    print(f"  - 合计: {agg_dim} + {rank_dim} + {industry_rel_dim} = {input_dim}维")

    risk_dim = train_samples[0]["risk"].shape[1]
    stock_risk_dim = 6
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


def handle_error(model, exc):
    print(f"\nFATAL ERROR: {exc}")
    if model in ("v9", "legacy"):
        traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            print(f"GPU mem after crash: alloc={alloc:.2f}GB, cache={reserved:.2f}GB")
    elif model in ("gat",):
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:  # gat_v2
        traceback.print_exc(file=sys.stderr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train(args):
    model = args.model
    mc = MODEL_CONFIGS[model]

    # ── setup ──
    os.chdir(PROJECT_ROOT)
    if not mc.get("no_utf8_fix"):
        configure_windows_utf8()

    log_file, new_out, new_err = setup_logging(model, mc)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = new_out, new_err

    try:
        print_banner(mc)

        cfg = build_config(mc, data_dir=args.data_dir, args=args)
        if args.test_stocks is not None:
            cfg.test_mode = True
            cfg.test_stocks = args.test_stocks
            cfg.max_stocks = args.test_stocks
            cfg.min_stocks_per_time = max(10, min(cfg.min_stocks_per_time, args.test_stocks // 2))

        print(f"数据: target_horizon={cfg.target_horizon}, seq_len={cfg.seq_len}, "
              f"horizons={cfg.horizon_indices}, weights={cfg.horizon_weights}, "
              f"market={cfg.use_market_features}, macro={cfg.use_macro_features}"
              + (f", use_gat={cfg.use_gat}" if mc["use_gat"] else ""))
        print(f"模型: Transformer {cfg.n_transformer_layers}层, dropout={cfg.transformer_dropout}, "
              f"weight_decay=2e-3, grad_clip=0.2")
        print(f"训练: LR={args.lr}, warmup={cfg.lr_warmup_epochs}epoch, "
              f"AdamW fused={cfg.use_fused_adam}, cleanup_cache={cfg.cleanup_cache_interval}")
        print(f"损失: industry={cfg.industry_loss_weight}, spread={cfg.spread_loss_weight}"
              f"@d{cfg.spread_delay_epochs}, top_focus={cfg.top_focus_loss_weight}"
              f"@T{cfg.top_focus_temperature},d{cfg.top_focus_delay_epochs}, "
              f"pairwise={cfg.pairwise_top_loss_weight}@top{cfg.pairwise_top_frac},"
              f"pairs{cfg.pairwise_num_pairs},d{cfg.pairwise_delay_epochs}")
        print(f"验证: best_val_metric={cfg.best_val_metric}, eval_top_fracs={cfg.eval_top_fracs}")

        print("\n构建数据集...")
        result = build_cross_section_dataset(cfg, use_cache=True)

        # 判断返回类型：dict=memmap元数据（含预计算截面），tuple=样本列表（旧式）
        if isinstance(result, dict):
            meta = result

            # 打开预计算截面 memmap（int16 scale=1000 → 训练时自动转 float32/1000）
            n_stocks = len(meta['all_codes'])
            n_dates = len(meta['all_dates'])
            x_norm_mm = _open_memmap(meta['x_norm_path'], np.int16,
                                     (n_stocks, n_dates, meta['x_dim']))
            risk_full_mm = _open_memmap(meta['risk_full_path'], np.int16,
                                        (n_stocks, n_dates, meta['risk_full_dim']))
            y_norm_mm = _open_memmap(meta['y_norm_path'], np.int16,
                                     (n_stocks, n_dates))
            y_seq_norm_mm = _open_memmap(meta['y_seq_norm_path'], np.int16,
                                         (n_stocks, n_dates, meta['max_horizon']))

            train_ds = PrecomputedMemmapDataset(
                x_norm_mm, risk_full_mm, y_norm_mm, y_seq_norm_mm,
                meta['industry_array'], meta['all_codes'], meta['all_dates'],
                meta['train_indices'], meta['n_industries'], meta['max_horizon'],
            )
            val_ds = PrecomputedMemmapDataset(
                x_norm_mm, risk_full_mm, y_norm_mm, y_seq_norm_mm,
                meta['industry_array'], meta['all_codes'], meta['all_dates'],
                meta['val_indices'], meta['n_industries'], meta['max_horizon'],
            )

            # 从元数据推算维度
            low_feat_dim = meta['low_agg_dim']
            input_dim = meta['x_dim']
            horizon = meta['max_horizon']
            regime_dim = get_regime_dim(cfg)
            num_industries = meta['n_industries']
            risk_dim = meta['risk_full_dim']
            base_feat_dim = meta['high_feat_dim']

            print(f"Input dim: {input_dim}, Horizon labels: {horizon}")
            print(f"训练截面: {len(train_ds)}/{len(meta['train_indices'])} (过滤后/总数), "
                  f"验证截面: {len(val_ds)}/{len(meta['val_indices'])}")
            print(f"  (预计算模式: X_norm={input_dim}维, risk_full={risk_dim}维)")
            if not mc["use_gat"]:
                print(f"\n{'='*60}")
                print("特征维度详情:")
                print(f"{'='*60}")
                print(f"X特征 (input_dim={input_dim}):")
                total_agg = meta['high_agg_dim'] + meta['low_agg_dim']
                high_feat_n = meta['high_feat_dim']  # 23
                low_feat_n = meta['low_agg_dim'] // 2  # 7 (last+qoq)
                print(f"  - 聚合特征: {total_agg}维 ({high_feat_n}个高频×{N_AGGS}种 + {low_feat_n}个低频×2种)")
                print(f"  - Rank特征: {meta['high_agg_dim']}维 (截面排序)")
                print(f"  - 行业相对: {len(INDUSTRY_REL_FEATURES)}维")
                macro_dim = 3 if cfg.use_macro_features else 0
                print(f"Risk: stock(6)+market({N_MARKET})+macro({macro_dim})={risk_dim}")
                print(f"{'='*60}\n")
        else:
            train_samples, val_samples = result
            low_feat_dim = 14  # 旧路径兼容

            # GAT: trim risk to regime_dim only
            if mc["risk_trim"]:
                regime_dim = get_regime_dim(cfg)
                for sample in train_samples + val_samples:
                    sample['risk'] = sample['risk'][:, :regime_dim].copy()
                gc.collect()

            input_dim = train_samples[0]["X"].shape[1]
            horizon = train_samples[0]["y_seq"].shape[1]
            print(f"Input dim: {input_dim}, Horizon labels: {horizon}")
            print(f"训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}")

            industry_rel_dim = len(INDUSTRY_REL_FEATURES)
            total_agg = (input_dim - industry_rel_dim) // 2
            base_feat_dim = total_agg // N_AGGS

            if mc["use_gat"]:
                num_industries = infer_num_industries(train_samples)
                risk_dim = train_samples[0]["risk"].shape[1]
                regime_dim = get_regime_dim(cfg)
                print(f"\n特征维度:")
                print(f"  input_dim={input_dim}, base_feat_dim={base_feat_dim}, n_aggs={N_AGGS}")
                print(f"  regime_dim={regime_dim}, risk_dim={risk_dim}, industries={num_industries}")
                print(f"  GAT: {num_industries}个行业子图 + {base_feat_dim}个基础特征")
            else:
                num_industries = print_feature_summary(train_samples, input_dim, base_feat_dim, cfg)

            train_ds = CrossSectionDataset(train_samples)
            val_ds = CrossSectionDataset(val_samples)

        # GPU / batch
        if args.device == "cpu":
            device = torch.device("cpu")
        elif args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("请求使用CUDA，但当前PyTorch不可用CUDA")
            device = torch.device("cuda")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size, accum_steps, val_bs, _gpu_mem = resolve_batch(model, device)
        print(f"Batch size: {batch_size} (val: {val_bs}), Accum steps: {accum_steps}")

        # collate
        if mc["keep_ratio"] is not None:
            train_collate = lambda b: collate_fn(b, keep_ratio=mc["keep_ratio"], min_keep=30)
        else:
            train_collate = collate_fn_eval

        # num_workers>0 会导致 memmap 文件描述符跨进程失败，暂时用单进程
        n_workers = 0
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=train_collate, num_workers=n_workers, pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=val_bs, shuffle=False,
            collate_fn=collate_fn_eval, num_workers=n_workers, pin_memory=False,
        )

        cfg.low_feat_dim = low_feat_dim

        # checkpoint
        ckpt_dir = PROJECT_ROOT / (args.output_dir or "checkpoints_exp")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(ckpt_dir / f"ultimate_v7_{mc['ckpt_suffix']}.pt")

        # gat_v2: backup + delete existing checkpoint before training
        if mc.get("backup_ckpt"):
            ckpt = Path(ckpt_path)
            if ckpt.exists():
                shutil.copy2(ckpt, ckpt.with_suffix(".pt.bak"))
                ckpt.unlink()
        try:
            train_model(
                train_loader, val_loader, input_dim, cfg,
                n_alpha=4, n_horizons=len(cfg.horizon_indices),
                epochs=args.epochs, lr=args.lr, weight_decay=2e-3, accum_steps=accum_steps,
                grad_clip=0.2, use_amp=False, num_industries=num_industries,
                resume=mc["resume"],
                save_path=ckpt_path,
                patience=getattr(cfg, 'early_stop_patience', 5),
                device=device,
            )
            print("训练完成")

        except Exception as e:
            handle_error(model, e)
            with open(PROJECT_ROOT / "errors.log", "a", encoding="utf-8") as ef:
                ef.write(f"[{__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}] train.py/{model} | {type(e).__name__}: {e}\n")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    train(parse_args())
