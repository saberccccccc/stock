from dataclasses import dataclass

from core.config import DataConfig, GAT_CKPT, V9_CKPT
from data.pipeline import build_cross_section_dataset, samples_from_precomputed_metadata
from backtest.engine import DLPredictor, load_price_volume, load_v9_checkpoint


@dataclass
class BacktestRuntime:
    cfg: DataConfig
    train: list
    val: list
    price_dict: dict
    vol_dict: dict


@dataclass
class V9GATPredictors:
    v9_predictor: DLPredictor
    gat_predictor: DLPredictor


def build_v9_backtest_config(
    target_horizon: int = 5,
    seq_len: int = 40,
    max_horizon: int = 10,
    min_stocks_per_time: int = 30,
) -> DataConfig:
    cfg = DataConfig()
    cfg.use_technical_features = True
    cfg.use_market_features = True
    cfg.use_macro_features = True
    cfg.use_fundamental_features = True
    cfg.use_shareholder_features = True
    cfg.use_restricted_features = True
    cfg.min_stocks_per_time = min_stocks_per_time
    cfg.target_horizon = target_horizon
    cfg.seq_len = seq_len
    cfg.max_horizon = max_horizon
    return cfg


def load_backtest_runtime(cfg: DataConfig | None = None, use_cache: bool = True) -> BacktestRuntime:
    cfg = build_v9_backtest_config() if cfg is None else cfg
    result = build_cross_section_dataset(cfg, use_cache=use_cache)
    if isinstance(result, dict):
        cfg.low_feat_dim = result.get('low_agg_dim', getattr(cfg, 'low_feat_dim', 14))
        train = samples_from_precomputed_metadata(result, 'train')
        val = samples_from_precomputed_metadata(result, 'val')
    else:
        train, val = result
    price_dict, vol_dict = load_price_volume(cfg)
    return BacktestRuntime(
        cfg=cfg,
        train=train,
        val=val,
        price_dict=price_dict,
        vol_dict=vol_dict,
    )


def load_dl_predictor(
    checkpoint_path: str,
    train_samples: list,
    cfg: DataConfig,
    device: str = "auto",
) -> DLPredictor:
    model, resolved_device, regime_dim = load_v9_checkpoint(checkpoint_path, train_samples, cfg, device)
    return DLPredictor(model, resolved_device, regime_dim)


def load_v9_gat_predictors(
    train_samples: list,
    cfg: DataConfig,
    v9_checkpoint: str = V9_CKPT,
    gat_checkpoint: str = GAT_CKPT,
    device: str = "auto",
) -> V9GATPredictors:
    v9_predictor = load_dl_predictor(v9_checkpoint, train_samples, cfg, device)
    gat_predictor = load_dl_predictor(gat_checkpoint, train_samples, cfg, device)
    return V9GATPredictors(v9_predictor=v9_predictor, gat_predictor=gat_predictor)
