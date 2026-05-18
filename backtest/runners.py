from dataclasses import dataclass
from typing import Optional

from core.config import DataConfig
from backtest.engine import calc_extended_metrics, calc_metrics, run_backtest_production, save_backtest_results
from backtest.layered_engine import run_backtest_layered_production


@dataclass
class ProductionBacktestParams:
    future_len: int
    rebalance_freq: Optional[int] = None
    hist_window: int = 60
    ewma_hl: int = 20
    adv_limit_ratio: float = 0.02
    adv_mode: str = "execution"
    portfolio_mode: str = "simple_ls"
    top_frac: float = 0.10
    max_weight: float = 0.05
    lambda_t: float = 0.05
    lambda_b: float = 0.2
    target_vol: float = 0.15
    impact_coeff: float = 0.1
    optimizer_base_mode: str = "simple_ls"
    optimizer_exposure_control: str = "beta"
    optimizer_beta_limit: float = 0.05
    optimizer_dollar_neutral: bool = True
    mvo_risk_aversion: float = 1.0
    mvo_lr: float = 0.02
    mvo_n_iter: int = 200


@dataclass
class LayeredBacktestParams:
    holding_days: int
    rebalance_freq: int = 1
    hist_window: int = 60
    ewma_hl: int = 20
    adv_limit_ratio: float = 0.02
    adv_mode: str = "execution"
    portfolio_mode: str = "optimizer_projected"
    top_frac: float = 0.10
    max_weight: float = 0.05
    lambda_t: float = 0.05
    lambda_b: float = 0.2
    target_vol: float = 0.15
    impact_coeff: float = 0.1
    optimizer_base_mode: str = "simple_ls"
    optimizer_exposure_control: str = "beta"
    optimizer_beta_limit: float = 0.05
    optimizer_dollar_neutral: bool = True
    mvo_risk_aversion: float = 1.0
    mvo_lr: float = 0.02
    mvo_n_iter: int = 200
    layer_scale: str = "auto"


def metrics_from_returns(raw_ret, neu_ret) -> dict:
    ann_raw, sharpe_raw, mdd_raw = calc_metrics(raw_ret)
    ann_neu, sharpe_neu, mdd_neu = calc_metrics(neu_ret)
    return {
        "ann_raw": ann_raw,
        "sharpe_raw": sharpe_raw,
        "mdd_raw": mdd_raw,
        "ann_neu": ann_neu,
        "sharpe_neu": sharpe_neu,
        "mdd_neu": mdd_neu,
    }


def _print_extended_metrics(raw_ret, neu_ret) -> None:
    ext_raw = calc_extended_metrics(raw_ret)
    ext_neu = calc_extended_metrics(neu_ret)
    print("\n========== 扩展指标 ==========")
    print("原始多空:")
    print(f"  Calmar比率: {ext_raw.get('calmar', 0):.3f}")
    print(f"  Sortino比率: {ext_raw.get('sortino', 0):.3f}")
    print(f"  胜率: {ext_raw.get('win_rate', 0):.2f}%")
    print(f"  盈亏比 {ext_raw.get('profit_loss_ratio', 0):.3f}")
    print("修正中性")
    print(f"  Calmar比率: {ext_neu.get('calmar', 0):.3f}")
    print(f"  Sortino比率: {ext_neu.get('sortino', 0):.3f}")
    print(f"  胜率: {ext_neu.get('win_rate', 0):.2f}%")
    print(f"  盈亏比 {ext_neu.get('profit_loss_ratio', 0):.3f}")


def run_production_backtest_once(
    predictor,
    val_samples: list,
    price_dict: dict,
    vol_dict: dict,
    cfg: DataConfig,
    params: ProductionBacktestParams,
    label: str,
    output_dir: str,
    save_results: bool = True,
    extra_fields: dict | None = None,
) -> tuple[dict, dict]:
    if hasattr(predictor, "reset_stats"):
        predictor.reset_stats()

    print(f"\n=== {predictor.name} / {label} / run_top_frac={params.top_frac:.2f} ===")
    raw_ret, neu_ret, backtest_data = run_backtest_production(
        predictor,
        val_samples,
        price_dict,
        vol_dict,
        future_len=params.future_len,
        rebalance_freq=params.rebalance_freq,
        hist_window=params.hist_window,
        ewma_hl=params.ewma_hl,
        adv_limit_ratio=params.adv_limit_ratio,
        adv_mode=params.adv_mode,
        portfolio_mode=params.portfolio_mode,
        top_frac=params.top_frac,
        max_weight=params.max_weight,
        lambda_t=params.lambda_t,
        lambda_b=params.lambda_b,
        target_vol=params.target_vol,
        impact_coeff=params.impact_coeff,
        config=cfg,
        optimizer_base_mode=params.optimizer_base_mode,
        optimizer_exposure_control=params.optimizer_exposure_control,
        optimizer_beta_limit=params.optimizer_beta_limit,
        optimizer_dollar_neutral=params.optimizer_dollar_neutral,
        mvo_risk_aversion=params.mvo_risk_aversion,
        mvo_lr=params.mvo_lr,
        mvo_n_iter=params.mvo_n_iter,
    )

    metrics = metrics_from_returns(raw_ret, neu_ret)
    if save_results:
        save_backtest_results(
            backtest_data,
            metrics,
            predictor.name,
            label,
            output_dir=output_dir,
        )

    stats = predictor.stats() if hasattr(predictor, "stats") else {}
    print(f"{predictor.name}/{label}: raw ann={metrics['ann_raw']:.2f}% sharpe={metrics['sharpe_raw']:.2f} mdd={metrics['mdd_raw']*100:.2f}%")
    print(f"{predictor.name}/{label}: neu ann={metrics['ann_neu']:.2f}% sharpe={metrics['sharpe_neu']:.2f} mdd={metrics['mdd_neu']*100:.2f}%")
    if stats:
        print("Signal stats:", {k: round(v, 4) for k, v in stats.items()})

    summary_row = {"mode": label, **metrics, **stats}
    if extra_fields:
        summary_row = {**extra_fields, **summary_row}
    return summary_row, backtest_data


def run_layered_backtest_once(
    predictor,
    val_samples: list,
    price_dict: dict,
    vol_dict: dict,
    cfg: DataConfig,
    params: LayeredBacktestParams,
    label: str,
    output_dir: str,
    save_results: bool = True,
    extra_fields: dict | None = None,
    print_extended: bool = False,
) -> tuple[dict, dict]:
    if hasattr(predictor, "reset_stats"):
        predictor.reset_stats()

    print(f"\n=== {predictor.name} / {label} / run_top_frac={params.top_frac:.2f} ===")
    raw_ret, neu_ret, backtest_data = run_backtest_layered_production(
        predictor,
        val_samples,
        price_dict,
        vol_dict,
        holding_days=params.holding_days,
        rebalance_freq=params.rebalance_freq,
        hist_window=params.hist_window,
        ewma_hl=params.ewma_hl,
        adv_limit_ratio=params.adv_limit_ratio,
        adv_mode=params.adv_mode,
        portfolio_mode=params.portfolio_mode,
        top_frac=params.top_frac,
        max_weight=params.max_weight,
        lambda_t=params.lambda_t,
        lambda_b=params.lambda_b,
        target_vol=params.target_vol,
        impact_coeff=params.impact_coeff,
        config=cfg,
        optimizer_base_mode=params.optimizer_base_mode,
        optimizer_exposure_control=params.optimizer_exposure_control,
        optimizer_beta_limit=params.optimizer_beta_limit,
        optimizer_dollar_neutral=params.optimizer_dollar_neutral,
        mvo_risk_aversion=params.mvo_risk_aversion,
        mvo_lr=params.mvo_lr,
        mvo_n_iter=params.mvo_n_iter,
        layer_scale=params.layer_scale,
    )

    metrics = metrics_from_returns(raw_ret, neu_ret)
    if print_extended:
        _print_extended_metrics(raw_ret, neu_ret)
    if save_results:
        save_backtest_results(
            backtest_data,
            metrics,
            predictor.name,
            label,
            output_dir=output_dir,
        )

    stats = predictor.stats() if hasattr(predictor, "stats") else {}
    print(f"{predictor.name}/{label}: raw ann={metrics['ann_raw']:.2f}% sharpe={metrics['sharpe_raw']:.2f} mdd={metrics['mdd_raw']*100:.2f}%")
    print(f"{predictor.name}/{label}: neu ann={metrics['ann_neu']:.2f}% sharpe={metrics['sharpe_neu']:.2f} mdd={metrics['mdd_neu']*100:.2f}%")
    if stats:
        print("Signal stats:", {k: round(v, 4) for k, v in stats.items()})

    summary_row = {"mode": label, **metrics, **stats}
    if extra_fields:
        summary_row = {**extra_fields, **summary_row}
    return summary_row, backtest_data
