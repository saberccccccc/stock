# hyper_search.py - 超参数网格搜索
import argparse
import os
import sys
import itertools
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

from core.config import DataConfig
from data.pipeline import build_cross_section_dataset
from backtest.engine import (
    LGBPredictor,
    run_backtest_production,
    calc_metrics,
    load_price_volume,
    train_multi_horizon_models,
    load_multi_horizon_models,
    models_match_feature_dim,
)


PARAM_GRID = {
    'target_vol': [0.06, 0.08, 0.10, 0.15],
    'lambda_t': [0.02, 0.05, 0.1, 0.2],
    'lambda_b': [0.1, 0.2, 0.5, 1.0],
    'max_weight': [0.03, 0.05, 0.08],
    'adv_limit_ratio': [0.01, 0.02, 0.05],
}

FIXED_PARAMS = {
    'future_len': 5,
    'hist_window': 60,
    'ewma_hl': 20,
    'impact_coeff': 0.1,
}


def load_data_and_models():
    """加载数据并训练/加载模型（只执行一次）"""
    cfg = DataConfig()
    cfg.use_technical_features = True
    cfg.use_macro_features = True
    cfg.min_stocks_per_time = 30
    cfg.target_horizon = 5
    cfg.seq_len = 40
    cfg.max_horizon = 10

    print("构建截面数据集...")
    train, val = build_cross_section_dataset(cfg, use_cache=True)

    horizon_list = [1, 3, 5, 10]
    model_dir = "models_multi_v9_tech_macro"
    expected_dim = train[0]['X'].shape[1]
    if not os.path.exists(f"{model_dir}/lgb_h1.txt"):
        print("训练多周期模型...")
        models, ic_decay = train_multi_horizon_models(train, val, horizon_list, model_dir)
    else:
        print("加载已有模型...")
        models, ic_decay = load_multi_horizon_models(horizon_list, model_dir)
        if not models_match_feature_dim(models, expected_dim):
            print("已有LightGBM模型特征维度不匹配，重新训练...")
            models, ic_decay = train_multi_horizon_models(train, val, horizon_list, model_dir)

    print("加载价格与成交量数据...")
    price_dict, vol_dict = load_price_volume(cfg)
    print(f"股票数 {len(price_dict)}")

    return cfg, models, ic_decay, val, price_dict, vol_dict


def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM/backtest hyper-parameter search")
    parser.add_argument("--output", default=None, help="CSV output path. Defaults to timestamped file in project root.")
    return parser.parse_args()


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    combinations = list(itertools.product(*values))
    print(f"total {len(combinations)} param combinations")

    cfg, models, ic_decay, val_samples, price_dict, vol_dict = load_data_and_models()
    fixed_params = FIXED_PARAMS.copy()
    fixed_params['future_len'] = cfg.target_horizon
    predictor = LGBPredictor(models, ic_decay)

    results = []
    for comb in tqdm(combinations, desc="参数扫描"):
        params = fixed_params.copy()
        for k, v in zip(keys, comb):
            params[k] = v

        try:
            raw_ret, neu_ret, _ = run_backtest_production(
                predictor, val_samples, price_dict, vol_dict,
                future_len=params['future_len'],
                rebalance_freq=None,
                hist_window=params['hist_window'],
                ewma_hl=params['ewma_hl'],
                adv_limit_ratio=params['adv_limit_ratio'],
                max_weight=params['max_weight'],
                lambda_t=params['lambda_t'],
                lambda_b=params['lambda_b'],
                target_vol=params['target_vol'],
                impact_coeff=params['impact_coeff'],
                config=cfg,
            )
            ann, sharpe, mdd = calc_metrics(raw_ret)
            ann_neu, sharpe_neu, mdd_neu = calc_metrics(neu_ret)
        except Exception as e:
            print(f"参数 {params} 出错: {e}")
            ann = sharpe = mdd = ann_neu = sharpe_neu = mdd_neu = np.nan

        results.append({
            **params,
            'raw_ann': ann, 'raw_sharpe': sharpe, 'raw_mdd': mdd,
            'neu_ann': ann_neu, 'neu_sharpe': sharpe_neu, 'neu_mdd': mdd_neu,
        })

    df = pd.DataFrame(results)
    output_path = Path(args.output) if args.output else PROJECT_ROOT / f"param_search_results_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(output_path, index=False)
    print(f"参数搜索结果已保存: {output_path}")

    best_raw = df.loc[df['raw_sharpe'].idxmax()]
    best_neu = df.loc[df['neu_sharpe'].idxmax()]

    print("\n========== 最优参数（原始多空夏普）=========")
    cols = ['target_vol', 'lambda_t', 'lambda_b', 'max_weight', 'adv_limit_ratio',
            'raw_sharpe', 'raw_ann', 'raw_mdd']
    print(best_raw[cols])
    print("\n========== 最优参数（修正中性夏普）==========")
    cols_neu = ['target_vol', 'lambda_t', 'lambda_b', 'max_weight', 'adv_limit_ratio',
                'neu_sharpe', 'neu_ann', 'neu_mdd']
    print(best_neu[cols_neu])


if __name__ == "__main__":
    main()
