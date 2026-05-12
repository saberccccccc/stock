# hyper_search.py 鈥?瓒呭弬鏁扮綉鏍兼悳绱?
import os
import sys
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import lightgbm as lgb

from core.config import DataConfig
from data.pipeline import build_cross_section_dataset
from backtest.engine import (
    run_backtest_production, calc_metrics, load_price_volume,
    train_multi_horizon_models, load_multi_horizon_models, models_match_feature_dim,
)


def load_data_and_models():
    """鍔犺浇鏁版嵁骞惰?缁?鍔犺浇妯″瀷锛堝彧鎵ц?涓€娆★級"""
    cfg = DataConfig()
    cfg.use_technical_features = True
    cfg.use_macro_features = True
    cfg.min_stocks_per_time = 30
    cfg.target_horizon = 5
    cfg.seq_len = 40
    cfg.max_horizon = 10

    print("鏋勫缓鎴?潰鏁版嵁闆?..")
    train, val = build_cross_section_dataset(cfg, use_cache=True)

    horizon_list = [1, 3, 5, 10]
    model_dir = "models_multi_v9_tech_macro"
    expected_dim = train[0]['X'].shape[1]
    if not os.path.exists(f"{model_dir}/lgb_h1.txt"):
        print("璁?粌澶氬懆鏈熸ā鍨?..")
        models, ic_decay = train_multi_horizon_models(train, val, horizon_list, model_dir)
    else:
        print("鍔犺浇宸叉湁妯″瀷...")
        models, ic_decay = load_multi_horizon_models(horizon_list, model_dir)
        if not models_match_feature_dim(models, expected_dim):
            print("宸叉湁LightGBM妯″瀷鐗瑰緛缁村害涓嶅尮閰嶏紝閲嶆柊璁?粌...")
            models, ic_decay = train_multi_horizon_models(train, val, horizon_list, model_dir)

    print("鍔犺浇浠锋牸涓庢垚浜ら噺鏁版嵁...")
    price_dict, vol_dict = load_price_volume(cfg)
    print(f"鑲＄エ鏁? {len(price_dict)}")

    return cfg, models, ic_decay, val, price_dict, vol_dict


# ==================== 鍙傛暟缃戞牸 ====================
param_grid = {
    'target_vol': [0.06, 0.08, 0.10, 0.15],
    'lambda_t': [0.02, 0.05, 0.1, 0.2],
    'lambda_b': [0.1, 0.2, 0.5, 1.0],
    'max_weight': [0.03, 0.05, 0.08],
    'adv_limit_ratio': [0.01, 0.02, 0.05],
}

fixed_params = {
    'future_len': 5,
    'hist_window': 60,
    'ewma_hl': 20,
    'impact_coeff': 0.1,
}

# 鐢熸垚缁勫悎
keys = list(param_grid.keys())
values = [param_grid[k] for k in keys]
combinations = list(itertools.product(*values))
print(f"total {len(combinations)} param combinations")

# 鍔犺浇鏁版嵁
cfg, models, ic_decay, val_samples, price_dict, vol_dict = load_data_and_models()
fixed_params['future_len'] = cfg.target_horizon

# 閬嶅巻
results = []
for comb in tqdm(combinations, desc="鍙傛暟鎵?弿"):
    params = fixed_params.copy()
    for k, v in zip(keys, comb):
        params[k] = v

    try:
        raw_ret, neu_ret = run_backtest_production(
            models, ic_decay, val_samples, price_dict, vol_dict,
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
        print(f"鍙傛暟 {params} 鍑洪敊: {e}")
        ann = sharpe = mdd = ann_neu = sharpe_neu = mdd_neu = np.nan

    results.append({
        **params,
        'raw_ann': ann, 'raw_sharpe': sharpe, 'raw_mdd': mdd,
        'neu_ann': ann_neu, 'neu_sharpe': sharpe_neu, 'neu_mdd': mdd_neu,
    })

# 淇濆瓨涓庢帓搴?
df = pd.DataFrame(results)
df.to_csv('param_search_results.csv', index=False)

best_raw = df.loc[df['raw_sharpe'].idxmax()]
best_neu = df.loc[df['neu_sharpe'].idxmax()]

print("\n========== 鏈€浼樺弬鏁帮紙鍘熷?澶氱┖澶忔櫘锛?=========")
cols = ['target_vol', 'lambda_t', 'lambda_b', 'max_weight', 'adv_limit_ratio',
        'raw_sharpe', 'raw_ann', 'raw_mdd']
print(best_raw[cols])
print("\n========== 鏈€浼樺弬鏁帮紙淇??涓?€у?鏅?級==========")
cols_neu = ['target_vol', 'lambda_t', 'lambda_b', 'max_weight', 'adv_limit_ratio',
            'neu_sharpe', 'neu_ann', 'neu_mdd']
print(best_neu[cols_neu])
