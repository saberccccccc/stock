#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9 Transformer vs GAT 相似度分析和并集回测
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from core.config import DataConfig
from data.pipeline import build_cross_section_dataset
from backtest.engine import load_v9_checkpoint, DLPredictor, load_price_volume


def compute_similarity(pred1, pred2, top_pct=0.1):
    """计算两个预测的 top/bottom 相似度"""
    n = len(pred1)
    k = max(1, int(n * top_pct))

    idx1_top = np.argsort(pred1)[-k:]
    idx2_top = np.argsort(pred2)[-k:]
    idx1_bot = np.argsort(pred1)[:k]
    idx2_bot = np.argsort(pred2)[:k]

    top_inter = len(set(idx1_top) & set(idx2_top))
    top_union = len(set(idx1_top) | set(idx2_top))
    bot_inter = len(set(idx1_bot) & set(idx2_bot))
    bot_union = len(set(idx1_bot) | set(idx2_bot))

    spearman = np.corrcoef(
        np.argsort(np.argsort(pred1)),
        np.argsort(np.argsort(pred2))
    )[0, 1]

    return {
        "top_jaccard": top_inter / top_union if top_union > 0 else 0,
        "bot_jaccard": bot_inter / bot_union if bot_union > 0 else 0,
        "top_overlap": top_inter / k,
        "bot_overlap": bot_inter / k,
        "spearman": spearman,
        "top_union_size": top_union,
        "bot_union_size": bot_union,
    }


def run_union_backtest(val, v9_pred, gat_pred, price_dict, top_pct=0.1, mode="union"):
    """运行并集或交集回测

    Args:
        mode: "union" 或 "intersection"
    """
    returns = []
    dates = []

    for i in range(len(val) - 1):
        sample = val[i]
        next_sample = val[i + 1]
        codes = sample["codes"]
        n_stocks = len(codes)
        valid = np.ones(n_stocks, dtype=bool)
        regime = None

        if n_stocks < 20:
            continue

        alpha_v9 = v9_pred.predict_alpha(sample, valid, regime)
        alpha_gat = gat_pred.predict_alpha(sample, valid, regime)

        n = len(alpha_v9)
        k = max(1, int(n * top_pct))

        idx_v9_top = set(np.argsort(alpha_v9)[-k:])
        idx_gat_top = set(np.argsort(alpha_gat)[-k:])
        idx_v9_bot = set(np.argsort(alpha_v9)[:k])
        idx_gat_bot = set(np.argsort(alpha_gat)[:k])

        if mode == "union":
            combined_top = list(idx_v9_top | idx_gat_top)
            combined_bot = list(idx_v9_bot | idx_gat_bot)
        else:  # intersection
            combined_top = list(idx_v9_top & idx_gat_top)
            combined_bot = list(idx_v9_bot & idx_gat_bot)

        long_codes = [codes[i] for i in combined_top]
        short_codes = [codes[i] for i in combined_bot]

        ret_long = []
        for code in long_codes:
            if code in price_dict and sample["date"] in price_dict[code].index:
                idx_curr = price_dict[code].index.get_loc(sample["date"])
                if idx_curr + 1 < len(price_dict[code]):
                    p0 = price_dict[code].iloc[idx_curr]
                    p1 = price_dict[code].iloc[idx_curr + 1]
                    if p0 > 0:
                        ret_long.append((p1 - p0) / p0)

        ret_short = []
        for code in short_codes:
            if code in price_dict and sample["date"] in price_dict[code].index:
                idx_curr = price_dict[code].index.get_loc(sample["date"])
                if idx_curr + 1 < len(price_dict[code]):
                    p0 = price_dict[code].iloc[idx_curr]
                    p1 = price_dict[code].iloc[idx_curr + 1]
                    if p0 > 0:
                        ret_short.append(-(p1 - p0) / p0)

        if ret_long or ret_short:
            daily_ret = np.mean(ret_long + ret_short)
            returns.append(daily_ret)
            dates.append(sample["date"])

    return np.array(returns), dates


def main():
    cfg = DataConfig()
    cfg.use_technical_features = True
    cfg.use_market_features = True
    cfg.use_macro_features = True
    cfg.min_stocks_per_time = 30
    cfg.target_horizon = 5
    cfg.seq_len = 40
    cfg.max_horizon = 10

    print("加载数据...")
    train, val = build_cross_section_dataset(cfg, use_cache=True)
    price_dict, vol_dict = load_price_volume(cfg)

    print("加载 V9 Transformer...")
    v9_model, v9_device, v9_regime = load_v9_checkpoint(
        "checkpoints/ultimate_v7_best.pt", train, cfg, "auto"
    )
    v9_pred = DLPredictor(v9_model, v9_device, v9_regime)

    print("加载 GAT...")
    gat_model, gat_device, gat_regime = load_v9_checkpoint(
        "checkpoints/ultimate_v7_gat_best.pt", train, cfg, "auto"
    )
    gat_pred = DLPredictor(gat_model, gat_device, gat_regime)

    print(f"验证集样本数: {len(val)}")
    print("\n=== 第1步：相似度分析 ===")

    similarities = []
    for i, sample in enumerate(val):
        codes = sample["codes"]
        n_stocks = len(codes)
        valid = np.ones(n_stocks, dtype=bool)
        regime = None

        if n_stocks < 20:
            continue

        alpha_v9 = v9_pred.predict_alpha(sample, valid, regime)
        alpha_gat = gat_pred.predict_alpha(sample, valid, regime)

        sim = compute_similarity(alpha_v9, alpha_gat)
        similarities.append(sim)

        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1}/{len(val)} 个样本")

    df_sim = pd.DataFrame(similarities)
    print("\n相似度统计:")
    print(df_sim.describe())

    out_path = PROJECT_ROOT / "backtest_results" / "v9_gat_similarity.csv"
    df_sim.to_csv(out_path, index=False)
    print(f"相似度数据已保存: {out_path}")

    print("\n=== 第2步：并集回测 ===")
    returns_union, dates_union = run_union_backtest(val, v9_pred, gat_pred, price_dict, mode="union")

    cum_ret = (1 + returns_union).cumprod()
    ann_ret = (cum_ret[-1] ** (252 / len(returns_union)) - 1) * 100
    sharpe = returns_union.mean() / returns_union.std() * np.sqrt(252)
    mdd = ((cum_ret / np.maximum.accumulate(cum_ret)) - 1).min() * 100

    print(f"\n并集 Long-Short 回测结果:")
    print(f"  年化收益: {ann_ret:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  最大回撤: {mdd:.2f}%")
    print(f"  交易日数: {len(returns_union)}")

    df_ret_union = pd.DataFrame({"date": dates_union, "return": returns_union})
    ret_path = PROJECT_ROOT / "backtest_results" / "v9_gat_union_returns.csv"
    df_ret_union.to_csv(ret_path, index=False)
    print(f"收益数据已保存: {ret_path}")

    print("\n=== 第3步：交集回测 ===")
    returns_inter, dates_inter = run_union_backtest(val, v9_pred, gat_pred, price_dict, mode="intersection")

    cum_ret_inter = (1 + returns_inter).cumprod()
    ann_ret_inter = (cum_ret_inter[-1] ** (252 / len(returns_inter)) - 1) * 100
    sharpe_inter = returns_inter.mean() / returns_inter.std() * np.sqrt(252)
    mdd_inter = ((cum_ret_inter / np.maximum.accumulate(cum_ret_inter)) - 1).min() * 100

    print(f"\n交集 Long-Short 回测结果:")
    print(f"  年化收益: {ann_ret_inter:.2f}%")
    print(f"  Sharpe: {sharpe_inter:.2f}")
    print(f"  最大回撤: {mdd_inter:.2f}%")
    print(f"  交易日数: {len(returns_inter)}")

    df_ret_inter = pd.DataFrame({"date": dates_inter, "return": returns_inter})
    ret_path_inter = PROJECT_ROOT / "backtest_results" / "v9_gat_intersection_returns.csv"
    df_ret_inter.to_csv(ret_path_inter, index=False)
    print(f"收益数据已保存: {ret_path_inter}")


if __name__ == "__main__":
    main()
