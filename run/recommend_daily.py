#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.runtime import build_v9_backtest_config, load_backtest_runtime
from data.pipeline import _normalize_ts_code, build_cross_section_dataset, build_inference_sample, build_inference_samples
from run.recommend_utils import (
    build_recommendation_predictor,
    filter_codes_by_prefix,
    filter_main_board,
    resolve_output_path,
    score_recommendation_sample,
)


def load_code_names(data_dir=None):
    data_dir = PROJECT_ROOT / "data" / "raw" if data_dir is None else Path(data_dir)
    path = data_dir / "stable_stocks.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
        if "ts_code" not in df.columns or "name" not in df.columns:
            return {}
        return dict(zip(df["ts_code"].map(_normalize_ts_code), df["name"]))
    except Exception:
        return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Recommend stocks or query one stock from latest/as-of model prediction")
    parser.add_argument("--as-of", default="latest", help="latest or YYYY-MM-DD. Uses the latest available trading date <= this date.")
    parser.add_argument("--from-date", default=None, help="Batch mode: start date YYYY-MM-DD. Requires --to-date.")
    parser.add_argument("--to-date", default=None, help="Batch mode: end date YYYY-MM-DD (inclusive). Requires --from-date.")
    parser.add_argument(
        "--predictor",
        choices=["v9", "gat", "avg_score", "union", "intersection", "top_union_bottom_intersection"],
        default="avg_score",
    )
    parser.add_argument("--top-n", type=int, default=None, help="Number of top recommendations to print. Defaults to 20 when --top-frac is omitted.")
    parser.add_argument("--top-frac", type=float, default=None, help="Top fraction to print when --top-n is omitted.")
    parser.add_argument("--signal-top-pct", type=float, default=0.10, help="Top fraction used inside intersection/union-style ensemble signals.")
    parser.add_argument("--code", default=None, help="Optional stock code to query, e.g. 000001.SZ or 000001")
    parser.add_argument("--output", default=None, help="Optional CSV output path for the ranked recommendation table.")
    parser.add_argument("--v9-checkpoint", default="checkpoints/ultimate_v7_best.pt")
    parser.add_argument("--gat-checkpoint", default="checkpoints/ultimate_v7_gat_best.pt")
    parser.add_argument("--test-stocks", type=int, default=None, help="Limit stocks loaded for fast smoke test.")
    parser.add_argument("--exclude-prefix", default=[], action="append", help="Exclude stock codes starting with this prefix (e.g. 300, 301, 688). Can be repeated.")
    parser.add_argument("--main-board-only", action="store_true", help="Only keep main-board stocks within the original top-N shortlist.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-dir", default=None, help="Data directory override (default: data/raw)")
    return parser.parse_args()


def selection_size(args, n):
    if args.top_n is not None:
        k = args.top_n
        label = f"top_n={args.top_n}"
    elif args.top_frac is not None:
        k = int(n * args.top_frac)
        label = f"top_frac={args.top_frac:.4f}"
    else:
        k = 20
        label = "top_n=20"
    return max(1, min(int(k), n)), label


def print_query(df, code, selected_codes=None, selection_label="推荐列表", fallback_k=None):
    normalized = _normalize_ts_code(code)
    row = df[df["code"] == normalized]
    if row.empty:
        print(f"\n未找到股票 {code}（归一化: {normalized}）在当前有效截面中的预测结果。")
        return
    item = row.iloc[0]
    in_list = normalized in set(selected_codes) if selected_codes is not None else int(item["rank"]) <= (fallback_k or len(df))
    print("\n========== 单股预测查询 ==========")
    print(f"日期: {item['date']}")
    print(f"代码: {item['code']}  板块: {item.get('board', '未知')}")
    print(f"Alpha截面分数: {item['alpha']:.6f}")
    print(f"原始排名: {int(item['rank'])} / {len(df)}")
    print(f"百分位: {item['percentile'] * 100:.2f}%")
    print(f"是否进入推荐列表({selection_label}): {'是' if in_list else '否'}")


def _generate_date_range(from_date, to_date):
    start = pd.Timestamp(from_date)
    end = pd.Timestamp(to_date)
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def _batch_run(args, runtime, predictor):
    dates = _generate_date_range(args.from_date, args.to_date)
    print(f"批量推理: {dates[0]} ~ {dates[-1]}，共 {len(dates)} 个交易日")
    if args.data_dir:
        runtime.cfg.data_dir = args.data_dir
    print("构建推理截面矩阵（一次性加载数据）...")
    samples = build_inference_samples(runtime.cfg, dates)

    all_dfs = []
    for sample in samples:
        ranked, _ = score_recommendation_sample(predictor, sample)
        ranked = filter_codes_by_prefix(ranked, args.exclude_prefix)
        all_dfs.append(ranked)

    if not all_dfs:
        print("没有成功构造任何截面")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    code_names = load_code_names(runtime.cfg.data_dir)
    combined = combined.copy()
    combined["name"] = combined["code"].map(code_names).fillna("")
    if args.main_board_only:
        display_df = filter_main_board(combined)
    else:
        display_df = combined

    print("\n========== 批量推荐结果 ==========")
    print(f"日期范围: {dates[0]} ~ {dates[-1]} | 预测器: {getattr(predictor, 'name', predictor.__class__.__name__)}")
    board_counts = display_df["board"].value_counts()
    board_summary = " | ".join(f"{b}: {c}" for b, c in board_counts.items())
    print(f"板块分布: {board_summary}")
    for d in dates:
        subset = display_df[display_df["date"] == d]
        if subset.empty:
            print(f"  {d}: 无数据")
        else:
            top5 = subset.head(5)
            names = ", ".join(top5["code"].tolist())
            print(f"  {d}: {len(subset)} 只 | Top5: {names}")

    if args.output:
        out_path = resolve_output_path(args.output, PROJECT_ROOT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        display_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n已保存完整排序: {out_path}")
    return display_df


def _single_run(args, runtime, predictor):
    as_of = None if str(args.as_of).lower() == "latest" else args.as_of
    if args.test_stocks is not None:
        runtime.cfg.test_mode = True
        runtime.cfg.test_stocks = args.test_stocks
        runtime.cfg.max_stocks = args.test_stocks

    if args.data_dir:
        runtime.cfg.data_dir = args.data_dir
    code_names = load_code_names(runtime.cfg.data_dir)
    print("构建无标签推理截面...")
    sample = build_inference_sample(runtime.cfg, as_of_date=as_of)

    ranked, regime = score_recommendation_sample(predictor, sample)
    ranked = filter_codes_by_prefix(ranked, args.exclude_prefix)
    ranked = ranked.copy()
    ranked["name"] = ranked["code"].map(code_names).fillna("")

    total_ranked = ranked.copy()
    total_selection_count, selection_label = selection_size(args, len(total_ranked))
    total_top = total_ranked.head(total_selection_count).copy()
    total_top_codes = set(total_top["code"])
    if args.main_board_only:
        display_df = filter_main_board(total_top)
    else:
        display_df = total_top
    display_df = display_df[["rank", "code", "name", "board", "alpha", "percentile"]].copy()

    board_counts = total_ranked["board"].value_counts()
    board_summary = " | ".join(f"{b}: {c}" for b, c in board_counts.items())

    print("\n========== 每日推荐 ==========")
    print(f"日期: {pd.Timestamp(sample['date']).strftime('%Y-%m-%d')} | 预测器: {getattr(predictor, 'name', predictor.__class__.__name__)} | 市场状态: {regime}")
    print(f"板块分布: {board_summary}")
    print(f"总有效股票数: {len(total_ranked)} | 原始Top{total_selection_count}: {len(total_top)} | 显示: {len(display_df)}")
    display = display_df.copy()
    display["percentile"] = (display["percentile"] * 100).map(lambda x: f"{x:.2f}%")
    print(display.to_string(index=False))

    if args.code:
        print_query(
            total_ranked,
            args.code,
            selected_codes=total_top_codes if args.main_board_only else None,
            selection_label=selection_label,
            fallback_k=total_selection_count,
        )

    if args.output:
        out_path = resolve_output_path(args.output, PROJECT_ROOT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        display_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n已保存完整排序: {out_path}")


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    cfg = build_v9_backtest_config()
    print("加载训练样本缓存用于模型维度推断...")
    runtime = load_backtest_runtime(cfg, use_cache=True)

    print("加载预测器...")
    predictor = build_recommendation_predictor(
        args.predictor,
        runtime.train,
        runtime.cfg,
        v9_checkpoint=args.v9_checkpoint,
        gat_checkpoint=args.gat_checkpoint,
        device=args.device,
        signal_top_pct=args.signal_top_pct,
    )

    if args.from_date and args.to_date:
        _batch_run(args, runtime, predictor)
    else:
        _single_run(args, runtime, predictor)

    data_label = args.data_dir or "data/raw"
    print(f"\n说明: alpha 是模型截面排序分数，不是预期收益率；latest 使用本地 {data_label} 最新可用日线数据。")


if __name__ == "__main__":
    main()
