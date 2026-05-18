# track_backtest_holdings.py - 基于回测最后持仓的收盘价盯市收益跟踪
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.tracking import (
    batch_row,
    best_full_data_path,
    build_tracking_close,
    compute_tracking,
    default_candidate_paths,
    holdings_table,
    last_nonzero_weights,
    latest_full_data_path,
    limit_tracking_close,
    parse_variants,
    resolve_candidate_paths,
    resolve_project_path,
    safe_tushare_call,
    strategy_label,
    update_tracking_prices,
    variant_weights,
    load_backtest_data,
    load_prices,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Track close-to-close PnL from saved backtest holdings")
    parser.add_argument("--result-pkl", default=None, help="*_full_data.pkl. Defaults to the best current production strategy if present.")
    parser.add_argument("--result-mode", choices=["best", "latest"], default="best", help="Default result selection when --result-pkl is omitted.")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="Initial capital used for notional estimates.")
    parser.add_argument("--start-date", default=None, help="Tracking start date. Defaults to the first tracking date after the backtest holding date.")
    parser.add_argument("--end-date", default=None, help="Tracking end date. Defaults to latest tracking close date.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of current holdings to print. 0 prints all.")
    parser.add_argument("--output", default=None, help="Optional output prefix for *_daily.csv and *_holdings.csv.")
    parser.add_argument("--tracking-dir", default="data/tracking_raw", help="Daily tracking CSV directory. Starts after train/validation data and must stay separate from data/raw.")
    parser.add_argument("--history-dir", default="data/raw", help="Historical training/backtest CSV directory used only for the entry close baseline.")
    parser.add_argument("--no-update-tracking", action="store_true", help="Skip automatic Tushare update into tracking-dir.")
    parser.add_argument("--token", default=None, help="Tushare token. Defaults to TUSHARE_TOKEN environment variable.")
    parser.add_argument("--batch-candidates", action="store_true", help="Compare multiple candidate backtest holdings in one table.")
    parser.add_argument("--candidate-pkl", action="append", default=[], help="Candidate *_full_data.pkl path. Can be repeated.")
    parser.add_argument("--candidate-glob", action="append", default=[], help="Candidate glob relative to project root. Can be repeated.")
    parser.add_argument("--batch-output", default=None, help="Optional output prefix for *_comparison.csv in batch mode.")
    parser.add_argument("--variants", default="full,long,top5,top10,top20,top30", help="Comma-separated batch variants: full,long,top5,top10,top20,top30.")
    parser.add_argument("--tracking-days", type=int, default=5, help="Trading return days for tracking_5d rows in batch mode.")
    return parser.parse_args()


def print_batch_comparison(df):
    display = df.copy()
    display = display.sort_values(["tracking_window", "return", "sharpe"], ascending=[True, False, False])
    for col in ["return", "max_drawdown", "leverage", "long_exposure", "short_exposure", "avg_missing_abs_weight", "latest_missing_abs_weight"]:
        display[col] = display[col].map(lambda x: f"{x:.2%}")
    display["ann_return"] = display["ann_return"].map(lambda x: f"{x:.2f}%")
    display["sharpe"] = display["sharpe"].map(lambda x: f"{x:.2f}")
    display["win_rate"] = display["win_rate"].map(lambda x: f"{x:.2f}%")
    columns = [
        "strategy", "variant", "tracking_window", "hold_date", "start_date", "end_date", "days",
        "return", "ann_return", "sharpe", "max_drawdown", "win_rate",
        "positions", "leverage", "long_exposure", "short_exposure",
        "avg_missing_abs_weight", "top_codes",
    ]
    print("\n========== 候选策略追踪对比 ==========")
    print(display[columns].to_string(index=False))


def print_summary(args, result_path, hold_date, close, daily, holdings):
    nav = float(daily["nav"].iloc[-1])
    total_return = nav - 1.0
    metrics = __import__("backtest.engine", fromlist=["calc_extended_metrics"]).calc_extended_metrics(daily["portfolio_return"].iloc[1:].values)
    last = daily.iloc[-1]

    print("\n========== 回测持仓收盘价盯市 ==========")
    print(f"回测文件: {result_path}")
    print(f"历史训练/验证行情目录(只读基准): {args.history_dir}")
    print(f"每日跟踪行情目录: {args.tracking_dir}")
    print(f"回测最后持仓日: {hold_date.date()}")
    print(f"跟踪区间: {close.index[1].date()} -> {close.index[-1].date()}")
    print(f"本金: {args.capital:,.2f}")
    print(f"当前净值: {nav:.6f}")
    print(f"当前估算市值: {args.capital * nav:,.2f}")
    print(f"区间累计收益: {total_return:.2%}")
    print(f"区间最大回撤: {daily['drawdown'].min():.2%}")
    print(f"平均缺失价格持仓数: {daily['missing_price_count'].iloc[1:].mean():.2f}")
    print(f"平均缺失价格绝对权重: {daily['missing_abs_weight'].iloc[1:].mean():.4f}")
    print(f"最新日缺失价格持仓数: {int(daily['missing_price_count'].iloc[-1])}")
    print(f"最新日缺失价格绝对权重: {daily['missing_abs_weight'].iloc[-1]:.4f}")
    if metrics:
        print(f"年化收益(区间折算): {metrics.get('ann_return', 0.0):.2f}%")
        print(f"Sharpe(区间折算): {metrics.get('sharpe', 0.0):.2f}")
        print(f"胜率: {metrics.get('win_rate', 0.0):.2f}%")

    print("\n========== 当前持仓摘要 ==========")
    print(f"持仓数量: {int(last['n_positions'])}")
    print(f"总杠杆: {last['leverage']:.3f}")
    print(f"多头暴露: {last['long_exposure']:.3f}")
    print(f"空头暴露: {last['short_exposure']:.3f}")
    print(f"多头估算市值: {holdings.loc[holdings['weight'] > 0, 'estimated_notional'].sum():,.2f}")
    print(f"空头估算市值: {holdings.loc[holdings['weight'] < 0, 'estimated_notional'].sum():,.2f}")

    print("\n========== 口径说明 ==========")
    print("这是基于回测最后 daily_weights 和本地收盘价的盯市跟踪，不是实盘成交记录。")
    print("现有回测语义是信号后下一交易日收盘价调仓，之后收益按 close-to-close 计算。")
    print("data/raw 只用于取得回测最后持仓日的基准收盘价；每日更新数据应从下一交易日开始写入 data/tracking_raw。")
    print("不要把 tracking_raw 的日更数据合并进 data/raw，否则会改变训练/验证数据集。")
    print("这里展示的是按回测权重、收盘价口径估算的模拟收益；不代表真实成交价、滑点、涨跌停可成交性或账户实际盈亏。")
    print("缺失价格的持仓当日收益按0处理，并在上方输出缺失持仓数和缺失绝对权重。")

    print("\n========== 当前持仓明细 ==========")
    top_n = len(holdings) if args.top_n == 0 else min(args.top_n, len(holdings))
    display_cols = ["code", "side", "weight", "estimated_notional", "entry_close", "latest_close", "period_return", "contribution_est"]
    print(holdings.head(top_n)[display_cols].to_string(index=False, formatters={
        "weight": "{:.4%}".format,
        "estimated_notional": "{:,.2f}".format,
        "entry_close": lambda x: "N/A" if pd.isna(x) else f"{x:.3f}",
        "latest_close": lambda x: "N/A" if pd.isna(x) else f"{x:.3f}",
        "period_return": lambda x: "N/A" if pd.isna(x) else f"{x:.2%}",
        "contribution_est": lambda x: "N/A" if pd.isna(x) else f"{x:.2%}",
    }))


def run_single_tracking(args):
    if args.result_pkl:
        result_path = Path(args.result_pkl)
    elif args.result_mode == "best":
        result_path = best_full_data_path()
    else:
        result_path = latest_full_data_path()
    if not result_path.is_absolute():
        result_path = PROJECT_ROOT / result_path
    data = load_backtest_data(result_path)
    hold_date, weights = last_nonzero_weights(data)

    active_codes = list(weights[np.abs(weights) > 1e-10].index)
    if not args.no_update_tracking:
        update_tracking_prices(active_codes, args.tracking_dir, hold_date, args.token)

    history_prices = load_prices(resolve_project_path(args.history_dir))
    tracking_prices = load_prices(resolve_project_path(args.tracking_dir))
    close = build_tracking_close(history_prices, tracking_prices, list(weights.index), hold_date, args.start_date, args.end_date)
    daily = compute_tracking(close, weights)
    nav = float(daily["nav"].iloc[-1])
    holdings = holdings_table(close, weights, args.capital, nav)
    print_summary(args, result_path, hold_date, close, daily, holdings)

    if args.output:
        output_prefix = Path(args.output)
        if not output_prefix.is_absolute():
            output_prefix = PROJECT_ROOT / output_prefix
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(f"{output_prefix}_daily.csv", index=False)
        holdings.to_csv(f"{output_prefix}_holdings.csv", index=False)
        print(f"\n已保存: {output_prefix}_daily.csv")
        print(f"已保存: {output_prefix}_holdings.csv")


def run_batch_tracking(args):
    variants = parse_variants(args.variants)
    candidate_paths = resolve_candidate_paths(args)
    loaded = []
    active_codes = set()
    hold_dates = []
    for result_path in candidate_paths:
        data = load_backtest_data(result_path)
        hold_date, weights = last_nonzero_weights(data)
        loaded.append((result_path, hold_date, weights))
        hold_dates.append(hold_date)
        active_codes.update(weights[np.abs(weights) > 1e-10].index)

    if not args.no_update_tracking:
        update_tracking_prices(active_codes, args.tracking_dir, min(hold_dates), args.token)

    history_prices = load_prices(resolve_project_path(args.history_dir))
    tracking_prices = load_prices(resolve_project_path(args.tracking_dir))
    rows = []
    errors = []
    for result_path, hold_date, weights in loaded:
        try:
            close = build_tracking_close(history_prices, tracking_prices, list(weights.index), hold_date, args.start_date, args.end_date)
        except Exception as exc:
            errors.append((result_path, "close", exc))
            continue
        for variant in variants:
            weights_v = variant_weights(weights, variant)
            if weights_v.empty:
                errors.append((result_path, variant, "empty weights"))
                continue
            try:
                close_5d = limit_tracking_close(close, args.tracking_days)
                rows.append(batch_row(result_path, hold_date, close_5d, weights_v, variant, "tracking_5d"))
                rows.append(batch_row(result_path, hold_date, close, weights_v, variant, "tracking_to_latest"))
            except Exception as exc:
                errors.append((result_path, variant, exc))

    if not rows:
        raise ValueError("没有可输出的候选策略追踪结果")
    df = pd.DataFrame(rows).sort_values(["tracking_window", "return", "sharpe"], ascending=[True, False, False])
    print_batch_comparison(df)
    if errors:
        print("\n跳过的候选/变体:")
        for path, variant, exc in errors:
            print(f"  {Path(path).name} / {variant}: {exc}")

    if args.batch_output:
        output_prefix = Path(args.batch_output)
        if not output_prefix.is_absolute():
            output_prefix = PROJECT_ROOT / output_prefix
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        out_path = f"{output_prefix}_comparison.csv"
        df.to_csv(out_path, index=False)
        print(f"\n已保存: {out_path}")


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    if args.batch_candidates:
        run_batch_tracking(args)
    else:
        run_single_tracking(args)


if __name__ == "__main__":
    main()
