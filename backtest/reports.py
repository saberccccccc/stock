import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import TRADING_DAYS


def plot_backtest(returns, dates, title="Backtest Performance", output_path=None):
    """Plot cumulative returns and drawdown. Saves to output_path if given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    returns = np.asarray(returns, dtype=float)
    dates = pd.DatetimeIndex(dates[:len(returns)])
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.concatenate(([1.0], cum)))[1:]
    drawdown = (cum - peak) / peak

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14)

    ax1.plot(dates, cum, color="steelblue", linewidth=1)
    ax1.axhline(y=1, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(dates, 0, drawdown, color="red", alpha=0.35)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")
    else:
        plt.show()
    plt.close()


def save_summary_csv(
    rows: list[dict],
    output_path: str | Path,
    display_columns: list[str] | None = None,
    title: str = "Summary",
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n=== {title} ===")
    if display_columns:
        existing_columns = [col for col in display_columns if col in df.columns]
        print(df[existing_columns].to_string(index=False))
    else:
        print(df.to_string(index=False))
    print(f"\nSummary saved to: {output_path}")
    return df


def calc_extended_metrics(returns):
    if len(returns) == 0:
        return {}

    returns = np.nan_to_num(
        np.asarray(returns, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    cum = np.concatenate(([1.0], np.cumprod(1 + returns)))
    days = len(returns)
    years = days / 252

    ann_ret = (cum[-1] ** (1 / years) - 1) * 100 if years > 0 and cum[-1] > 0 else 0.0
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(TRADING_DAYS)
    peak = np.maximum.accumulate(cum)
    mdd = ((peak - cum) / (peak + 1e-12)).max()

    calmar = ann_ret / (mdd * 100 + 1e-8) if mdd > 0 else 0.0

    downside = np.minimum(returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    sortino = returns.mean() / (downside_deviation + 1e-8) * np.sqrt(TRADING_DAYS)

    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.0

    avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.0
    avg_loss = np.mean(np.abs(returns[returns < 0])) if np.any(returns < 0) else 1e-8
    profit_loss_ratio = avg_win / (avg_loss + 1e-8)

    return {
        'ann_return': ann_ret,
        'sharpe': sharpe,
        'max_drawdown': mdd * 100,
        'calmar': calmar,
        'sortino': sortino,
        'win_rate': win_rate * 100,
        'profit_loss_ratio': profit_loss_ratio,
        'total_days': days,
    }


def calc_metrics(returns):
    ext = calc_extended_metrics(returns)
    if not ext:
        return 0, 0, 0
    return ext["ann_return"], ext["sharpe"], ext["max_drawdown"] / 100.0


def calc_active_management_metrics(strategy_returns, benchmark_returns=None):
    """Return benchmark-relative diagnostics used by active management reports."""
    strategy = np.nan_to_num(
        np.asarray(strategy_returns, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if benchmark_returns is None:
        benchmark = np.zeros_like(strategy)
    else:
        benchmark = np.nan_to_num(
            np.asarray(benchmark_returns, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    n = min(len(strategy), len(benchmark))
    if n == 0:
        return {
            "benchmark_ann": 0.0,
            "benchmark_sharpe": 0.0,
            "benchmark_mdd": 0.0,
            "active_ann": 0.0,
            "active_sharpe": 0.0,
            "active_mdd": 0.0,
            "tracking_error": 0.0,
            "information_ratio": 0.0,
            "beta_to_benchmark": 0.0,
            "benchmark_corr": 0.0,
        }

    strategy = strategy[:n]
    benchmark = benchmark[:n]
    active = strategy - benchmark

    benchmark_ann, benchmark_sharpe, benchmark_mdd = calc_metrics(benchmark)
    active_ann, active_sharpe, active_mdd = calc_metrics(active)

    active_std = float(np.std(active))
    tracking_error = active_std * np.sqrt(TRADING_DAYS)
    information_ratio = float(
        np.mean(active) / (active_std + 1e-8) * np.sqrt(TRADING_DAYS)
    )

    benchmark_var = float(np.var(benchmark))
    beta = (
        float(np.cov(strategy, benchmark, ddof=0)[0, 1] / (benchmark_var + 1e-12))
        if n > 1 and benchmark_var > 0
        else 0.0
    )
    corr = (
        float(np.corrcoef(strategy, benchmark)[0, 1])
        if n > 1 and np.std(strategy) > 0 and np.std(benchmark) > 0
        else 0.0
    )

    return {
        "benchmark_ann": float(benchmark_ann),
        "benchmark_sharpe": float(benchmark_sharpe),
        "benchmark_mdd": float(benchmark_mdd),
        "active_ann": float(active_ann),
        "active_sharpe": float(active_sharpe),
        "active_mdd": float(active_mdd),
        "tracking_error": float(tracking_error),
        "information_ratio": float(information_ratio),
        "beta_to_benchmark": beta,
        "benchmark_corr": corr,
    }


def analyze_by_period(returns, dates, period='year'):
    if len(returns) == 0 or len(dates) == 0:
        return {}

    returns = np.array(returns)
    dates = pd.DatetimeIndex(dates[:len(returns)])

    results = {}
    if period == 'year':
        for year in dates.year.unique():
            mask = dates.year == year
            year_returns = returns[mask]
            if len(year_returns) > 0:
                results[str(year)] = calc_extended_metrics(year_returns)

    return results


def compare_predictors(all_results, output_dir="backtest_results"):
    if len(all_results) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*60)
    print("Predictor comparison")
    print("="*60)

    comparison = []
    for name, (data, metrics) in all_results.items():
        ext_raw = calc_extended_metrics(data['daily_returns'])
        ext_neu = calc_extended_metrics(data['neutral_returns'])
        comparison.append({
            'predictor': name,
            'ann_return_raw': f"{ext_raw.get('ann_return', 0):.2f}%",
            'sharpe_raw': f"{ext_raw.get('sharpe', 0):.2f}",
            'mdd_raw': f"{ext_raw.get('max_drawdown', 0):.2f}%",
            'calmar_raw': f"{ext_raw.get('calmar', 0):.2f}",
            'ann_return_neu': f"{ext_neu.get('ann_return', 0):.2f}%",
            'sharpe_neu': f"{ext_neu.get('sharpe', 0):.2f}",
            'mdd_neu': f"{ext_neu.get('max_drawdown', 0):.2f}%",
        })

    df_comp = pd.DataFrame(comparison)
    print("\n" + df_comp.to_string(index=False))
    df_comp.to_csv(f"{output_dir}/comparison_{timestamp}.csv", index=False)
    print(f"\n对比结果已保存到: {output_dir}/comparison_{timestamp}.csv")


def save_backtest_results(
    backtest_data,
    metrics,
    predictor_name,
    portfolio_mode,
    output_dir="backtest_results",
    save_full_data=True,
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{output_dir}/{predictor_name}_{portfolio_mode}_{timestamp}"

    return_dates = backtest_data.get(
        'return_dates',
        backtest_data['dates'][1:len(backtest_data['daily_returns']) + 1],
    )
    return_costs = backtest_data.get(
        'return_costs',
        backtest_data['daily_costs'][1:len(backtest_data['daily_returns']) + 1],
    )
    returns_df = pd.DataFrame({
        'date': return_dates,
        'raw_return': backtest_data['daily_returns'],
        'neutral_return': backtest_data['neutral_returns'],
        'cost': return_costs,
    })
    returns_df.to_csv(f"{prefix}_returns.csv", index=False)

    diag_dict = backtest_data['diagnostics']
    if diag_dict:
        seq_types = (list, np.ndarray)
        max_len = max(len(v) for v in diag_dict.values() if isinstance(v, seq_types))
        diag_dict_padded = {}
        for k, v in diag_dict.items():
            if isinstance(v, seq_types):
                diag_dict_padded[k] = list(v) + [np.nan] * (max_len - len(v))
            else:
                diag_dict_padded[k] = v
        diag_df = pd.DataFrame(diag_dict_padded)
        diag_df.to_csv(f"{prefix}_diagnostics.csv", index=False)

    ext_metrics_raw = calc_extended_metrics(backtest_data['daily_returns'])
    ext_metrics_neu = calc_extended_metrics(backtest_data['neutral_returns'])

    yearly_raw = analyze_by_period(backtest_data['daily_returns'], return_dates, period='year')
    yearly_neu = analyze_by_period(backtest_data['neutral_returns'], return_dates, period='year')

    with open(f"{prefix}_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"预测器 {predictor_name}\n")
        f.write(f"组合模式: {portfolio_mode}\n")
        f.write(f"\n========== 整体表现 ==========\n")
        f.write(f"\n原始多空:\n")
        for k, v in ext_metrics_raw.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\n修正中性\n")
        for k, v in ext_metrics_neu.items():
            f.write(f"  {k}: {v:.4f}\n")

        f.write(f"\n========== 分年度表现（原始）==========\n")
        for year, m in yearly_raw.items():
            f.write(f"\n{year}:\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:.4f}\n")

        f.write(f"\n========== 诊断统计 ==========\n")
        for k, v in backtest_data['diagnostic_counts'].items():
            f.write(f"{k}: {v}\n")

    if save_full_data:
        with open(f"{prefix}_full_data.pkl", 'wb') as f:
            pickle.dump(backtest_data, f)

    print(f"\n结果已保存到: {prefix}_*")
    return prefix
