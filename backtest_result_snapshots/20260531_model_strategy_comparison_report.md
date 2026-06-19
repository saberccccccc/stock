# Model + Strategy Comparison 2026-05-31

Objective:

```text
结合不同模型，优化回测策略，找出夏普比和年化收益最好的组合
```

## Main Conclusion

The best current unconstrained same-engine combination after the V9 strategy and signal-smoothing sweep is:

```text
V9 topfocus checkpoint
+ raw daily alpha signal
+ retention-first daily state strategy
+ target3/hold25
+ legacy market timing
+ equal weight
+ explicit full costs
```

Result:

| split | annual return | Sharpe | max drawdown | avg turnover | avg holding days |
|---|---:|---:|---:|---:|---:|
| validation | 109.50% | 2.128 | 15.81% | 0.360 | 4.61 |
| test | 146.25% | 4.218 | 12.11% | 0.508 | 3.69 |

The more practical lower-turnover unconstrained candidate is:

```text
V9 topfocus checkpoint
+ raw daily alpha signal
+ retention-first daily state strategy
+ target3/hold40
+ legacy market timing
+ equal weight
+ explicit full costs
```

Result:

| split | annual return | Sharpe | max drawdown | avg turnover | avg holding days |
|---|---:|---:|---:|---:|---:|
| validation | 96.68% | 2.011 | 15.29% | 0.268 | 6.26 |
| test | 142.78% | 4.223 | 11.66% | 0.392 | 4.84 |

Previous V10 best balanced combination was:

```text
V10 temporal original seed 5590
+ retention-first daily state strategy
+ target30/hold80
+ legacy market timing
+ equal weight
+ explicit full costs
```

Result:

| split | annual return | Sharpe | max drawdown | avg turnover | avg holding days |
|---|---:|---:|---:|---:|---:|
| validation | 32.65% | 0.991 | 16.73% | 0.072 | 26.00 |
| test | 48.73% | 2.161 | 12.58% | 0.095 | 23.11 |

The raw `target3/hold25` result is selected by unconstrained validation Sharpe. The raw `target3/hold40` variant has slightly better unconstrained test Sharpe, lower drawdown, lower turnover, and better 2x-cost behavior.

However, after adding stricter execution constraints, the best production-realistic candidate changes to:

```text
V9 topfocus checkpoint
+ average_w3 signal
+ retention-first daily state strategy
+ target3/hold30
+ legacy market timing
+ equal weight
+ explicit full costs
+ 100M CNY portfolio
+ per-stock 5% ADV participation cap
+ min 20M CNY ADV filter
+ approximate limit-up/down trade blocking
```

Result:

| split | annual return | Sharpe | max drawdown | executed turnover | avg holding days |
|---|---:|---:|---:|---:|---:|
| validation | 43.97% | 1.246 | 16.24% | 0.192 | 9.10 |
| test | 68.22% | 2.581 | 13.29% | 0.240 | 8.10 |

The stricter test-best variant is `average_w3 target3/hold40`:

| split | annual return | Sharpe | max drawdown | executed turnover | avg holding days |
|---|---:|---:|---:|---:|---:|
| validation | 38.79% | 1.143 | 16.71% | 0.160 | 10.95 |
| test | 70.20% | 2.661 | 13.38% | 0.201 | 9.79 |

The V9 result is now the stronger same-engine candidate because it has:

- the highest tested Sharpe among reliable test results,
- the highest reliable test annual return,
- comparable drawdown,
- full explicit transaction costs,
- and a daily portfolio-state engine rather than the older fixed-holding engine.

The trade-off is higher turnover than the earlier `target20/hold80` version, though costs are already included.

Cost assumptions:

```text
commission: 0.01% per side
stamp tax: 0.05% sell side
slippage/spread: 0.05% per side
```

## Latest Signal Smoothing Sweep

The previous mainline used `average_w3` smoothing. A direct raw-vs-smoothed comparison showed that smoothing was suppressing useful short-horizon V9 signal strength for this retention strategy.

Normal cost results:

| signal | split | setup | ann | Sharpe | max drawdown | avg turnover | avg holding days |
|---|---|---|---:|---:|---:|---:|---:|
| raw | val | target3/hold25 | 109.50% | 2.128 | 15.81% | 0.360 | 4.61 |
| raw | val | target3/hold40 | 96.68% | 2.011 | 15.29% | 0.268 | 6.26 |
| raw | test | target3/hold25 | 146.25% | 4.218 | 12.11% | 0.508 | 3.69 |
| raw | test | target3/hold40 | 142.78% | 4.223 | 11.66% | 0.392 | 4.84 |
| average_w3 | val | target3/hold25 | 64.12% | 1.584 | 16.62% | 0.219 | 7.85 |
| average_w3 | val | target3/hold40 | 55.73% | 1.462 | 15.83% | 0.165 | 10.18 |
| average_w3 | test | target3/hold25 | 88.50% | 3.126 | 13.10% | 0.268 | 7.16 |
| average_w3 | test | target3/hold40 | 89.15% | 3.175 | 13.14% | 0.203 | 9.49 |
| composite_w3 | val | target3/hold25 | 37.66% | 1.024 | 17.24% | 0.920 | 1.76 |
| composite_w3 | val | target3/hold40 | 47.39% | 1.220 | 15.78% | 0.799 | 2.03 |

2x explicit-cost stress for raw signal:

| signal | split | setup | ann | Sharpe | max drawdown | avg turnover |
|---|---|---|---:|---:|---:|---:|
| raw | val | target3/hold25 | 94.10% | 1.927 | 15.95% | 0.360 |
| raw | val | target3/hold40 | 85.80% | 1.858 | 15.43% | 0.268 |
| raw | test | target3/hold25 | 121.06% | 3.728 | 12.60% | 0.508 |
| raw | test | target3/hold40 | 123.37% | 3.838 | 12.01% | 0.392 |

Interpretation:

- `raw` is now the strongest signal mode by both validation and test.
- `average_w3` is still useful as a lower-turnover conservative fallback, but no longer the mainline.
- `composite_w3` is rejected because it creates very high turnover and much weaker validation Sharpe.
- `target3/hold40` is likely the better practical raw variant because it keeps almost all return, improves drawdown, lowers turnover, and wins under 2x costs on test.

## Reliable Candidate Ranking

### Best by test annual return

| rank | model / strategy | market | test ann | test Sharpe | test mdd | validation ann | validation Sharpe | note |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | V9 topfocus average_w3, target3/hold40 retention | legacy | 89.15% | 3.175 | 13.14% | 55.73% | 1.462 | Highest test ann/Sharpe, but not validation-selected. |
| 2 | V9 topfocus average_w3, target3/hold25 retention | legacy | 88.50% | 3.126 | 13.10% | 64.12% | 1.584 | Validation-selected best Sharpe. |
| 3 | V9 topfocus average_w3, target3/hold30 retention | legacy | 85.72% | 3.066 | 12.92% | 62.95% | 1.578 | Slightly smoother than hold25. |
| 4 | V9 topfocus average_w3, target5/hold25 retention | legacy | 84.27% | 3.068 | 12.80% | 60.31% | 1.540 | More diversified top pool. |
| 5 | V9 topfocus average_w3, target5/hold30 retention | legacy | 83.37% | 3.062 | 12.59% | 57.57% | 1.502 | Balanced narrow-pool candidate. |
| 6 | V9 topfocus average_w3, target20/hold80 retention | legacy | 64.22% | 2.665 | 12.79% | 33.52% | 1.063 | Lower-turnover older candidate. |
| 7 | V10 seed5590, target30/hold80 retention | legacy | 48.73% | 2.161 | 12.58% | 32.65% | 0.991 | Previous best V10 candidate. |

### Best by test Sharpe

| rank | model / strategy | market | test Sharpe | test ann | test mdd | validation Sharpe | validation ann | note |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | V9 topfocus average_w3, target3/hold40 retention | legacy | 3.175 | 89.15% | 13.14% | 1.462 | 55.73% | Highest test Sharpe. |
| 2 | V9 topfocus average_w3, target3/hold25 retention | legacy | 3.126 | 88.50% | 13.10% | 1.584 | 64.12% | Validation-selected mainline. |
| 3 | V9 topfocus average_w3, target3/hold50 retention | legacy | 3.071 | 85.20% | 13.31% | 1.379 | 50.73% | Lower turnover than hold25/40. |
| 4 | V9 topfocus average_w3, target5/hold25 retention | legacy | 3.068 | 84.27% | 12.80% | 1.540 | 60.31% | Slightly wider top pool. |
| 5 | V9 topfocus average_w3, target3/hold30 retention | legacy | 3.066 | 85.72% | 12.92% | 1.578 | 62.95% | Close to selected mainline. |
| 6 | V10 seed5590, target30/hold80 retention | legacy | 2.161 | 48.73% | 12.58% | 0.991 | 32.65% | Previous best V10 choice. |

## V9 + Legacy/Dynamic Result

Tested the previously best reliable V9 setup:

```text
checkpoint = checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt
signal = average_w3
top_frac = 5%
rank_exit_frac = 15%
rebalance_freq = 4
portfolio = simple_long
```

Results:

| market timing | min mult | raw ann | raw Sharpe | raw mdd | neutral ann | neutral Sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 0.20 | 36.46% | 1.484 | 15.86% | 30.61% | 1.628 | 11.78% |
| dynamic | 0.20 | 26.63% | 1.389 | 14.36% | 22.84% | 1.510 | 11.45% |
| dynamic | 0.50 | 33.10% | 1.448 | 19.46% | 26.50% | 1.531 | 16.02% |

Conclusion:

- The old V9 best reliable strategy already used `legacy`.
- Replacing it with `dynamic` does **not** improve annual return or Sharpe.
- `dynamic min0.20` lowers drawdown slightly but cuts too much return.
- `dynamic min0.50` still fails to beat `legacy` and has worse drawdown.
- Therefore V9 + dynamic is not the best direction.

## V9 Under V10 Retention Engine

To make the V9 vs V10 comparison fair, V9 alpha was retested through the same newer retention engine used by V10.

Script:

```text
run/backtest_v9_retention.py
```

Setup:

```text
checkpoint = checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt
predictor = v9_raw_persistent_average_w3
split val = 2024-01-01 to 2024-12-31
split test = 2025-01-01 onward
market = legacy
weight = equal
costs = commission 0.01% both sides + stamp tax 0.05% sell side + slippage 0.05% both sides
```

Results:

| split | setup | ann | Sharpe | max drawdown | avg turnover | avg holding days |
|---|---|---:|---:|---:|---:|---:|
| val | target20/hold80 | 33.52% | 1.063 | 15.19% | 0.069 | 26.60 |
| val | target20/hold90 | 31.26% | 1.012 | 14.97% | 0.046 | 42.05 |
| val | target30/hold80 | 34.01% | 1.088 | 14.99% | 0.069 | 26.66 |
| val | target30/hold90 | 30.47% | 1.002 | 15.16% | 0.046 | 42.67 |
| test | target20/hold80 | 64.22% | 2.665 | 12.79% | 0.077 | 29.62 |
| test | target20/hold90 | 59.98% | 2.500 | 13.89% | 0.054 | 48.49 |
| test | target30/hold80 | 60.29% | 2.608 | 12.19% | 0.077 | 29.54 |
| test | target30/hold90 | 56.41% | 2.457 | 13.17% | 0.053 | 49.80 |

Conclusion:

- Under the same daily retention/full-cost engine, V9 topfocus average_w3 beats current V10 on both validation and test.
- `target20/hold80` has the best test return and Sharpe.
- `target30/hold80` has the best validation Sharpe and slightly lower test drawdown.
- The current best production candidate should move from V10 to V9 retention unless later V10 ensembles exceed it.

## V9 Fine Target/Hold Sweep

After the first same-engine V9 test, a finer grid was run around narrower top pools.

Grid:

```text
target_fracs = 3%, 5%, 7.5%, 10%, 12.5%, 15%
hold_fracs = 25%, 30%, 40%, 50%, 60%
```

Best validation rows by Sharpe:

| setup | val ann | val Sharpe | val mdd | val turnover | test ann | test Sharpe | test mdd | test turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target3/hold25 | 64.12% | 1.584 | 16.62% | 0.219 | 88.50% | 3.126 | 13.10% | 0.268 |
| target3/hold30 | 62.95% | 1.578 | 15.39% | 0.198 | 85.72% | 3.066 | 12.92% | 0.242 |
| target5/hold25 | 60.31% | 1.540 | 15.27% | 0.222 | 84.27% | 3.068 | 12.80% | 0.274 |
| target5/hold30 | 57.57% | 1.502 | 15.30% | 0.201 | 83.37% | 3.062 | 12.59% | 0.246 |
| target7.5/hold25 | 56.52% | 1.483 | 14.69% | 0.229 | 80.20% | 3.027 | 12.56% | 0.279 |

Best test row:

| setup | test ann | test Sharpe | test mdd | test turnover | val ann | val Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| target3/hold40 | 89.15% | 3.175 | 13.14% | 0.203 | 55.73% | 1.462 |

Selection:

- If selecting only by test, `target3/hold40` wins.
- If selecting by validation first, `target3/hold25` wins and still performs almost as well on test.
- Current mainline should therefore be `target3/hold25`, with `target3/hold40` recorded as the test-best exploratory variant.

### Stage stability

For the current validation-selected mainline `target3/hold25`:

| split | period | days | ann | Sharpe | max drawdown | sum return |
|---|---|---:|---:|---:|---:|---:|
| val | 2024 | 241 | 71.91% | 1.718 | 16.62% | 0.5780 |
| val | all | 243 | 64.12% | 1.584 | 16.62% | 0.5381 |
| test | 2025 | 242 | 79.63% | 2.922 | 13.10% | 0.5841 |
| test | 2026 | 78 | 118.94% | 3.741 | 8.11% | 0.2501 |
| test | all | 320 | 88.50% | 3.126 | 13.10% | 0.8343 |

Notes:

- Validation has two return days spilling into early 2025 due next-day execution alignment; they are negative and reduce the all-period validation metric.
- The result is not only a 2026 effect. The full 2025 segment is already strong.
- Weak validation months are mainly `2024-01`, `2024-06`, and `2024-08`.
- Weak test month is mainly `2026-03`.

### Cost stress

The main narrow-pool candidates were retested with all explicit costs doubled:

```text
commission: 0.02% per side
stamp tax: 0.10% sell side
slippage/spread: 0.10% per side
```

| split | setup | ann | Sharpe | max drawdown | avg turnover | avg names | total cost |
|---|---|---:|---:|---:|---:|---:|---:|
| val | target3/hold25 | 56.67% | 1.452 | 17.80% | 0.219 | 148.1 | 0.0897 |
| val | target3/hold30 | 56.26% | 1.458 | 16.43% | 0.198 | 148.1 | 0.0809 |
| val | target3/hold40 | 50.39% | 1.360 | 16.71% | 0.165 | 148.1 | 0.0675 |
| test | target3/hold25 | 78.07% | 2.852 | 13.34% | 0.268 | 149.6 | 0.1449 |
| test | target3/hold30 | 76.40% | 2.818 | 13.14% | 0.242 | 149.6 | 0.1310 |
| test | target3/hold40 | 81.16% | 2.966 | 13.33% | 0.203 | 149.6 | 0.1098 |

Interpretation:

- The narrow-pool strategy survives a 2x explicit-cost stress test.
- `target3/hold40` becomes more attractive under cost stress because it reduces turnover while preserving most of the return.
- `target3/hold25` remains the validation-selected high-return mainline, but `target3/hold30` or `target3/hold40` are more practical if cost/capacity risk is prioritized.

### V9 yearly breakdown

Stage breakdown files:

```text
backtest_result_snapshots/20260531_v9_best_market_timing_stage_breakdown/yearly_summary.csv
backtest_result_snapshots/20260531_v9_best_market_timing_stage_breakdown/monthly_summary.csv
```

Raw-return yearly results:

| mode | period | days | ann | Sharpe | max drawdown | sum return |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 2023 | 211 | -2.79% | -0.225 | 8.92% | -0.0193 |
| legacy | 2024 | 242 | 38.62% | 1.188 | 15.86% | 0.3628 |
| legacy | 2025 | 243 | 57.83% | 2.411 | 12.03% | 0.4595 |
| legacy | 2026 | 81 | 103.59% | 3.413 | 9.13% | 0.2363 |
| legacy | all | 777 | 36.46% | 1.484 | 15.86% | 1.0393 |
| dynamic min0.20 | 2023 | 211 | -1.45% | -0.164 | 6.66% | -0.0100 |
| dynamic min0.20 | 2024 | 242 | 30.92% | 1.147 | 12.06% | 0.2932 |
| dynamic min0.20 | 2025 | 243 | 35.63% | 2.141 | 8.64% | 0.3046 |
| dynamic min0.20 | 2026 | 81 | 79.34% | 3.721 | 6.62% | 0.1921 |
| dynamic min0.20 | all | 777 | 26.63% | 1.389 | 14.36% | 0.7799 |
| dynamic min0.50 | 2023 | 211 | -1.34% | -0.092 | 8.71% | -0.0074 |
| dynamic min0.50 | 2024 | 242 | 34.87% | 1.137 | 16.06% | 0.3323 |
| dynamic min0.50 | 2025 | 243 | 50.78% | 2.380 | 11.15% | 0.4120 |
| dynamic min0.50 | 2026 | 81 | 92.02% | 3.451 | 8.28% | 0.2161 |
| dynamic min0.50 | all | 777 | 33.10% | 1.448 | 19.46% | 0.9530 |

Interpretation:

- Yes, V9 should also be split by year when comparing to V10. Otherwise a strong late period can hide a weak early period.
- V9 legacy is strongest overall, but it still has a weak 2023 segment.
- Dynamic reduces early drawdown a little, but the return sacrifice is too large.
- V10 and V9 are still not perfectly same-window comparable here: V9 old-engine validation spans `2023-2026`, while V10 retention comparison uses `2024 validation` and `2025-2026 test`.

## Caveats

V9 results are useful but less comparable than V10 retention results:

- V9 uses the older production backtest engine.
- The code audit found possible cost trimming in older engines.
- Some V9 daily rank-exit results around `50%` annualized were marked as partial / directionally useful, not final production evidence.
- V10 retention uses a daily portfolio-state engine and explicit full costs:
  - commission `0.01%` both sides,
  - stamp tax `0.05%` sell side,
  - slippage/spread `0.05%` both sides.

After same-engine retesting, V9 retention is currently stronger than V10 retention. V10 remains useful as a lower-turnover / lower-model-correlation candidate, but it is no longer the mainline by return or Sharpe.

## Capacity / Liquidity Diagnostics

Added script:

```text
run/analyze_retention_capacity.py
```

Method:

```text
reconstruct retention holdings from daily alpha ranks
estimate each trade's value as abs(delta_weight) * portfolio_value
compare trade value with previous 20 trading-day ADV
ADV uses raw money column scaled by 1000 to CNY and is shifted by 1 day
```

Capacity files:

```text
backtest_results_v9_retention_20260531/capacity_val_target030_hold250/
backtest_results_v9_retention_20260531/capacity_val_target030_hold400/
backtest_results_v9_retention_20260531/capacity_test_target030_hold250/
backtest_results_v9_retention_20260531/capacity_test_target030_hold400/
```

At 100 million CNY portfolio value:

| signal | split | setup | trades | p50 ADV | p90 ADV | p95 ADV | p99 ADV | max ADV | >5% ADV | >10% ADV |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| average_w3 | val | target3/hold25 | 14,530 | 0.24% | 1.48% | 2.10% | 4.00% | 13.07% | 0.45% | 0.04% |
| average_w3 | val | target3/hold40 | 12,495 | 0.15% | 1.31% | 1.91% | 3.83% | 12.91% | 0.38% | 0.02% |
| average_w3 | test | target3/hold25 | 20,421 | 0.18% | 1.10% | 1.58% | 2.82% | 7.46% | 0.09% | 0.00% |
| average_w3 | test | target3/hold40 | 17,328 | 0.15% | 0.98% | 1.47% | 2.67% | 6.95% | 0.11% | 0.00% |
| raw | val | target3/hold25 | 20,233 | 0.29% | 1.53% | 2.16% | 4.17% | 22.39% | 0.52% | 0.04% |
| raw | val | target3/hold40 | 16,489 | 0.23% | 1.41% | 2.01% | 3.92% | 12.55% | 0.46% | 0.03% |
| raw | test | target3/hold25 | 31,705 | 0.19% | 1.02% | 1.49% | 2.74% | 10.63% | 0.12% | 0.00% |
| raw | test | target3/hold40 | 26,171 | 0.16% | 0.90% | 1.32% | 2.54% | 10.53% | 0.09% | 0.00% |

At 500 million CNY portfolio value:

| signal | split | setup | p50 ADV | p90 ADV | p95 ADV | p99 ADV | max ADV | >5% ADV | >10% ADV | >20% ADV |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| average_w3 | val | target3/hold25 | 1.18% | 7.41% | 10.52% | 19.98% | 65.34% | 17.72% | 5.60% | 1.00% |
| average_w3 | val | target3/hold40 | 0.76% | 6.55% | 9.56% | 19.16% | 64.57% | 14.61% | 4.57% | 0.82% |
| average_w3 | test | target3/hold25 | 0.92% | 5.51% | 7.92% | 14.09% | 37.32% | 11.62% | 2.75% | 0.26% |
| average_w3 | test | target3/hold40 | 0.73% | 4.92% | 7.33% | 13.34% | 34.76% | 9.77% | 2.28% | 0.23% |
| raw | val | target3/hold25 | 1.47% | 7.66% | 10.79% | 20.83% | 111.97% | 18.76% | 5.86% | 1.11% |
| raw | val | target3/hold40 | 1.16% | 7.04% | 10.03% | 19.62% | 62.75% | 16.29% | 5.05% | 0.99% |
| raw | test | target3/hold25 | 0.95% | 5.09% | 7.44% | 13.69% | 53.16% | 10.31% | 2.40% | 0.26% |
| raw | test | target3/hold40 | 0.79% | 4.51% | 6.62% | 12.71% | 52.66% | 8.44% | 1.90% | 0.21% |

Interpretation:

- 10-100 million CNY scale looks capacity-safe under this ADV diagnostic, including raw signal.
- 500 million CNY starts to show meaningful liquidity pressure, especially on validation 2024.
- `target3/hold40` is more capacity-friendly than `target3/hold25` because it cuts trade count and turnover while preserving most performance. This is especially important for raw signal.
- The next stricter check should add ST/new-listing flags when those fields are available. Current raw CSV files include price/volume/money but not stock names or listing dates, so ST/new-listing filters are not yet exact.

## Execution-Constrained Retest

Added script:

```text
run/backtest_retention_execution_constraints.py
run/sweep_execution_constraints.py
```

This goes beyond diagnostics and changes the simulated portfolio state:

```text
portfolio value = 100M CNY
per-stock trade cap = 5% of previous 20-day ADV
minimum ADV = 20M CNY
buy blocked when daily close return >= +9.5%
sell blocked when daily close return <= -9.5%
unfilled trades remain as the previous position/cash state
costs remain commission 0.01% both sides + stamp tax 0.05% sell side + slippage 0.05% both sides
```

Strict execution result:

| signal | split | setup | ann | Sharpe | max drawdown | executed turnover | unfilled turnover | avg holding days |
|---|---|---|---:|---:|---:|---:|---:|---:|
| raw | val | target3/hold25 | 42.94% | 1.220 | 15.84% | 0.336 | 0.044 | 4.95 |
| raw | val | target3/hold35 | 42.99% | 1.226 | 15.84% | 0.275 | 0.035 | 6.09 |
| raw | val | target3/hold40 | 42.83% | 1.226 | 16.08% | 0.251 | 0.033 | 6.69 |
| raw | test | target3/hold40 | 60.15% | 2.311 | 12.68% | 0.376 | 0.020 | 5.00 |
| raw | test | target3/hold50 | 61.86% | 2.405 | 12.43% | 0.320 | 0.018 | 5.93 |
| average_w3 | val | target3/hold30 | 43.97% | 1.246 | 16.24% | 0.192 | 0.033 | 9.10 |
| average_w3 | val | target3/hold35 | 43.50% | 1.240 | 16.14% | 0.175 | 0.029 | 10.01 |
| average_w3 | test | target3/hold30 | 68.22% | 2.581 | 13.29% | 0.240 | 0.011 | 8.10 |
| average_w3 | test | target3/hold40 | 70.20% | 2.661 | 13.38% | 0.201 | 0.009 | 9.79 |
| average_w3 | test | target3/hold50 | 69.97% | 2.650 | 13.67% | 0.163 | 0.008 | 12.24 |

2x-cost stress under the same execution constraints:

| signal | split | setup | ann | Sharpe | max drawdown | executed turnover |
|---|---|---|---:|---:|---:|---:|
| average_w3 | val | target3/hold30 | 38.25% | 1.126 | 17.25% | 0.192 |
| average_w3 | val | target3/hold40 | 34.17% | 1.043 | 17.53% | 0.160 |
| average_w3 | test | target3/hold30 | 59.88% | 2.338 | 13.50% | 0.240 |
| average_w3 | test | target3/hold40 | 63.11% | 2.456 | 13.56% | 0.201 |

Interpretation:

- The raw signal's unconstrained 140%+ annualized result is not production-realistic at 100M CNY with strict ADV/limit constraints.
- Under strict execution, `average_w3` becomes stronger because lower turnover means fewer unfilled or blocked trades.
- Validation-selected strict mainline is `average_w3 target3/hold30`.
- Test-best strict practical variant is `average_w3 target3/hold40`; it has lower turnover and better test Sharpe, but weaker validation.
- The strict result is still strong enough to remain useful: validation around 40% annualized and test around 60-70% annualized after full costs and execution constraints.

### Execution sensitivity grid

Ran a 54-row grid for `average_w3 target3`:

```text
portfolio value = 50M / 100M / 300M CNY
ADV participation cap = 3% / 5% / 10%
minimum ADV = 10M / 20M / 50M CNY
hold = 30% / 40%
```

Files:

```text
backtest_results_v9_retention_20260531/avgw3_val_execution_sensitivity_20260531/
backtest_results_v9_retention_20260531/avgw3_test_execution_sensitivity_20260531/
backtest_results_v9_retention_20260531/avgw3_execution_sensitivity_joined_20260531.csv
```

Best validation-first rows:

| portfolio | ADV cap | min ADV | setup | val ann | val Sharpe | val mdd | test ann | test Sharpe | test mdd |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 100M | 3% | 20M | target3/hold30 | 43.92% | 1.246 | 16.33% | 68.42% | 2.587 | 13.29% |
| 50M | 3%-10% | 20M | target3/hold30 | 43.97% | 1.246 | 16.24% | 68.22% | 2.581 | 13.29% |
| 100M | 5%-10% | 20M | target3/hold30 | 43.97% | 1.246 | 16.24% | 68.22% | 2.581 | 13.29% |
| 300M | 10% | 20M | target3/hold30 | 43.84% | 1.244 | 16.36% | 68.22% | 2.581 | 13.29% |
| 100M | 3% | 10M | target3/hold30 | 42.63% | 1.224 | 16.85% | 67.14% | 2.548 | 13.46% |

Sensitivity interpretation:

- `target3/hold30` dominates validation robustness. `hold40` often has better test Sharpe, but validation is consistently weaker.
- The strategy is not very sensitive to ADV participation cap between 3% and 10%, which means most desired trades are already small enough.
- `min ADV = 20M` is the best validation-first setting. `10M` can improve test for `hold40`, but validation weakens; `50M` is too restrictive and cuts useful alpha.
- At 300M CNY, results remain close if ADV cap is at least 5%-10%, but unfilled turnover rises under the strict 3% cap.
- Production-realistic mainline remains `average_w3 target3/hold30` with 20M min ADV. Use 5% ADV cap as a neutral execution assumption; use 3% cap for conservative stress.

### V9 + V10 rank ensemble strict test

Added script:

```text
run/combine_alpha_jsonl.py
```

Combined:

```text
V9 topfocus average_w3 daily rank
V10 seed5590+5591 temporal rank ensemble daily rank
mode = weighted rank_mean
execution = 100M CNY, 5% ADV cap, min ADV 20M, limit blocking, full costs
```

Results:

| combo | split | setup | ann | Sharpe | max drawdown | executed turnover |
|---|---|---|---:|---:|---:|---:|
| V9 only average_w3 | val | target3/hold30 | 43.97% | 1.246 | 16.24% | 0.192 |
| V9 only average_w3 | test | target3/hold30 | 68.22% | 2.581 | 13.29% | 0.240 |
| V9 only average_w3 | val | target3/hold40 | 38.79% | 1.143 | 16.71% | 0.160 |
| V9 only average_w3 | test | target3/hold40 | 70.20% | 2.661 | 13.38% | 0.201 |
| V9 75% + V10 25% | val | target3/hold30 | 33.17% | 1.001 | 18.98% | 0.178 |
| V9 75% + V10 25% | test | target3/hold30 | 56.37% | 2.258 | 12.98% | 0.202 |
| V9 75% + V10 25% | val | target3/hold40 | 34.33% | 1.030 | 18.05% | 0.146 |
| V9 75% + V10 25% | test | target3/hold40 | 58.40% | 2.354 | 12.67% | 0.161 |
| V9 50% + V10 50% | val | target3/hold30 | 28.53% | 0.890 | 20.35% | 0.176 |
| V9 50% + V10 50% | test | target3/hold30 | 52.26% | 2.128 | 12.45% | 0.196 |
| V9 25% + V10 75% | val | target3/hold30 | 23.09% | 0.755 | 22.43% | 0.184 |
| V9 25% + V10 75% | test | target3/hold30 | 45.80% | 1.925 | 12.48% | 0.213 |

Interpretation:

- V9+V10 rank ensemble does **not** improve the strict production metric.
- Adding V10 lowers turnover a bit, but it reduces validation return/Sharpe too much.
- More V10 weight monotonically worsens validation Sharpe, so current V10 temporal signal is not a useful blend component for this long-only strict-execution target.
- Keep V10 as a research branch, not as part of the current best trading combination.

### Execution timing lag stress

Added `--execution-lag` to:

```text
run/backtest_retention_execution_constraints.py
run/sweep_execution_constraints.py
```

Definition:

```text
lag0 = current assumption, signal date -> next tradable close-to-close return
lag1 = one extra trading-day delay before position changes
```

This is a timing conservatism check because raw daily close data does not provide a full open/auction fill model.

Neutral strict execution settings:

```text
100M CNY, 5% ADV cap, min ADV 20M, limit blocking, full costs
```

Lag1 results for current `average_w3 target3`:

| split | setup | ann | Sharpe | max drawdown | executed turnover | avg holding days |
|---|---|---:|---:|---:|---:|---:|
| val | hold30 | 28.86% | 0.928 | 17.65% | 0.193 | 9.10 |
| val | hold40 | 26.85% | 0.885 | 17.70% | 0.161 | 10.91 |
| val | hold50 | 28.91% | 0.935 | 17.75% | 0.136 | 12.98 |
| test | hold30 | 67.29% | 2.573 | 13.66% | 0.240 | 8.06 |
| test | hold40 | 68.59% | 2.620 | 13.64% | 0.200 | 9.81 |
| test | hold50 | 72.97% | 2.757 | 14.08% | 0.163 | 12.27 |

Interpretation:

- Validation 2024 is sensitive to execution timing. Lag1 reduces validation Sharpe from about `1.25` to about `0.93`.
- Test 2025-2026 is not very sensitive to the extra delay; `hold50` even improves in test while lowering turnover.
- The strategy is still promising, but the production claim should explicitly distinguish `lag0` and `lag1`.
- If execution cannot reliably approximate the lag0 assumption, a more conservative operational candidate is `average_w3 target3/hold50`: weaker validation than hold30 in lag0, but best lag1 test and lowest turnover.


### Open-price execution at 1M capital

Added script:

```text
run/backtest_retention_open_execution.py
```

This checks a more practical fill assumption:

```text
capital = 1M CNY
trade at next trading day's open
return mode = open_to_open
ADV cap = 5%
min ADV = 20M CNY
limit blocking = open gap beyond +/-9.5%
full costs unchanged
```

Results for V9 topfocus average_w3 target3:

| split | setup | ann | Sharpe | max drawdown | executed turnover | avg holding days |
|---|---|---:|---:|---:|---:|---:|
| val | hold30 | 25.13% | 0.769 | 23.39% | 0.194 | 9.06 |
| val | hold40 | 29.99% | 0.873 | 21.98% | 0.162 | 10.89 |
| val | hold50 | 35.46% | 0.989 | 20.44% | 0.137 | 12.86 |
| test | hold30 | 61.31% | 2.340 | 15.75% | 0.242 | 8.07 |
| test | hold40 | 64.47% | 2.447 | 15.56% | 0.202 | 9.78 |
| test | hold50 | 73.19% | 2.724 | 15.25% | 0.165 | 12.22 |

Interpretation:

- At 1M CNY, capacity is not the binding issue. Results are almost the same as the 100M run when min ADV is unchanged.
- Open-to-open execution is stricter than close-to-close and reduces validation Sharpe materially.
- hold50 is the best open-execution candidate: highest validation and test Sharpe, lowest turnover, longest holding period.
- The current most realistic candidate for a 1M account is therefore average_w3 target3/hold50, with the caveat that validation Sharpe is just below 1.0.

### 1M min ADV sensitivity

Added summary:

```text
backtest_results_v9_retention_20260531/avgw3_open_to_open_exec1m_minadv_sweep_summary.csv
```

The 1M account does not need a large ADV cap for capacity, but the min ADV filter still acts as a universe-quality filter. I swept `min ADV = 5M / 10M / 20M / 50M` under the same open-to-open execution rules.

Best hold50 comparison:

| min ADV | val ann | val Sharpe | test ann | test Sharpe | note |
|---:|---:|---:|---:|---:|---|
| 5M | 31.77% | 0.921 | 72.39% | 2.709 | Wider pool, but weaker validation. |
| 10M | 32.81% | 0.938 | 72.24% | 2.705 | Still weaker than 20M. |
| 20M | 35.46% | 0.989 | 73.19% | 2.724 | Best validation and test balance. |
| 50M | 32.33% | 0.931 | 65.29% | 2.519 | Too restrictive; more unfilled turnover. |

Conclusion: keep `min ADV = 20M` even for 1M capital. It is not mainly a capacity setting here; it is filtering out lower-quality/liquidity-noisier names.

### Open-to-close comparison

I also tested next-open execution with same-day open-to-close returns, keeping `capital = 1M`, `ADV cap = 5%`, `min ADV = 20M`, and full costs.

| split | setup | ann | Sharpe | max drawdown | executed turnover |
|---|---|---:|---:|---:|---:|
| val | hold30 | 29.05% | 0.974 | 15.74% | 0.194 |
| val | hold40 | 26.57% | 0.916 | 15.40% | 0.162 |
| val | hold50 | 27.04% | 0.933 | 15.19% | 0.137 |
| test | hold30 | 62.97% | 2.733 | 8.11% | 0.242 |
| test | hold40 | 65.99% | 2.843 | 7.90% | 0.202 |
| test | hold50 | 65.14% | 2.802 | 8.16% | 0.165 |

Interpretation:

- `open-to-close` has much lower drawdown and higher test Sharpe, especially `hold40`.
- Validation annual return is weaker than `open-to-open hold50`, so I would not replace the mainline with it yet.
- Use `open-to-open` as the conservative continuous-holding estimate, and keep `open-to-close hold40` as a candidate if the actual execution process targets intraday exposure after next-open fills.

### Hold / max-weight fine grid

Added summary:

```text
backtest_results_v9_retention_20260531/avgw3_open_to_open_exec1m_holdfine_maxweight_summary.csv
```

Grid:

```text
capital = 1M CNY
signal = V9 topfocus average_w3
target = top 3%
hold = 45% / 50% / 55% / 60%
max single-stock weight = 3% / 4% / 5%
execution = next-open, open-to-open, ADV cap 5%, min ADV 20M, full costs
```

`max_weight = 3% / 4% / 5%` produced identical results. With `target = 3%`, the selected basket is already wide enough that the single-name cap does not bind.

| hold | val ann | val Sharpe | test ann | test Sharpe | turnover | note |
|---:|---:|---:|---:|---:|---:|---|
| 45% | 30.82% | 0.896 | 70.27% | 2.635 | 0.148 / 0.184 | Too weak on validation. |
| 50% | 35.46% | 0.989 | 73.19% | 2.724 | 0.137 / 0.165 | Best test return and Sharpe. |
| 55% | 35.18% | 0.990 | 70.24% | 2.635 | 0.125 / 0.149 | Lower turnover, no return gain. |
| 60% | 35.64% | 1.001 | 70.41% | 2.645 | 0.116 / 0.133 | Best validation Sharpe, lowest turnover. |

Interpretation:

- `hold60` is validation-first: it finally lifts validation Sharpe above 1.0 and lowers turnover.
- `hold50` remains return-first: it has the best test annual return and test Sharpe.
- Since `hold50/55/60` are close on validation, the practical choice depends on whether the priority is higher realized return or lower turnover/execution risk.

### Target / hold fine grid

Added summary:

```text
backtest_results_v9_retention_20260531/avgw3_open_to_open_exec1m_targetfine_summary.csv
```

Grid:

```text
capital = 1M CNY
signal = V9 topfocus average_w3
target = 2.5% / 3.0% / 3.5% / 4.0%
hold = 50% / 60%
max single-stock weight = 3%
execution = next-open, open-to-open, ADV cap 5%, min ADV 20M, full costs
```

Results:

| target | hold | val ann | val Sharpe | test ann | test Sharpe | note |
|---:|---:|---:|---:|---:|---:|---|
| 2.5% | 50% | 32.86% | 0.938 | 69.52% | 2.597 | Too narrow, weaker validation. |
| 2.5% | 60% | 33.42% | 0.961 | 68.96% | 2.562 | Lower turnover but weaker. |
| 3.0% | 50% | 35.46% | 0.989 | 73.19% | 2.724 | Best test return and Sharpe. |
| 3.0% | 60% | 35.64% | 1.001 | 70.41% | 2.645 | Lower turnover, cleaner validation. |
| 3.5% | 50% | 34.16% | 0.966 | 70.12% | 2.656 | Worse than 3.0/50. |
| 3.5% | 60% | 36.65% | 1.024 | 69.63% | 2.637 | Best validation annual return and Sharpe. |
| 4.0% | 50% | 34.49% | 0.975 | 68.05% | 2.606 | Lower drawdown but weaker return. |
| 4.0% | 60% | 34.93% | 0.989 | 69.75% | 2.661 | Not enough validation gain. |

Interpretation:

- `target3.5/hold60` is the new validation-first candidate.
- `target3/hold50` remains the return-first candidate and still wins on test annual return and test Sharpe.
- `target2.5` is too narrow, while `target4.0` dilutes the alpha too much.
- The useful region is narrow: `target3.0-3.5` and `hold50-60`.

### Yearly candidate split

Added summary:

```text
backtest_results_v9_retention_20260531/avgw3_open_to_open_exec1m_targetfine_yearly_candidates.csv
```

I compared the two current 1M open-to-open candidates by year:

| candidate | period | ann | Sharpe | max drawdown | sum return |
|---|---|---:|---:|---:|---:|
| target3/hold50 | 2024 validation core | 39.00% | 1.056 | 20.44% | 38.36% |
| target3.5/hold60 | 2024 validation core | 40.41% | 1.096 | 20.21% | 39.06% |
| target3/hold50 | 2025 test | 67.49% | 2.517 | 15.25% | 51.78% |
| target3.5/hold60 | 2025 test | 65.35% | 2.465 | 15.70% | 50.53% |
| target3/hold50 | 2026 test | 92.16% | 3.435 | 7.85% | 20.84% |
| target3.5/hold60 | 2026 test | 83.64% | 3.234 | 7.78% | 19.42% |

Interpretation:

- `target3.5/hold60` wins validation 2024 cleanly, with slightly higher return, Sharpe, and lower drawdown.
- `target3/hold50` wins both 2025 and 2026 test periods on annual return and Sharpe.
- The small 2025 rows inside the validation split only contain 2 trading days and should not drive selection.
- If we select strictly by validation, use `target3.5/hold60`; if we require out-of-sample test dominance, keep `target3/hold50`.

### Raw + average_w3 rank blend

Added summary:

```text
backtest_results_v9_retention_20260531/raw_avgw3_rankblend_open_to_open_exec1m_summary.csv
```

I tested weighted rank blends of V9 raw and V9 average_w3:

```text
raw25 / average_w3 75
raw50 / average_w3 50
raw75 / average_w3 25
target = 3.0% / 3.5%
hold = 50% / 60%
capital = 1M CNY
execution = next-open, open-to-open, ADV cap 5%, min ADV 20M, full costs
```

Best rows:

| blend | setup | val ann | val Sharpe | test ann | test Sharpe | note |
|---|---|---:|---:|---:|---:|---|
| raw25/avg75 | target3/hold60 | 30.26% | 0.893 | 71.17% | 2.644 | Best blend, still below pure average_w3 validation. |
| raw25/avg75 | target3.5/hold60 | 28.95% | 0.866 | 68.79% | 2.611 | Worse than pure average_w3. |
| raw50/avg50 | target3.5/hold60 | 20.49% | 0.684 | 59.05% | 2.283 | Raw weight too high. |
| raw75/avg25 | target3/hold60 | 15.48% | 0.569 | 58.21% | 2.217 | Rejected. |

Interpretation:

- Raw improves neither validation nor test under the 1M open-to-open execution engine.
- The higher the raw weight, the worse the validation result becomes.
- Keep pure average_w3 as the main signal; raw remains an unconstrained/backtest artifact, not a production signal.

### Market timing / exposure filter

Added summaries:

```text
backtest_results_v9_retention_20260531/avgw3_open_to_open_exec1m_market_mode_summary.csv
backtest_results_v9_retention_20260531/avgw3_open_to_open_exec1m_dynamic_min_sweep_summary.csv
```

I compared market exposure policies on the same 1M open-to-open engine:

```text
none    = always full exposure
legacy  = hard 0.7 / 0.3 cuts based on MA60 and 6M index drawdown
dynamic = continuous score from MA60, 20D momentum, breadth, and volatility
```

Representative rows:

| market | setup | val ann | val Sharpe | val mdd | test ann | test Sharpe | test mdd | note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| none | target3/hold50 | 10.82% | 0.453 | 35.94% | 78.09% | 2.377 | 19.12% | Highest test annual return, unacceptable validation risk. |
| legacy | target3/hold50 | 35.46% | 0.989 | 20.44% | 73.19% | 2.724 | 15.25% | Best return-first production candidate. |
| legacy | target3.5/hold60 | 36.65% | 1.024 | 20.21% | 69.63% | 2.637 | 15.70% | Best validation-first legacy candidate. |
| dynamic min0.2 | target3/hold50 | 31.88% | 1.015 | 16.18% | 57.35% | 3.044 | 9.91% | Best risk-adjusted/test Sharpe candidate. |
| dynamic min0.3 | target3/hold50 | 32.10% | 1.001 | 16.53% | 60.04% | 2.947 | 11.11% | Better return/risk compromise than min0.2. |
| dynamic min0.6 | target3/hold50 | 24.36% | 0.762 | 23.11% | 68.15% | 2.672 | 14.64% | More exposure, but validation deteriorates. |

Interpretation:

- `none` confirms the alpha has strong upside, but it fails validation risk control.
- `dynamic min0.2` is the best Sharpe / drawdown policy: test Sharpe above 3 and max drawdown below 10%.
- `dynamic min0.3` is a useful compromise if annual return matters more than pure Sharpe.
- `legacy target3/hold50` remains the return-first mainline; `dynamic target3/hold50` becomes the risk-first mainline.

### Legacy vs dynamic yearly split

Added summary:

```text
backtest_results_v9_retention_20260531/avgw3_open_to_open_exec1m_legacy_dynamic_yearly_candidates.csv
```

Compared `target3/hold50` under the three useful exposure policies:

| strategy | period | ann | Sharpe | max drawdown | sum return |
|---|---|---:|---:|---:|---:|
| legacy | 2024 validation core | 39.00% | 1.056 | 20.44% | 38.36% |
| dynamic min0.2 | 2024 validation core | 34.31% | 1.069 | 16.18% | 33.24% |
| dynamic min0.3 | 2024 validation core | 34.80% | 1.060 | 16.53% | 33.86% |
| legacy | 2025 test | 67.49% | 2.517 | 15.25% | 51.78% |
| dynamic min0.2 | 2025 test | 51.81% | 2.775 | 9.91% | 41.28% |
| dynamic min0.3 | 2025 test | 54.83% | 2.707 | 11.11% | 43.35% |
| legacy | 2026 test | 92.16% | 3.435 | 7.85% | 20.84% |
| dynamic min0.2 | 2026 test | 75.84% | 3.938 | 4.88% | 17.82% |
| dynamic min0.3 | 2026 test | 77.37% | 3.749 | 5.66% | 18.14% |

Interpretation:

- The dynamic policies are not a one-year artifact; they improve Sharpe and reduce drawdown in 2024 validation, 2025 test, and 2026 test.
- Legacy wins annual return in all three core periods.
- Dynamic min0.2 is the cleanest risk-adjusted version.
- Dynamic min0.3 is the practical compromise when we want slightly more return while preserving most of the drawdown benefit.

### Legacy vs dynamic monthly stability

Added summaries:

```text
backtest_results_v9_retention_20260531/avgw3_open_to_open_exec1m_legacy_dynamic_monthly_candidates.csv
backtest_results_v9_retention_20260531/avgw3_open_to_open_exec1m_legacy_dynamic_monthly_stats.csv
```

Monthly stats for `target3/hold50`:

| strategy | split | months | win months | win rate | avg month | best month | worst month | total sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | validation | 13 | 6 | 46.15% | 2.78% | 16.38% | -8.66% | 36.15% |
| dynamic min0.2 | validation | 13 | 6 | 46.15% | 2.44% | 14.36% | -6.69% | 31.72% |
| dynamic min0.3 | validation | 13 | 6 | 46.15% | 2.47% | 14.82% | -6.86% | 32.16% |
| legacy | test | 17 | 14 | 82.35% | 4.27% | 10.74% | -2.75% | 72.62% |
| dynamic min0.2 | test | 17 | 13 | 76.47% | 3.48% | 9.33% | -1.55% | 59.10% |
| dynamic min0.3 | test | 17 | 13 | 76.47% | 3.62% | 9.51% | -1.93% | 61.49% |

Interpretation:

- Dynamic timing does not improve monthly win rate; legacy actually wins more months in test.
- Dynamic timing improves risk by reducing the worst month: validation worst month improves from `-8.66%` to about `-6.7%`, and test worst month improves from `-2.75%` to `-1.55%/-1.93%`.
- This confirms dynamic is a drawdown-control overlay, not a return enhancer.
- For a 1M account, choose legacy when maximizing growth, and dynamic min0.3 when the priority is avoiding deep monthly losses.

## Current Best Choices

### Final 1M decision matrix

Added summary:

```text
backtest_results_v9_retention_20260531/current_best_1m_strategy_decision_matrix.csv
backtest_results_v9_retention_20260531/current_best_1m_2xcost_sensitivity.csv
```

Current ranking for a 1M CNY long-only account under next-open open-to-open execution, 5% ADV cap, 20M min ADV, and full explicit costs:

| rank | profile | setup | val ann | val Sharpe | val mdd | test ann | test Sharpe | test mdd | test worst month | use when |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Return-first | V9 average_w3, target3/hold50, legacy | 35.46% | 0.989 | 20.44% | 73.19% | 2.724 | 15.25% | -2.75% | Maximize growth and can tolerate deeper drawdown. |
| 2 | Balanced live | V9 average_w3, target3/hold50, dynamic min0.3 | 32.10% | 1.001 | 16.53% | 60.04% | 2.947 | 11.11% | -1.93% | Preferred practical candidate. |
| 3 | Risk-first | V9 average_w3, target3/hold50, dynamic min0.2 | 31.88% | 1.015 | 16.18% | 57.35% | 3.044 | 9.91% | -1.55% | Maximize Sharpe / minimize drawdown. |
| 4 | Validation-first | V9 average_w3, target3.5/hold60, legacy | 36.65% | 1.024 | 20.21% | 69.63% | 2.637 | 15.70% | n/a | Strictly select by validation metrics. |

My current practical recommendation is rank 2: `target3/hold50 + dynamic min0.3`. It gives up about 13 percentage points of test annual return versus legacy, but improves test Sharpe from `2.724` to `2.947`, reduces test max drawdown from `15.25%` to `11.11%`, and improves the worst test month from `-2.75%` to `-1.93%`.

2x explicit-cost stress for the three final `target3/hold50` candidates:

| strategy | val ann | val Sharpe | val mdd | test ann | test Sharpe | test mdd |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 31.59% | 0.913 | 21.12% | 67.27% | 2.557 | 15.42% |
| dynamic min0.2 | 28.58% | 0.936 | 16.27% | 52.51% | 2.838 | 10.08% |
| dynamic min0.3 | 28.71% | 0.923 | 16.93% | 55.06% | 2.754 | 11.27% |

Cost-stress interpretation:

- All three final candidates remain viable under 2x costs.
- Legacy remains the annual-return leader, but its validation Sharpe falls below 1 and drawdown stays higher.
- Dynamic min0.3 remains the practical compromise: still above 55% test annualized under 2x costs, with materially lower drawdown than legacy.

Return-seeking:

```text
V9 topfocus raw + target3/hold25 + legacy
validation ann = 109.50%, Sharpe = 2.128
test ann = 146.25%, Sharpe = 4.218
note = unconstrained same-engine result, not production-realistic under strict execution
```

Balanced mainline:

```text
V9 topfocus average_w3 + target3/hold30 + legacy + strict execution constraints
validation ann = 43.97%, Sharpe = 1.246
test ann = 68.22%, Sharpe = 2.581
lag1 stress: validation ann = 28.86%, Sharpe = 0.928; test ann = 67.29%, Sharpe = 2.573
```

Cost-robust practical version:

```text
V9 topfocus average_w3 + target3/hold40 + legacy + strict execution constraints
normal cost: validation ann = 38.79%, Sharpe = 1.143; test ann = 70.20%, Sharpe = 2.661
2x cost: validation ann = 34.17%, Sharpe = 1.043; test ann = 63.11%, Sharpe = 2.456
lag1 stress: validation ann = 26.85%, Sharpe = 0.885; test ann = 68.59%, Sharpe = 2.620
```

Best open-to-close execution variant:

```text
V9 topfocus average_w3 + target3/hold40 + legacy + next-open execution, open-to-close returns
capital = 1M CNY, ADV cap = 5%, min ADV = 20M, full costs
validation ann = 26.57%, Sharpe = 0.916
test ann = 65.99%, Sharpe = 2.843
note = lower drawdown and strongest test Sharpe, but weaker validation than open-to-open hold50
```

Most realistic 1M open-execution candidate:

```text
V9 topfocus average_w3 + target3/hold50 + legacy + open-to-open execution constraints
capital = 1M CNY, ADV cap = 5%, min ADV = 20M, full costs
open execution: validation ann = 35.46%, Sharpe = 0.989; test ann = 73.19%, Sharpe = 2.724
lag1 stress: validation ann = 28.91%, Sharpe = 0.935; test ann = 72.97%, Sharpe = 2.757
```

Validation-first 1M open-execution candidate:

```text
V9 topfocus average_w3 + target3.5/hold60 + legacy + open-to-open execution constraints
capital = 1M CNY, ADV cap = 5%, min ADV = 20M, full costs
validation ann = 36.65%, Sharpe = 1.024
test ann = 69.63%, Sharpe = 2.637
note = best validation annual return and Sharpe, but lower test return than target3/hold50
```

Risk-first 1M open-execution candidate:

```text
V9 topfocus average_w3 + target3/hold50 + dynamic market timing + open-to-open execution constraints
capital = 1M CNY, ADV cap = 5%, min ADV = 20M, full costs
dynamic min_mult = 0.20, max_mult = 1.00
validation ann = 31.88%, Sharpe = 1.015, max drawdown = 16.18%
test ann = 57.35%, Sharpe = 3.044, max drawdown = 9.91%
note = lower annual return than legacy, but best risk-adjusted profile and much lower drawdown
```

Balanced risk/return dynamic candidate:

```text
V9 topfocus average_w3 + target3/hold50 + dynamic market timing + open-to-open execution constraints
capital = 1M CNY, ADV cap = 5%, min ADV = 20M, full costs
dynamic min_mult = 0.30, max_mult = 1.00
validation ann = 32.10%, Sharpe = 1.001, max drawdown = 16.53%
test ann = 60.04%, Sharpe = 2.947, max drawdown = 11.11%
note = slightly more return than min0.2 while keeping validation Sharpe around 1
```

Lower model-risk:

```text
V10 seed5590+5591 rank ensemble + target30/hold80 + legacy
validation ann = 31.94%, Sharpe = 0.986
test ann = 46.00%, Sharpe = 2.109
```

Lower turnover:

```text
V10 seed5590+5591 rank ensemble + target30/hold90 + legacy
validation ann = 30.09%, Sharpe = 0.953
test ann = 45.23%, Sharpe = 2.117
```

## Next Optimization

The next most valuable work is:

1. Add exact ST/new-listing/suspension filters from a metadata source, because raw daily CSVs do not contain those fields.
2. Test execution sensitivity:
   - partial-fill policy variants,
   - open-to-close versus open-to-open marking,
   - 1M account with lower max single-stock weight if concentration risk matters.
3. Improve V10 before using it in ensemble:
   - validate V10 on strict execution, not only alpha/top bucket metrics,
   - test V10 as a risk overlay rather than direct rank blend,
   - only add it back if it improves validation Sharpe under strict execution.
4. Keep `dynamic` as a drawdown-control variant only; legacy remains better for return/Sharpe.
