# 2026-05-30 trade strategy tests report

## Baseline

Current model checkpoint:

`checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt`

Original best strategy before this batch:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| raw V9, simple_long, top_frac=0.045, rebalance_freq=5 | 34.08% | 1.395 | 18.63% | 24.72% | 1.428 | 14.96% |

## 1. Three-day average alpha

Output:

`backtest_results_exp_v9_persistent_refine_topfocus_w005_topic_20260530/v9_persistent_refine_partial_summary.csv`

Best completed rows:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| average_w3, top4.0% | 34.64% | 1.426 | 18.30% | 26.12% | 1.431 | 15.76% |
| average_w3, top4.5% | 34.59% | 1.426 | 18.63% | 26.27% | 1.442 | 16.11% |
| average_w3, top5.0% | 34.52% | 1.426 | 18.97% | 26.29% | 1.448 | 16.01% |

Notes:

- Three-day averaging improves over raw V9.
- `alpha_vol_power` hurts raw return heavily. Example: `alpha_vol_power=0.25` drops raw ann to about 28.3%-28.7%; `0.50` drops to about 25.4%-25.6%.
- Volatility filter rows matched the unfiltered rows in this simple_long branch, so they are not useful evidence.

## 2. Rank exit with rebalance_freq=5

Output:

`backtest_results_exp_v9_rank_exit_topfocus_w005_topic_20260530/v9_rank_exit_summary.csv`

Best rows:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| top5 buy, exit if below top15 | 34.76% | 1.433 | 18.85% | 26.55% | 1.460 | 15.95% |
| top4.5 buy, exit if below top7.5 | 34.67% | 1.429 | 18.51% | 26.36% | 1.446 | 16.00% |
| top5 buy, exit if below top10 | 34.53% | 1.427 | 18.88% | 26.32% | 1.450 | 15.95% |

Interpretation:

- Rank-based exit helps modestly.
- The best `rebalance_freq=5` candidate is `average_w3 + top5 buy + top15 exit`.
- Too strict an exit is not always better; the signal has some short-term recovery value.

## 3. Price stop-loss

Output:

`backtest_results_exp_v9_rank_exit_stoploss_topfocus_w005_topic_20260530/v9_rank_exit_stoploss_summary.csv`

Best no-stop row:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| top5 buy, exit below top15, no stop | 34.76% | 1.433 | 18.85% | 26.55% | 1.460 | 15.95% |

Stop-loss rows:

| stop loss | raw ann range | raw mdd range | comment |
|---|---:|---:|---|
| -5% | about 28.69%-28.78% | about 25.94%-26.05% | hurts return and drawdown |
| -8% | about 28.83%-29.02% | about 24.85%-24.93% | also hurts |

Interpretation:

- Pure price stop-loss is not suitable here.
- It cuts positions during normal alpha noise and does not improve drawdown.

## 4. Rebalance frequency and rank exit

Output:

`backtest_results_exp_v9_rebalance_rank_exit_topfocus_w005_topic_20260530/v9_rebalance_rank_exit_summary.csv`

Best rows:

| rebalance_freq | exit line | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | top15 | 36.46% | 1.484 | 15.86% | 30.61% | 1.628 | 11.78% |
| 4 | top12.5 | 36.40% | 1.482 | 15.80% | 30.58% | 1.626 | 11.69% |
| 3 | top12.5 | 36.45% | 1.505 | 17.20% | 27.89% | 1.546 | 12.46% |
| 5 | top15 | 34.76% | 1.433 | 18.85% | 26.55% | 1.460 | 15.95% |

Interpretation:

- Checking the signal more often helps a lot.
- `rebalance_freq=4` gives the best neutral Sharpe and drawdown.
- `rebalance_freq=3` gives strong raw Sharpe, but neutral metrics are weaker than `rebalance_freq=4`.
- Best current stable candidate from completed full-grid tests:

`average_w3 + top5 buy + top15 exit + rebalance_freq=4`

## 5. Daily rank-exit partial test

Output:

`backtest_results_exp_v9_daily_rank_exit_topfocus_w005_topic_20260530/v9_daily_rank_exit_partial_summary.csv`

This run was interrupted after user pointed out a strategy-design issue. Completed partial rows:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| daily, top4.5 buy, exit top15 | 50.07% | 1.873 | 14.78% | 40.01% | 2.083 | 11.36% |
| daily, top4.5 buy, exit top4.5 | 50.01% | 1.871 | 14.70% | 39.97% | 2.083 | 11.38% |
| daily, top5 buy, exit top5 | 48.81% | 1.843 | 14.71% | 38.87% | 2.044 | 11.45% |

Important caveat:

- These numbers are promising but should not be treated as final production evidence.
- The current engine still carries `future_len=5` semantics internally, even when `rebalance_freq=1`.
- The user is right: a cleaner implementation should rank every day and manage positions by daily signal state, not by fixed five-day holding logic.

## Current conclusions

1. The best fully completed candidate is:

`average_w3 + simple_long top5 + rank exit top15 + rebalance_freq=4`

Metrics:

`raw ann 36.46%, raw sharpe 1.484, raw mdd 15.86%, neutral ann 30.61%, neutral sharpe 1.628, neutral mdd 11.78%`

2. Do not use pure price stop-loss for this alpha.

3. Shorter signal checking helps more than extra model loss.

4. The next correct step is a dedicated daily trading-policy backtest:

- rank every day,
- buy from top pool,
- hold existing positions only while their daily rank/alpha state remains healthy,
- sell immediately when the learned or rule-based policy says the signal has deteriorated,
- remove fixed five-day holding assumptions from the strategy layer.

5. Longer-term direction:

Build a no-hand-tuned-threshold trading policy model. It should learn `buy / hold / sell` from rank change, alpha change, holding PnL, holding age, volatility, liquidity, and market state.
