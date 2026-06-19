# Backtest Results Summary - 2026-05-31

This file consolidates the backtest and policy-model results generated around
2026-05-30. It separates reliable comparisons from exploratory or audited-invalid
results.

## Executive Summary

Best reliable fixed-window/rank-exit candidate:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| average_w3 + top5 buy + exit below top15 + rebalance_freq=4 | 36.46% | 1.484 | 15.86% | 30.61% | 1.628 | 11.78% |

Best cleaner daily-state result before full cost expansion:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd | avg turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| daily alpha top5 baseline | 60.84% | 2.130 | 14.42% | 51.08% | 2.472 | 12.17% | 0.975 |
| v2 retention-first policy | 61.58% | 2.147 | 14.88% | 51.80% | 2.498 | 12.17% | 0.870 |

Latest explicit-cost switch-value backtest result:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd | avg turnover | total trade cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| daily alpha topk baseline, full explicit costs | 24.68% | 1.052 | 25.65% | 17.04% | 0.996 | 19.54% | 0.975 | 0.799 |
| switch value retention-first, full explicit costs | 19.68% | 0.932 | 21.60% | 12.52% | 0.841 | 14.79% | 0.632 | 0.518 |

Current interpretation:

- The alpha signal is usable, but raw top-pool quality is not strong enough to ignore turnover/cost.
- Daily-state trading logic improved apparent performance substantially before full explicit-cost accounting.
- After adding full explicit costs, high turnover severely compresses returns.
- The value/switch model currently reduces turnover but also sacrifices too much alpha return.
- The next alpha-side improvement should be temporal modeling and stronger stable top-pool quality, not only more trading-policy complexity.

## 1. Original V9 Long-Only Baseline

Source:

- `backtest_result_snapshots/20260530_trade_strategy_tests_report.md`

Reference checkpoint:

- `checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt`

Original best strategy before the later daily-state policy tests:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| raw V9, simple_long, top_frac=0.045, rebalance_freq=5 | 34.08% | 1.395 | 18.63% | 24.72% | 1.428 | 14.96% |

## 2. Signal Smoothing And Rank Exit

Source:

- `backtest_results_exp_v9_persistent_refine_topfocus_w005_topic_20260530/v9_persistent_refine_partial_summary.csv`
- `backtest_results_exp_v9_rank_exit_topfocus_w005_topic_20260530/v9_rank_exit_summary.csv`
- `backtest_results_exp_v9_rebalance_rank_exit_topfocus_w005_topic_20260530/v9_rebalance_rank_exit_summary.csv`

Three-day average alpha improved the baseline slightly:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| average_w3, top4.0% | 34.64% | 1.426 | 18.30% | 26.12% | 1.431 | 15.76% |
| average_w3, top4.5% | 34.59% | 1.426 | 18.63% | 26.27% | 1.442 | 16.11% |
| average_w3, top5.0% | 34.52% | 1.426 | 18.97% | 26.29% | 1.448 | 16.01% |

Rank exit with `rebalance_freq=5` helped modestly:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| top5 buy, exit if below top15 | 34.76% | 1.433 | 18.85% | 26.55% | 1.460 | 15.95% |
| top4.5 buy, exit if below top7.5 | 34.67% | 1.429 | 18.51% | 26.36% | 1.446 | 16.00% |
| top5 buy, exit if below top10 | 34.53% | 1.427 | 18.88% | 26.32% | 1.450 | 15.95% |

The strongest completed rank-exit grid was `rebalance_freq=4`:

| rebalance_freq | exit line | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | top15 | 36.46% | 1.484 | 15.86% | 30.61% | 1.628 | 11.78% |
| 4 | top12.5 | 36.40% | 1.482 | 15.80% | 30.58% | 1.626 | 11.69% |
| 3 | top12.5 | 36.45% | 1.505 | 17.20% | 27.89% | 1.546 | 12.46% |
| 5 | top15 | 34.76% | 1.433 | 18.85% | 26.55% | 1.460 | 15.95% |

Conclusion:

- Checking signal deterioration more frequently helped.
- `rebalance_freq=4` was the best stable fixed-window/rank-exit setting.
- Too strict an exit did not consistently help.

## 3. Price Stop-Loss

Source:

- `backtest_results_exp_v9_rank_exit_stoploss_topfocus_w005_topic_20260530/v9_rank_exit_stoploss_summary.csv`

Best no-stop row:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| top5 buy, exit below top15, no stop | 34.76% | 1.433 | 18.85% | 26.55% | 1.460 | 15.95% |

Stop-loss rows were worse:

| stop loss | raw ann range | raw mdd range | comment |
|---|---:|---:|---|
| -5% | about 28.69%-28.78% | about 25.94%-26.05% | hurts return and drawdown |
| -8% | about 28.83%-29.02% | about 24.85%-24.93% | also hurts |

Conclusion:

- Pure price stop-loss is not suitable for this alpha.
- It cuts normal alpha noise and does not improve drawdown.

## 4. Daily Rank-Exit Partial Test

Source:

- `backtest_results_exp_v9_daily_rank_exit_topfocus_w005_topic_20260530/v9_daily_rank_exit_partial_summary.csv`

Completed partial rows:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| daily, top4.5 buy, exit top15 | 50.07% | 1.873 | 14.78% | 40.01% | 2.083 | 11.36% |
| daily, top4.5 buy, exit top4.5 | 50.01% | 1.871 | 14.70% | 39.97% | 2.083 | 11.38% |
| daily, top5 buy, exit top5 | 48.81% | 1.843 | 14.71% | 38.87% | 2.044 | 11.45% |

Important caveat:

- This run was interrupted after identifying a strategy-design issue.
- It still carried fixed-window semantics internally.
- It should be treated as directionally useful evidence, not final production evidence.

## 5. Backtest Code Audit

Source:

- `backtest_result_snapshots/20260530_backtest_code_audit.md`

High-severity findings:

1. `trade_policy_v1` mixed incompatible scores.
   - Existing holdings were scored by `hold_prob`.
   - Non-held candidates were scored by alpha rank.
   - The resulting top-K selection was not a true hold/sell policy.

2. Initial and gap-period transaction costs could be trimmed away in older engines.
   - Execution cost was charged before weights became active.
   - Later trimming could drop those cost-only days.

Medium-severity findings:

- Fixed-window engine remains correct only for fixed holding tests.
- Daily hold/sell logic needs a dedicated daily state engine.
- The hold/sell label is usable for retention-first policy, not as a full buy model.

Invalid/not comparable result:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd | avg turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| trade_policy_v1 | 93.34% | 2.654 | 14.62% | 81.36% | 3.294 | 11.93% | 1.607 |

Conclusion:

- `trade_policy_v1` should not be used for decision-making.

## 6. Trade Policy V2: Retention-First Daily State Engine

Source:

- `backtest_result_snapshots/20260530_trade_policy_v2_report.md`
- `models_trade_policy_20260530/hold_sell_lgb_v2_threshold/metrics.json`
- `backtest_results_trade_policy_v2_20260530/trade_policy_v2_summary.csv`
- `backtest_results_trade_policy_v2_alpha_baseline_20260530/trade_policy_v2_summary.csv`

Model threshold:

- Learned threshold: `hold_prob >= 0.49` means keep.
- Threshold selected on train split by balanced accuracy.

Model quality:

| split | rows | AUC | accuracy | precision hold | recall hold | edge mean |
|---|---:|---:|---:|---:|---:|---:|
| train | 415,691 | 0.5972 | 0.5685 | 0.5661 | 0.4988 | 0.0964 |
| val | 190,754 | 0.5264 | 0.5169 | 0.5177 | 0.4169 | 0.1005 |

Threshold diagnostics:

| split | balanced accuracy | keep rate | kept edge mean | sold edge mean |
|---|---:|---:|---:|---:|
| train | 0.5680 | 0.4767 | 0.2460 | -0.0399 |
| val | 0.5174 | 0.4489 | 0.1376 | 0.0702 |

Backtest comparison before full explicit-cost expansion:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd | avg turnover | impact cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| daily alpha top5 baseline | 60.84% | 2.130 | 14.42% | 51.08% | 2.472 | 12.17% | 0.975 | 0.00347 |
| v2 retention-first policy | 61.58% | 2.147 | 14.88% | 51.80% | 2.498 | 12.17% | 0.870 | 0.00306 |

Conclusion:

- V2 policy has weak but positive hold/sell signal.
- It improved turnover and Sharpe slightly.
- Most performance came from the cleaner daily engine, not from the policy model.

## 7. Full Explicit-Cost Switch-Value Policy Backtest

Source:

- `backtest_result_snapshots/20260530_switch_value_fixed_report.md`
- `backtest_results_switch_value_20260530/switch_value/switch_value_summary.csv`
- `backtest_results_switch_value_20260530/alpha_baseline/switch_value_summary.csv`
- `switch_value_models_20260530_fixed/switch_edge_lgb_h5/metrics.json`
- `switch_value_data_20260530_fixed/switch_value_dataset_summary.csv`

Cost assumptions reflected in this stage:

- Commission: 0.01% each side after later adjustment.
- Stamp tax: sell side.
- Slippage/spread included.
- Average full switch cost in dataset: about 0.21%.

Dataset:

| split | rows | h1 net edge mean | h3 net edge mean | h5 net edge mean | h5 success | cost mean |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,242,012 | 0.1932 | 0.1166 | 0.1083 | 0.5218 | 0.0021 |
| val | 571,269 | 0.2561 | 0.1258 | 0.0766 | 0.5075 | 0.0021 |

H5 value model:

| split | rows | target mean | Spearman IC | success | pred positive true edge | pred positive success |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,242,012 | 0.1083 | 0.3676 | 0.5218 | 0.4773 | 0.6184 |
| val | 571,269 | 0.0766 | 0.0347 | 0.5075 | 0.1125 | 0.5175 |

Validation bucket note:

- Highest predicted bucket: true h5 net edge `0.2945`, success `0.5401`.
- The highest bucket is useful.
- Ranking is not monotonic because the lowest bucket is also anomalously positive.

Full explicit-cost backtest:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd | avg turnover | avg holding days | total explicit cost | total trade cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daily alpha topk baseline | 24.68% | 1.052 | 25.65% | 17.04% | 0.996 | 19.54% | 0.975 | 1.99 | 0.795 | 0.799 |
| switch value retention-first | 19.68% | 0.932 | 21.60% | 12.52% | 0.841 | 14.79% | 0.632 | 3.39 | 0.515 | 0.518 |

Conclusion:

- Full explicit costs change the picture materially.
- Reducing turnover helps cost and drawdown, but the current switch-value policy gives up too much alpha.
- The value model is not yet strong enough to replace alpha-top selection.

## 8. What The Results Mean

Stable conclusions:

1. The alpha model has positive long-only signal.
2. Signal smoothing and rank-exit logic help more than simple price stop-loss.
3. Pure price stop-loss should not be used.
4. Daily state management is the right strategy-engine direction.
5. Transaction costs are now the central bottleneck because average turnover is high.
6. Current hold/sell and switch-value models are weakly useful, but not strong enough as standalone trading policies.

Unstable or not-final conclusions:

1. The 93% `trade_policy_v1` result is invalid/not comparable.
2. The 60%+ V2 daily engine result is promising but was before the later full explicit-cost accounting.
3. The latest full-cost result around 20%-25% is more conservative and likely closer to realistic implementation assumptions.

## 9. Recommended Next Steps

Priority 1: Improve alpha top-pool quality.

- Build and train the temporal-tower model.
- Keep loss mostly unchanged first, with horizon weights `(0.10, 0.30, 0.30, 0.30)`.
- Compare against V9 under the same daily/full-cost engine.

Priority 2: Reduce turnover without giving up too much alpha.

- Use retention-first policy only as a conservative filter.
- Do not let switch-value model force many daily replacements yet.
- Add average holding-day and turnover constraints as evaluation metrics, not hand-tuned thresholds at first.

Priority 3: Re-run consistent cost-aware comparisons.

All future comparisons should report:

- raw annualized return,
- neutral annualized return,
- Sharpe,
- max drawdown,
- average turnover,
- average holding days,
- total commission,
- total stamp tax,
- total slippage,
- total trade cost,
- yearly performance.

Priority 4: Treat model selection as multi-metric.

Do not select only by `topret` or only by `topic`.

Recommended validation dashboard:

- alpha IC,
- h3/h5/h7 IC,
- topret top5/top10,
- top-bottom spread,
- turnover,
- full-cost annualized return,
- yearly consistency.
