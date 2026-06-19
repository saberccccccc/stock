# 2026-05-30 trade policy v2 report

## What changed

Implemented a retention-first daily hold/sell policy backtest.

Files:

- `run/train_trade_policy.py`
- `run/backtest_trade_policy_v2.py`

New model output:

- `models_trade_policy_20260530/hold_sell_lgb_v2_threshold/hold_sell_policy_model.pkl`
- `models_trade_policy_20260530/hold_sell_lgb_v2_threshold/metrics.json`
- `models_trade_policy_20260530/hold_sell_lgb_v2_threshold/threshold_report_train.csv`

New backtest outputs:

- `backtest_results_trade_policy_v2_20260530/trade_policy_v2_summary.csv`
- `backtest_results_trade_policy_v2_alpha_baseline_20260530/trade_policy_v2_summary.csv`

## Model threshold

The sell threshold is learned from the train split by grid-searching hold probability and maximizing balanced accuracy.

Learned threshold:

`hold_prob >= 0.49` means keep; below that means sell.

Train at threshold:

| metric | value |
|---|---:|
| accuracy | 0.5685 |
| balanced accuracy | 0.5680 |
| precision hold | 0.5597 |
| recall hold | 0.5463 |
| keep rate | 0.4767 |
| kept edge mean | 0.2460 |
| sold edge mean | -0.0399 |

Validation at threshold:

| metric | value |
|---|---:|
| accuracy | 0.5176 |
| balanced accuracy | 0.5174 |
| precision hold | 0.5166 |
| recall hold | 0.4664 |
| keep rate | 0.4489 |
| kept edge mean | 0.1376 |
| sold edge mean | 0.0702 |

Interpretation:

The model has a weak but positive hold/sell signal. It separates better holdings from worse holdings, but the out-of-sample spread is small.

## Backtest comparison

Both rows use the same corrected daily state engine:

- signal at current close,
- execute at next close,
- new weights active from the following day,
- execution cost retained even if paid before the first active weight day.

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd | avg turnover | total impact cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| daily alpha top5 baseline | 60.84% | 2.130 | 14.42% | 51.08% | 2.472 | 12.17% | 0.975 | 0.00347 |
| v2 retention-first policy | 61.58% | 2.147 | 14.88% | 51.80% | 2.498 | 12.17% | 0.870 | 0.00306 |

## Key conclusion

The large jump from earlier 30%-36% backtests is mostly from the daily trading engine itself, not from the hold/sell policy model.

The policy adds:

- about `+0.73%` raw annualized return,
- about `+0.71%` neutral annualized return,
- lower average turnover: `0.975 -> 0.870`,
- slightly better Sharpe,
- slightly worse raw drawdown.

So the retention-first policy is directionally useful, but the current model is not the main source of performance.

## Important caveat

This daily engine is cleaner than the old fixed-window engine for daily rank/hold/sell ideas, but the result is high enough that it should be treated as a research candidate, not final production evidence.

Next checks:

1. Add stricter limit-up/limit-down trade blocking to the daily engine.
2. Add ST/listing-age filters to the daily engine.
3. Compare transaction cost assumptions at higher portfolio values.
4. Run top fractions `0.04, 0.045, 0.05, 0.055, 0.06`.
5. Run a daily engine with alpha smoothing `average_w3`.
