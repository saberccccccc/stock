# V10 temporal probe report

## Setup

- checkpoint: `checkpoints_exp\temporal_cross_alpha_probe_capped_20260531.pt`
- checkpoint epoch: `5`
- model: `TemporalCrossAlphaModel`
- train split: 2018-01-01 to 2023-12-31
- validation split: 2024-01-01 to 2024-12-31
- temporal input: `X_seq=(60,23)`
- cross-section input: `X_dim=236`
- training sampling: `keep_ratio=0.20`, `max_train_stocks=800`
- training validation cap: `max_eval_stocks=1200`
- full validation method: temporal encoder chunked, cross-section Transformer full-stock

## Training Log

| epoch | train loss | capped val alpha |
|---:|---:|---:|
| 1 | -0.1038 | 0.0971 |
| 2 | -0.1278 | 0.0981 |
| 3 | -0.1413 | 0.0990 |
| 4 | -0.1505 | 0.0958 |
| 5 | -0.1610 | 0.0991 |

Best checkpoint was saved at epoch 5.

## Full-Stock Validation

Source: `reports\temporal_cross_alpha_probe_capped_20260531_val_full_eval.md`

| metric | value |
|---|---:|
| alpha | 0.096135 |
| h1 | 0.055115 |
| h3 | 0.079694 |
| h5 | 0.089697 |
| h7 | 0.091821 |
| topret_h5_top5 | 0.009351 |
| topret_h5_top10 | 0.011483 |
| topic_h5_top5 | -0.009183 |
| topic_h5_top10 | -0.007784 |
| topbot_h5 | 0.229237 |
| topret_h7_top5 | 0.000002 |
| topret_h7_top10 | 0.005003 |
| topic_h7_top5 | -0.009336 |
| topic_h7_top10 | -0.010189 |
| topbot_h7 | 0.232735 |

Interpretation: the model has positive full-cross-section IC and positive top-bottom spread, but top-bucket internal IC is still negative for h5/h7. This matches the earlier concern that the model can roughly separate good and bad buckets but does not rank the top bucket well.

## Daily Top Backtest

Source: `backtest_results_temporal_daily_top_20260531\temporal_daily_top_summary.csv`

Costs:

- commission: 0.01% double-sided
- stamp tax: 0.05% sell-side
- slippage/spread: 0.05% double-sided

| top_frac | annual return | sharpe | max drawdown | avg turnover | total cost |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 16.49% | 0.627 | 43.59% | 1.138 | 0.2339 |
| 0.10 | 19.42% | 0.708 | 41.96% | 0.958 | 0.1968 |

Interpretation: raw daily top selection is tradable but not strong enough. Turnover is high, drawdown is large, and the full-cost return is below the previous V9-derived best fixed/rank-exit results.

## Conclusion

This first V10 temporal probe is technically successful but not yet alpha-successful. The temporal architecture trains stably under capped sampling and full-stock evaluation is feasible, but the current checkpoint does not clearly beat V9. The next useful optimization is not more blind epochs; it is better training/evaluation alignment:

1. Use full-stock validation during training through chunked temporal encoding, or at least increase/remove validation cap.
2. Improve training sampling from random 800 stocks to top/bottom/random sampling using a cheap score or previous V9 alpha.
3. Add a turnover-aware/retention-aware trading layer only after alpha quality improves.
4. Consider sequence missingness handling with fill + `seq_mask` instead of strict 60/60 valid filtering.
