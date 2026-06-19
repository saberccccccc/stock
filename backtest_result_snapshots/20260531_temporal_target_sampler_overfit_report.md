# V10 target-aware sampler overfit report

## What Was Tested

- checkpoint: `checkpoints_exp\temporal_cross_alpha_target_sampler_20260531.pt`
- training sampler: `target_top_bottom_random`
- sampler target: true future h5 label, `y_seq[:, 4]`
- sampler mix: 30% true top, 30% true bottom, 40% random
- max train stocks per date: 800
- validation cap during training: 1200
- full-stock validation: temporal encoder chunked, full cross-section Transformer

The true top/bottom buckets are used only for training sample selection. Validation and backtest do not use future returns for selection.

## Training Behavior

| epoch | train loss | capped val alpha |
|---:|---:|---:|
| 1 | -0.1427 | 0.0879 |
| 2 | -0.1726 | 0.0996 |
| 3 | -0.1942 | 0.0918 |
| 4 | -0.2108 | 0.0776 |
| 5 | -0.2283 | 0.0805 |

This is overfitting: training loss keeps improving while validation alpha peaks at epoch 2 and then deteriorates sharply. The saved checkpoint is correctly epoch 2.

Resource usage was stable:

- GPU memory during training: about 1.8-1.9GB / 6GB
- process RSS: about 5.9-6.2GB
- stderr: empty

## Full-Stock Validation

Source: `reports\temporal_cross_alpha_target_sampler_20260531_val_full_eval.md`

| metric | random sampler V10 | target-aware sampler V10 |
|---|---:|---:|
| alpha | 0.096135 | 0.097444 |
| h5 | 0.089697 | 0.089937 |
| h7 | 0.091821 | 0.094658 |
| topret_h5_top5 | 0.009351 | 0.019488 |
| topret_h7_top5 | 0.000002 | 0.014549 |
| topic_h5_top5 | -0.009183 | -0.001079 |
| topic_h7_top5 | -0.009336 | 0.000180 |
| topbot_h5 | 0.229237 | 0.244006 |
| topbot_h7 | 0.232735 | 0.247965 |

The sampler improves top-bucket validation metrics, but only at the early checkpoint. Later epochs overfit.

## Corrected Daily Top Backtest

The earlier daily top backtest had a time-window bug: after the last validation signal, the last portfolio continued to be marked to market through later price history. The script was fixed to crop returns from the first entry day to the last validation signal plus one trading day.

Corrected source directories:

- random sampler: `backtest_results_temporal_daily_top_20260531_fixedwindow`
- target-aware sampler: `backtest_results_temporal_target_sampler_20260531_fixedwindow`

| checkpoint | top_frac | annual return | sharpe | max drawdown | avg turnover | total cost |
|---|---:|---:|---:|---:|---:|---:|
| random sampler V10 | 0.05 | -23.65% | -0.414 | 43.59% | 1.138 | 0.2339 |
| random sampler V10 | 0.10 | -19.68% | -0.308 | 41.96% | 0.958 | 0.1968 |
| target-aware sampler V10 | 0.05 | -18.12% | -0.401 | 33.97% | 0.898 | 0.1846 |
| target-aware sampler V10 | 0.10 | -15.17% | -0.297 | 33.52% | 0.737 | 0.1513 |

The target-aware sampler reduces losses, turnover, and drawdown versus random sampling, but daily top is still not tradable in this corrected validation window.

## Conclusion

The current V10 target-aware sampler is a useful diagnostic but not a production improvement. It confirms that focusing the sampler on true winners/losers can improve top-bucket validation metrics, but it also makes overfitting faster. The daily top trading result remains weak after correcting the evaluation window.

Recommended next step:

1. Do not continue the current 30/30/40 target-aware sampler as-is.
2. Use a milder sampler, such as 10% true top, 10% true bottom, 80% random, or mix target-aware sampling only after a warmup epoch.
3. Add early stopping/checkpoint selection by a mixed metric: `0.5*alpha + 0.25*topbot_h5 + 0.25*topret_h5_top10`, instead of alpha alone.
4. Fix the trading layer before judging alpha by returns: daily top has too much turnover and should be replaced by retention/rank-exit or no-trade-threshold logic.
