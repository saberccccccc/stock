# V10 Temporal Long-Only Metric Experiment - 2026-05-31

## Objective

Improve long-only top-stock quality and backtest return. The checkpoint selection metric was changed away from `topbot_h5` because the strategy cannot short the bottom bucket.

New long-only checkpoint metric:

```text
0.4 * alpha
+ 0.3 * topret_h5_top5
+ 0.2 * topret_h5_top10
+ 0.1 * topic_h5_top10
```

## Code Changes

- `run/train_temporal.py`
  - Added `h3h5h7` horizon mode with weights `(0.25, 0.45, 0.30)`.
  - Added composite best metric expressions, e.g. `composite:alpha=0.4,topret_h5_top5=0.3,...`.
  - Added full validation progress logging.
  - Added optional validation date limiting for fast screening, but the final experiment below used full validation.
  - Added `torch.cuda.empty_cache()` after validation to reduce retained CUDA cache.

## Best New Checkpoint

```text
checkpoints_exp/temporal_cross_alpha_h3h5h7_longonly_h128_t64_n3000_acc4_lr1e4_5590_fullval_20260531.pt
```

Training setup:

```text
horizon_mode=h3h5h7
hidden_dim=128
temporal_dim=64
max_train_stocks=3000
batch_size=1
accum_steps=4
lr=1e-4
sampler=5% true top / 5% true bottom / 90% random
validation=full 2024 validation split, 242 dates
```

Epoch 2 was selected:

```text
alpha=0.106354
h3=0.080086
h5=0.094228
h7=0.095172
topret_h5_top5=0.024699
topret_h5_top10=0.031792
topic_h5_top5=-0.007675
topic_h5_top10=-0.009631
topret_h7_top5=0.021270
topret_h7_top10=0.028060
topbot_h5=0.261361
```

Continuation check:

The same checkpoint was resumed from epoch 2 and trained through epoch 3. Epoch 3 overfit relative to the long-only objective:

```text
epoch 2 composite=0.0553, alpha=0.1064, topret_h5_top5=0.0247, topret_h5_top10=0.0318
epoch 3 composite=0.0460, alpha=0.1019, topret_h5_top5=0.0087, topret_h5_top10=0.0188
```

Epoch 4 was stopped before validation. The saved checkpoint remains the epoch 2 best.

## Validation Metric Comparison

| checkpoint | alpha | h5 | topret_h5_top5 | topret_h5_top10 | topic_h5_top10 | topret_h7_top5 |
|---|---:|---:|---:|---:|---:|---:|
| new h3h5h7 long-only small | 0.106354 | 0.094228 | 0.024699 | 0.031792 | -0.009631 | 0.021270 |
| h5-only h192/t96 full-val | 0.095663 | 0.095538 | 0.023514 | 0.030145 | -0.011756 | n/a |
| target sampler epoch2 | 0.099618 | 0.092415 | 0.017445 | 0.023035 | -0.009372 | 0.008379 |
| random/capped V10 probe | 0.099058 | 0.093522 | 0.015028 | 0.017650 | -0.006748 | 0.002233 |

Interpretation:

- The new checkpoint is the best so far on overall `alpha`.
- It also has the best h5 and h7 top-bucket average returns.
- Top-bucket internal IC is still negative, so the model finds a useful top pool but does not rank names inside that pool well.

## Corrected Daily Long-Only Backtest

Costs:

```text
commission: 0.01% both sides
stamp tax: 0.05% sell side
slippage/spread: 0.05% both sides
```

The daily top strategy fully rebuilds the target portfolio every day.

### Narrow Top Buckets

| checkpoint | top_frac | ann | sharpe | max drawdown | avg turnover | total cost |
|---|---:|---:|---:|---:|---:|---:|
| new h3h5h7 long-only small | 3% | -23.50% | -0.318 | 45.37% | 1.337 | 0.275 |
| new h3h5h7 long-only small | 5% | -21.11% | -0.272 | 45.01% | 1.203 | 0.247 |
| new h3h5h7 long-only small | 10% | -15.98% | -0.158 | 43.16% | 0.993 | 0.204 |
| h5-only h192/t96 | 5% | -19.71% | -0.237 | 43.53% | 1.118 | 0.230 |
| target sampler | 5% | -18.12% | -0.401 | 33.97% | 0.898 | 0.185 |
| target sampler | 10% | -15.17% | -0.297 | 33.52% | 0.737 | 0.151 |

### Wider Top Buckets For New Checkpoint

| top_frac | ann | sharpe | max drawdown | avg turnover | total cost |
|---:|---:|---:|---:|---:|---:|
| 10% | -15.98% | -0.158 | 43.16% | 0.993 | 0.204 |
| 15% | -11.27% | -0.047 | 41.96% | 0.855 | 0.176 |
| 20% | -9.26% | -0.004 | 40.96% | 0.750 | 0.154 |
| 30% | -6.09% | 0.064 | 39.43% | 0.593 | 0.122 |

## Conclusion

The new long-only metric improved validation prediction quality, especially top-bucket h5/h7 returns. However, this did not translate into positive daily long-only backtest returns under full explicit costs.

The bottleneck is now clearly the trading layer:

- Daily full top5/top10 reconstruction has very high turnover.
- Wider top buckets reduce loss and turnover, which means signal quality is broad but the narrow top ranking is not precise enough.
- The model can identify a better-than-average pool, but it cannot yet support expensive daily replacement of a narrow top portfolio.

## Next Actions

1. Keep the new checkpoint as the best V10 prediction candidate so far.
2. Do not judge it only with daily full-replacement top5.
3. Test a temporal retention-first daily state engine:
   - rank every day,
   - keep existing positions when they remain in the broad high-alpha pool,
   - replace only when a candidate has enough learned expected advantage over the current holding,
   - evaluate turnover, average holding days, and full explicit costs.
4. Improve top internal ranking:
   - add a small top-pairwise loss only after the base IC stabilizes,
   - or train a separate lightweight reranker for the model-selected top 10%-20% pool.

## Retention-First Trading Layer Test

A temporal retention-first daily state engine was added:

```text
run/backtest_temporal_retention.py
```

Logic:

- rank every day using temporal alpha,
- hold a target number of names,
- keep existing holdings while they remain inside a wider hold bucket,
- fill vacancies from the current top-ranked names,
- charge the same full explicit costs.

This is a diagnostic hysteresis engine, not yet the final no-parameter trading model.

### New V10 checkpoint, retention sweep

| target_frac | hold_frac | ann | sharpe | max drawdown | avg turnover | avg holding days | total cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 10% | -14.40% | -0.092 | 44.65% | 0.780 | 2.55 | 0.160 |
| 5% | 20% | -6.52% | 0.098 | 43.94% | 0.438 | 4.53 | 0.090 |
| 5% | 30% | -5.35% | 0.125 | 43.00% | 0.293 | 6.74 | 0.060 |
| 10% | 30% | -5.29% | 0.114 | 43.06% | 0.331 | 5.98 | 0.068 |
| 15% | 50% | 0.01% | 0.225 | 40.79% | 0.179 | 10.96 | 0.037 |
| 20% | 50% | -0.21% | 0.215 | 40.34% | 0.191 | 10.28 | 0.039 |
| 30% | 50% | -0.47% | 0.201 | 39.19% | 0.224 | 8.81 | 0.046 |

### Old target-sampler checkpoint, same wide retention sweep

| target_frac | hold_frac | ann | sharpe | max drawdown | avg turnover | avg holding days | total cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30% | 50% | -2.90% | 0.094 | 32.94% | 0.155 | 12.65 | 0.032 |
| 30% | 40% | -3.47% | 0.077 | 32.71% | 0.247 | 7.99 | 0.051 |
| 20% | 40% | -5.09% | 0.027 | 33.26% | 0.185 | 10.61 | 0.038 |
| 20% | 50% | -5.24% | 0.023 | 33.47% | 0.128 | 15.24 | 0.026 |

Retention-first materially improves the new V10 checkpoint:

```text
daily top5 full replacement: -21.11% ann, turnover 1.203
target5/hold30 retention:     -5.35% ann, turnover 0.293
target15/hold50 retention:     0.01% ann, turnover 0.179
```

This confirms that turnover is a central bottleneck. The signal is not strong enough for narrow daily full replacement, but a wider persistent high-alpha pool is close to breakeven after full costs.

### Zero-cost diagnostic

The same new V10 retention setups were rerun with all explicit costs set to zero.

| target_frac | hold_frac | gross ann | sharpe | max drawdown | avg turnover |
|---:|---:|---:|---:|---:|---:|
| 15% | 40% | 2.07% | 0.271 | 41.85% | 0.251 |
| 15% | 50% | 3.88% | 0.309 | 40.61% | 0.179 |
| 20% | 40% | 3.31% | 0.295 | 40.81% | 0.277 |
| 20% | 50% | 3.92% | 0.307 | 40.14% | 0.191 |
| 30% | 40% | 5.00% | 0.327 | 39.04% | 0.349 |
| 30% | 50% | 4.37% | 0.313 | 38.94% | 0.224 |

This separates the bottlenecks:

- gross alpha is positive but weak,
- explicit trading costs are enough to erase most of it,
- drawdown remains large even with zero costs.

So the next improvement should not be a wider static hold bucket alone. It needs a learned replacement/reranking step that switches only when the expected advantage is large enough.

### Ultra-low-turnover retention sweep

The wider hold buckets were extended to check whether explicit costs were the main blocker.

New V10 checkpoint:

| target_frac | hold_frac | ann | sharpe | max drawdown | avg turnover | avg holding days | total cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | 90% | 5.46% | 0.335 | 33.60% | 0.044 | 42.62 | 0.0087 |
| 30% | 80% | 4.84% | 0.321 | 35.19% | 0.074 | 26.00 | 0.0149 |
| 20% | 80% | 4.74% | 0.321 | 35.03% | 0.072 | 26.72 | 0.0145 |
| 40% | 80% | 4.24% | 0.305 | 34.70% | 0.076 | 25.25 | 0.0154 |
| 30% | 90% | 4.13% | 0.302 | 33.44% | 0.043 | 42.88 | 0.0087 |
| 40% | 90% | 4.07% | 0.299 | 33.05% | 0.044 | 42.41 | 0.0088 |

Old target-sampler checkpoint, same extreme hold sweep:

| target_frac | hold_frac | ann | sharpe | max drawdown | avg turnover | avg holding days | total cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40% | 80% | 1.68% | 0.223 | 30.96% | 0.049 | 38.13 | 0.0099 |
| 40% | 90% | 0.71% | 0.191 | 29.78% | 0.028 | 63.47 | 0.0056 |
| 40% | 70% | 0.64% | 0.195 | 32.01% | 0.078 | 24.70 | 0.0157 |

Interpretation:

- The new V10 checkpoint clearly beats the older target-sampler checkpoint under the same low-turnover retention engine.
- The best net result so far is `target20/hold90`: `5.46%` annual, `0.335` Sharpe, `33.60%` max drawdown.
- This is positive after full explicit costs, but the alpha is spread across a very broad persistent pool. It is not yet a strong high-conviction top strategy.
- The next step should be a learned replacement/reranking layer to increase alpha concentration without returning to high turnover.

## First Temporal Switch-Value Attempt

A first temporal switch-value dataset was built:

```text
run/build_temporal_switch_dataset.py
temporal_switch_value_data_20260531/h3h5h7_longonly_target20_hold90/
```

Rows describe replacing current holding `A` with candidate `B` under the `target20/hold90` retention state.

Label:

```text
switch_edge_net_h5 = B_ret_fwd_h5 - A_ret_fwd_h5 - explicit_switch_cost
```

Dataset summary:

| split | rows | net edge mean | raw edge mean | success rate | cost mean |
|---|---:|---:|---:|---:|---:|
| train | 251,436 | 0.1974 | 0.1991 | 56.25% | 0.0017 |
| val | 41,753 | 0.1130 | 0.1147 | 54.46% | 0.0017 |

LightGBM model:

```text
temporal_switch_value_models_20260531/h3h5h7_longonly_target20_hold90_lgb_h5/switch_value_model.pkl
```

Validation diagnostics:

| metric | value |
|---|---:|
| val Spearman IC | 0.0153 |
| val R2 | -0.0261 |
| val pred-positive rate | 89.03% |
| val pred-positive true edge mean | 0.1032 |
| val overall true edge mean | 0.1130 |

Bucket diagnostics show weak and non-monotonic validation separation. The highest predicted bucket has positive edge, but the lowest bucket also has positive edge, and overall rank ordering is not reliable.

Conclusion:

- The first learned switch-value model should **not** be used as a trading gate yet.
- It overfits train and does not robustly identify better replacement pairs on validation.
- The candidate-pair construction already has positive average edge, so the harder problem is ranking/selecting the best few switches, not merely detecting positive switches.
- Next model attempt should include richer state:
  - realized volatility/liquidity/limit features,
  - market regime,
  - industry-relative rank,
  - broader multi-horizon labels,
  - and a ranking/listwise objective over candidates per day rather than independent pair regression.

## Pool Weighting Test

The retention engine was extended with pool weighting modes:

```text
run/backtest_temporal_retention.py --weight-mode equal|rank_linear|alpha_positive
```

Results on the best V10 checkpoint:

| weight mode | best setup | ann | sharpe | max drawdown | avg turnover |
|---|---|---:|---:|---:|---:|
| equal | target20/hold90 | 5.46% | 0.335 | 33.60% | 0.044 |
| rank_linear | target40/hold90 | -2.75% | 0.147 | 38.08% | 0.375 |
| alpha_positive | target40/hold90 | -9.48% | -0.001 | 40.85% | 0.693 |

Conclusion:

- Dynamic weighting destroys the low-turnover advantage because weights change every day even when the name list is mostly stable.
- Equal-weight retention is currently better.
- Future trading-layer optimization should focus on replacement decisions, not daily alpha-proportional reweighting.

## Test Split Check

The best V10 retention setups were tested on the held-out test split.

Test signal window:

```text
2025-01-02 to 2026-04-29
319 signal dates
320 active return days
```

### V10 retention, test split

| target_frac | hold_frac | ann | sharpe | max drawdown | avg turnover | avg holding days | total cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | 80% | 61.41% | 2.122 | 16.77% | 0.082 | 23.74 | 0.0220 |
| 20% | 90% | 59.60% | 2.108 | 16.80% | 0.046 | 41.49 | 0.0122 |
| 30% | 80% | 60.45% | 2.131 | 15.98% | 0.084 | 23.11 | 0.0226 |
| 30% | 90% | 58.13% | 2.104 | 16.37% | 0.046 | 41.71 | 0.0122 |
| 40% | 80% | 58.56% | 2.113 | 15.61% | 0.087 | 22.42 | 0.0233 |
| 40% | 90% | 57.15% | 2.104 | 15.90% | 0.046 | 41.15 | 0.0123 |

### V10 daily-top baseline, test split

| top_frac | ann | sharpe | max drawdown | avg turnover |
|---:|---:|---:|---:|---:|
| 5% | 37.20% | 1.359 | 16.80% | 1.219 |
| 10% | 42.92% | 1.559 | 15.93% | 1.020 |
| 20% | 45.86% | 1.678 | 15.49% | 0.786 |
| 30% | 47.70% | 1.755 | 15.45% | 0.633 |

Interpretation:

- The retention strategy generalizes much better on the test split than on 2024 validation.
- Daily top is also strong on test, but retention produces higher return with far lower turnover.
- The result appears time-window consistent: active returns are cropped to the test signal window and do not continue to the historical price tail.
- This is promising but should not be treated as final proof. The strong difference between 2024 validation and 2025+ test suggests market-regime sensitivity.

### Test split stage breakdown

Stage breakdown files:

```text
backtest_result_snapshots/20260531_temporal_test_stage_breakdown/yearly_summary.csv
backtest_result_snapshots/20260531_temporal_test_stage_breakdown/monthly_summary.csv
```

For `target20/hold80`:

| period | days | ann | sharpe | max drawdown | sum return |
|---|---:|---:|---:|---:|---:|
| 2025 | 242 | 65.28% | 2.260 | 16.77% | 0.5103 |
| 2026 | 78 | 49.97% | 1.729 | 14.23% | 0.1355 |
| all | 320 | 61.41% | 2.122 | 16.77% | 0.6458 |

Best months:

| month | sum return |
|---|---:|
| 2025-02 | 0.0980 |
| 2025-08 | 0.0911 |
| 2026-01 | 0.0883 |
| 2026-04 | 0.0881 |
| 2025-06 | 0.0677 |

Weak months:

| month | sum return |
|---|---:|
| 2026-03 | -0.0985 |
| 2025-11 | -0.0025 |
| 2025-04 | 0.0158 |

Interpretation:

- Test performance is not only one calendar year: both 2025 and 2026 are positive.
- The strategy still has regime-specific stress, especially March 2026.
- Maximum drawdown date for the main setup is around `2025-04-07`.
- A market/regime defense layer is likely more useful than further daily reweighting.

## Market Timing / Regime Defense Test

The retention engine was extended with a market exposure multiplier:

```text
run/backtest_temporal_retention.py --market-timing-mode none|legacy|dynamic
```

Cost assumptions are unchanged:

```text
commission: 0.01% per side
stamp tax: 0.05% sell side
slippage/spread: 0.05% per side
```

Modes:

- `none`: full exposure, no market defense.
- `legacy`: simple historical market filter based on index trend/momentum. Average exposure was about `0.81` in validation and `0.90` in test.
- `dynamic`: more continuous defensive exposure using trend/momentum/breadth/volatility. Average exposure was about `0.64` in validation and `0.71` in test.

### Validation split

| setup | market mode | ann | sharpe | max drawdown | avg turnover | avg holding days | avg market mult |
|---|---|---:|---:|---:|---:|---:|---:|
| target20/hold80 | none | 4.74% | 0.321 | 35.03% | 0.072 | 26.72 | 1.00 |
| target20/hold80 | legacy | 32.72% | 0.973 | 17.39% | 0.070 | 26.72 | 0.81 |
| target20/hold80 | dynamic | 26.85% | 0.942 | 13.99% | 0.068 | 26.72 | 0.64 |
| target20/hold90 | none | 5.46% | 0.335 | 33.60% | 0.044 | 42.62 | 1.00 |
| target20/hold90 | legacy | 31.30% | 0.948 | 17.59% | 0.046 | 42.62 | 0.81 |
| target20/hold90 | dynamic | 25.85% | 0.922 | 14.17% | 0.049 | 42.62 | 0.64 |
| target30/hold80 | none | 4.84% | 0.321 | 35.19% | 0.074 | 26.00 | 1.00 |
| target30/hold80 | legacy | 32.65% | 0.991 | 16.73% | 0.072 | 26.00 | 0.81 |
| target30/hold80 | dynamic | 26.57% | 0.955 | 13.78% | 0.069 | 26.00 | 0.64 |

### Test split

| setup | market mode | ann | sharpe | max drawdown | avg turnover | avg holding days | avg market mult |
|---|---|---:|---:|---:|---:|---:|---:|
| target20/hold80 | none | 61.41% | 2.122 | 16.77% | 0.082 | 23.74 | 1.00 |
| target20/hold80 | legacy | 49.59% | 2.154 | 13.30% | 0.094 | 23.74 | 0.90 |
| target20/hold80 | dynamic | 32.81% | 2.039 | 9.23% | 0.094 | 23.74 | 0.71 |
| target20/hold90 | none | 59.60% | 2.108 | 16.80% | 0.046 | 41.49 | 1.00 |
| target20/hold90 | legacy | 48.53% | 2.145 | 13.44% | 0.061 | 41.49 | 0.90 |
| target20/hold90 | dynamic | 32.29% | 2.030 | 9.38% | 0.068 | 41.49 | 0.71 |
| target30/hold80 | none | 60.45% | 2.131 | 15.98% | 0.084 | 23.11 | 1.00 |
| target30/hold80 | legacy | 48.73% | 2.161 | 12.58% | 0.095 | 23.11 | 0.90 |
| target30/hold80 | dynamic | 32.19% | 2.042 | 8.67% | 0.096 | 23.11 | 0.71 |

Interpretation:

- The weak 2024 validation period is mostly a market-regime problem, not only a stock-selection problem. Market defense turns validation from roughly `5%` annualized with `33%-35%` drawdown into `25%-33%` annualized with `14%-18%` drawdown.
- `legacy` is currently the better practical trade-off: it preserves more return than `dynamic`, improves validation substantially, and also lowers test drawdown.
- `dynamic` is too defensive for the current objective. It is useful if the priority is drawdown control, but it sacrifices too much upside.
- Among tested variants, `target30/hold80 + legacy` is the best balanced candidate so far: validation `32.65%` annualized / `16.73%` max drawdown, test `48.73%` annualized / `12.58%` max drawdown, with explicit costs included.
- This layer is still a hand-designed market filter. Before treating it as final, the next step should be either:
  - freeze `legacy` as a conservative baseline and test it on more checkpoints/seeds, or
  - learn the exposure multiplier from market-state features with walk-forward validation.

## Seed 5591 Replication Test

The training script was updated with an explicit `--seed` argument so V10 experiments can be reproduced and compared without overwriting checkpoints.

New checkpoint:

```text
checkpoints_exp/temporal_cross_alpha_h3h5h7_longonly_h128_t64_n3000_acc4_lr1e4_seed5591_fullval_20260531.pt
```

Training config:

```text
h3h5h7
hidden_dim=128
temporal_dim=64
max_train_stocks=3000
batch_size=1
accum_steps=4
lr=1e-4
temporal adapters on
5/5/90 target-aware sampler
full validation
best metric = 0.4*alpha + 0.3*topret_h5_top5 + 0.2*topret_h5_top10 + 0.1*topic_h5_top10
seed=5591
```

Full validation metrics:

| checkpoint | epoch | composite | alpha | topret_h5_top5 | topret_h5_top10 | topic_h5_top10 |
|---|---:|---:|---:|---:|---:|---:|
| original best | 2 | 0.0553 | 0.1064 | 0.0247 | 0.0318 | -0.0096 |
| seed5591 | 2 | 0.0566 | 0.1057 | 0.0284 | 0.0320 | -0.0057 |

Interpretation:

- Seed 5591 slightly improves the validation composite and top5 validation return.
- This confirms the current V10 setup is not a one-off failure, but it also shows validation top metrics alone are not enough to choose the final trading model.

### Seed 5591 retention backtest

No market timing:

| split | setup | ann | sharpe | max drawdown | avg turnover | avg holding days |
|---|---|---:|---:|---:|---:|---:|
| val | target20/hold80 | 6.10% | 0.350 | 34.37% | 0.070 | 27.38 |
| val | target20/hold90 | 7.09% | 0.371 | 31.92% | 0.041 | 45.58 |
| val | target30/hold80 | 5.95% | 0.345 | 34.56% | 0.072 | 26.70 |
| val | target30/hold90 | 5.70% | 0.338 | 32.42% | 0.040 | 45.75 |
| test | target20/hold80 | 54.30% | 2.002 | 15.80% | 0.066 | 29.36 |
| test | target20/hold90 | 51.60% | 1.956 | 15.74% | 0.036 | 51.91 |
| test | target30/hold80 | 52.69% | 1.985 | 15.50% | 0.068 | 28.52 |
| test | target30/hold90 | 51.12% | 1.967 | 15.58% | 0.036 | 51.88 |

Legacy market timing:

| split | setup | ann | sharpe | max drawdown | avg turnover | avg holding days |
|---|---|---:|---:|---:|---:|---:|
| val | target20/hold80 | 32.61% | 0.996 | 17.23% | 0.067 | 27.38 |
| val | target20/hold90 | 31.23% | 0.981 | 16.33% | 0.044 | 45.58 |
| val | target30/hold80 | 32.38% | 1.004 | 16.32% | 0.069 | 26.70 |
| val | target30/hold90 | 29.99% | 0.960 | 15.99% | 0.043 | 45.75 |
| test | target20/hold80 | 43.99% | 2.040 | 12.51% | 0.078 | 29.36 |
| test | target20/hold90 | 41.69% | 1.982 | 12.53% | 0.052 | 51.91 |
| test | target30/hold80 | 42.67% | 2.020 | 12.27% | 0.080 | 28.52 |
| test | target30/hold90 | 41.57% | 2.002 | 12.43% | 0.052 | 51.88 |

Conclusion:

- Seed 5591 has slightly better validation top metrics and slightly better no-market validation return than the original seed.
- It does **not** beat the original seed on test return. Original `target30/hold80 + legacy` is still stronger on test: `48.73%` annualized vs seed5591 `42.67%`.
- Seed 5591 is useful as an ensemble candidate, but not as a single-checkpoint replacement.
- Next priority should be checkpoint/seed rank ensemble and model-selection rules based on combined validation + trading diagnostics, not simply continuing one seed for more epochs.

## Two-Seed Rank Ensemble Test

The retention backtest now supports comma-separated checkpoints and rank averaging:

```text
run/backtest_temporal_retention.py \
  --checkpoint ckpt_a.pt,ckpt_b.pt \
  --ensemble-mode rank_mean
```

Tested checkpoints:

```text
temporal_cross_alpha_h3h5h7_longonly_h128_t64_n3000_acc4_lr1e4_5590_fullval_20260531.pt
temporal_cross_alpha_h3h5h7_longonly_h128_t64_n3000_acc4_lr1e4_seed5591_fullval_20260531.pt
```

Legacy market timing results:

| split | setup | ann | sharpe | max drawdown | avg turnover | avg holding days |
|---|---|---:|---:|---:|---:|---:|
| val | target20/hold80 | 33.97% | 1.012 | 17.36% | 0.065 | 28.73 |
| val | target20/hold90 | 30.29% | 0.939 | 17.20% | 0.044 | 45.79 |
| val | target30/hold80 | 31.94% | 0.986 | 16.50% | 0.066 | 28.30 |
| val | target30/hold90 | 30.09% | 0.953 | 16.43% | 0.044 | 45.80 |
| test | target20/hold80 | 45.66% | 2.066 | 12.73% | 0.079 | 29.05 |
| test | target20/hold90 | 45.41% | 2.080 | 12.74% | 0.054 | 49.22 |
| test | target30/hold80 | 46.00% | 2.109 | 12.37% | 0.082 | 27.82 |
| test | target30/hold90 | 45.23% | 2.117 | 12.57% | 0.054 | 48.50 |

Interpretation:

- Ensemble improves validation robustness: `target20/hold80 + legacy` reaches `33.97%`, better than original seed `32.72%` and seed5591 `32.61%`.
- Ensemble does not beat the best original seed on test return. Original `target30/hold80 + legacy` remains the highest balanced test candidate at `48.73%` annualized.
- Ensemble is still attractive if the objective is smoother model risk rather than peak test return. It gives similar drawdown with less dependence on a single checkpoint.
- Current best practical candidates:
  - highest balanced single model: original seed `target30/hold80 + legacy`;
  - more robust ensemble candidate: two-seed rank ensemble `target30/hold80 + legacy`;
  - lower turnover ensemble candidate: two-seed rank ensemble `target30/hold90 + legacy`.
