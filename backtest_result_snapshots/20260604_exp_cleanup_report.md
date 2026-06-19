# EXP Cleanup Report 2026-06-04

## Cleanup Summary

This cleanup reduced root-directory clutter and removed rebuildable temporary artifacts while preserving the current best model, current comparison reports, and all non-smoke checkpoint directories.

Actions completed:

- Deleted low-risk temporary artifacts:
  - `__pycache__` directories
  - `checkpoints_exp_smoke`
  - temporal smoke/fixed-window result directories
  - `cache/*test300*` small-sample cache files
  - daily alpha `.jsonl` files under large backtest output folders
- Freed approximately `4.39 GB`.
- Moved historical backtest result folders into `_archive_results_20260604`.
- Moved obsolete switch/trade-policy generated data and models into `_archive_models_data_20260604`.
- Deleted empty directories:
  - `experiments.log`
  - `switch_value_data_20260530`
  - `switch_value_data_20260530_comm001`

## Current Root Layout

Important directories left at root:

| directory | purpose |
|---|---|
| `backtest_result_snapshots` | high-level reports and decision records |
| `backtest_results_v9_retention_20260531` | current V9 retention strategy summaries |
| `backtest_results_temporal_retention_20260604_v10_v9warm_toploss` | latest V10 warm-start retention check |
| `backtest_results_temporal_full_eval_20260604` | latest V10 full-stock test eval |
| `checkpoints_exp_topfocus_w005_topic` | current best V9 topfocus checkpoint |
| `checkpoints_exp` | V10 and other experimental checkpoints |
| `cache` | large rebuildable feature/cache data, currently retained |
| `_archive_results_20260604` | old result directories moved out of root |
| `_archive_models_data_20260604` | old generated policy datasets/models moved out of root |

## Current Best Result

Current best model/strategy remains V9, not the latest V10.

Best unconstrained same-engine candidate:

```text
V9 topfocus checkpoint
+ raw daily alpha signal
+ retention-first daily state strategy
+ target3/hold25
+ legacy market timing
+ equal weight
+ explicit full costs
```

Reported result:

| split | annual return | Sharpe | max drawdown | avg turnover | avg holding days |
|---|---:|---:|---:|---:|---:|
| validation | 109.50% | 2.128 | 15.81% | 0.360 | 4.61 |
| test | 146.25% | 4.218 | 12.11% | 0.508 | 3.69 |

More practical lower-turnover candidate:

```text
V9 topfocus checkpoint
+ raw daily alpha signal
+ retention-first daily state strategy
+ target3/hold40
+ legacy market timing
+ equal weight
+ explicit full costs
```

Reported result:

| split | annual return | Sharpe | max drawdown | avg turnover | avg holding days |
|---|---:|---:|---:|---:|---:|
| validation | 96.68% | 2.011 | 15.29% | 0.268 | 6.26 |
| test | 142.78% | 4.223 | 11.66% | 0.392 | 4.84 |

Production-realistic constrained candidate from the current report:

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

Reported result:

| split | annual return | Sharpe | max drawdown | executed turnover | avg holding days |
|---|---:|---:|---:|---:|---:|
| validation | 43.97% | 1.246 | 16.24% | 0.192 | 9.10 |
| test | 68.22% | 2.581 | 13.29% | 0.240 | 8.10 |

## Latest V10 Status

Latest V10 checkpoint:

```text
checkpoints_exp/temporal_v10_v9_warm_adapter_toploss_n4500_seed20260601_screen80.pt
```

Full-stock test eval:

| metric | value |
|---|---:|
| alpha | 0.057086 |
| h5 | 0.054301 |
| topret_h5_top5 | 0.037880 |
| topret_h5_top10 | 0.032649 |
| topic_h5_top10 | 0.005603 |

Retention validation check:

| setup | annual return | Sharpe | max drawdown | turnover | avg holding days |
|---|---:|---:|---:|---:|---:|
| target3/hold40 | 34.38% | 1.087 | 16.72% | 0.148 | 11.78 |
| target3/hold25 | 29.92% | 0.986 | 17.26% | 0.213 | 8.07 |

Conclusion: this V10 run is usable as an experiment but is not competitive with the V9 mainline.

## Do Not Delete Without Rechecking

- `checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt`
- `backtest_result_snapshots/20260531_model_strategy_comparison_report.md`
- `backtest_results_v9_retention_20260531/*.csv`
- `checkpoints_exp/temporal_v10_v9_warm_adapter_toploss_n4500_seed20260601_screen80.pt`
- `cache/temporal_cross_section_temporal_v1_*_meta.pkl`
- source files under `core`, `data`, `run`, `backtest`

## Future Cleanup Candidates

These are retained for now because they may still be useful, but they are rebuildable or secondary:

- `cache`: about 64 GB; largest space user. Keep while V9/V10 iteration continues.
- older non-best checkpoints in `checkpoints_exp`: keep until V10 direction is settled.
- `_archive_results_20260604`: can be compressed or deleted later if reports are enough.
- `_archive_models_data_20260604`: can be deleted later if switch/trade-policy experiments are abandoned.
