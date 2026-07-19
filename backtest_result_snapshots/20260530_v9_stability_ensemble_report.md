# 2026-05-30 V9 stability and ensemble test report

## Baseline to beat

Current best checkpoint remains:

`checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt`

Best production-style long-only backtest so far:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| single V9, simple_long, top_frac=0.045 | 34.08% | 1.395 | 18.63% | 24.72% | 1.428 | 14.96% |

Validation metrics for this checkpoint:

| metric | value |
|---|---:|
| alpha_IC | 0.1124 |
| topic_h5_top10 | 0.0231 |
| topret_h5_top5 | 0.1247 |
| topbot_h5 | 0.3349 |

## Layer diagnostics

Command output:

`backtest_results_exp_v9_layer_diag_topfocus_w005_topic_20260530/`

The current best model has healthy monotonic layering on h5 labels:

| bucket | mean h5 label |
|---|---:|
| top01 | 0.1623 |
| top02 | 0.1436 |
| top05 | 0.1248 |
| top10 | 0.1087 |
| top20 | 0.0918 |
| mid40_60 | 0.0155 |
| bot20 | -0.1395 |
| bot10 | -0.2259 |
| bot05 | -0.3403 |

Daily IC summary:

| horizon | IC | rank IC |
|---|---:|---:|
| h1 | 0.0617 | 0.0923 |
| h3 | 0.0768 | 0.1017 |
| h5 | 0.0831 | 0.1075 |
| h7 | 0.0862 | 0.1101 |

Interpretation: the model is not only hitting one noisy top5 point. The alpha has a broad monotonic cross-sectional structure, so the main problem is not "no signal"; it is how to convert the signal into a better tradable portfolio without overfitting top loss.

## Checkpoint rank ensemble

Command output:

`backtest_results_exp_v9_checkpoint_ensemble_20260530/`

Four-checkpoint equal rank ensemble:

| top_frac | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---:|---:|---:|---:|---:|---:|---:|
| 0.045 | 32.21% | 1.320 | 19.30% | 24.38% | 1.419 | 14.10% |
| 0.050 | 33.10% | 1.351 | 19.21% | 25.23% | 1.462 | 14.16% |
| 0.055 | 33.28% | 1.361 | 18.85% | 25.38% | 1.475 | 14.01% |

Two-checkpoint ensembles did not improve raw return:

| ensemble | best raw ann | best raw sharpe | best neutral sharpe |
|---|---:|---:|---:|
| topic + original, equal | 32.30% | 1.353 | 1.436 |
| topic + original, 0.7/0.3 | 32.31% | 1.357 | 1.415 |

Interpretation: checkpoint ensemble improves neutral Sharpe and neutral drawdown modestly, but still trails the single best checkpoint on raw annualized return.

## V9 + GAT long-only test

Command output:

`backtest_results_exp_long_only/v9_gat_long_only_modes_summary.csv`

Used:

`--v9-checkpoint checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt`

`--gat-checkpoint checkpoints_exp/ultimate_v7_gat_best.pt`

Best rows from the run:

| setup | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|
| avg_score, optimizer_projected_long, top5% | 30.99% | 1.348 | 20.02% | 24.81% | 1.514 | 13.15% |
| avg_score, simple_long, top10% | 30.45% | 1.305 | 18.62% | 23.75% | 1.461 | 13.49% |
| avg_score, simple_long, top5% | 30.43% | 1.290 | 20.21% | 23.93% | 1.436 | 13.54% |

Interpretation: V9+GAT does not beat the current best single V9 raw result. It does improve neutral Sharpe/drawdown in the projected-long version, but the raw return penalty is too large for it to be the primary candidate.

## Pairwise loss result

Pairwise top ranking was implemented and tested, but validation and backtest were weaker:

| run | validation / backtest result |
|---|---|
| pairwise-only, weight 0.003 | val topic_h5_top10 0.0178; best raw ann about 26.27% |
| top_focus 0.005 + pairwise 0.001 | val topic_h5_top10 0.0202; alpha_IC 0.1023; weaker than current best |

Interpretation: pairwise is not the next priority. It appears to over-constrain a noisy head region and weakens the broader IC structure.

## Current conclusion

For raw return, keep the current best single V9 checkpoint:

`checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt`

Use:

`simple_long top_frac=0.045`

For lower beta/market-adjusted stability, keep the four-checkpoint rank ensemble as a secondary candidate:

`v9_rank_ens_4ckpt top_frac=0.055`

It has lower raw return than the single checkpoint, but the best neutral Sharpe observed in this batch.

## Recommended next tests

1. Stop adding top-focused losses for now. The layer diagnostics show the signal structure is already good.
2. Use `topic_h5_top10` or a blended validation metric for checkpoint selection, but compare final candidates by layered monotonicity plus long-only backtest.
3. Train or reuse 3-5 seeds around the same stable configuration: IC/multi-horizon main loss + small top_focus=0.005. Save each to a unique directory.
4. Test only strong checkpoints in rank ensemble. Weak checkpoints dilute raw return.
5. Shift effort to portfolio controls: volatility/beta filter, execution/ADV caps, and top_frac stability around 4.5%-5.5%.

## Follow-up: top pool quality concern

The top pool is relatively good, but not absolutely excellent. For h5 top5 daily bucket quality:

| metric | value |
|---|---:|
| mean | 0.1248 |
| median | 0.1078 |
| positive-rate | 68.98% |
| 25% quantile | -0.0382 |
| 5% quantile | -0.3476 |

Key spreads:

| spread | mean | positive-rate |
|---|---:|---:|
| top05 - mid40_60 | 0.1092 | 63.96% |
| top05 - top20 | 0.0330 | 60.88% |
| top01 - top05 | 0.0376 | 57.01% |
| top05 - bot05 | 0.4650 | 83.27% |

Interpretation: the model is much better at avoiding bad stocks than at reliably selecting truly outstanding stocks inside the head bucket. This matches the concern that "top stocks are not excellent enough."

## Follow-up: V9-only persistence sweep

Script:

`run/v9_persistent_sweep.py`

Partial output:

`backtest_results_exp_v9_persistent_sweep_topfocus_w005_topic_20260530/v9_persistent_partial_summary.csv`

| variant | top_frac | raw ann | raw sharpe | raw mdd | neutral ann | neutral sharpe | neutral mdd |
|---|---:|---:|---:|---:|---:|---:|---:|
| average_w3 | 0.045 | 34.59% | 1.426 | 18.63% | 26.27% | 1.442 | 16.11% |
| average_w3 | 0.050 | 34.52% | 1.426 | 18.97% | 26.29% | 1.448 | 16.01% |
| raw | 0.045 | 34.08% | 1.395 | 18.63% | 24.72% | 1.428 | 14.96% |
| average_w5 | 0.045 | 31.70% | 1.308 | 18.75% | 23.18% | 1.260 | 18.13% |
| average_w10 | 0.045 | 27.41% | 1.154 | 21.35% | 18.81% | 1.032 | 20.30% |

Interpretation: very short persistence helps. Longer smoothing destroys timeliness. The best next production candidate is now:

`V9 topfocus_w005_topic + 3-day average alpha + simple_long top_frac=0.045/0.050`
