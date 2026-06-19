# Loss Ablation Decision

Date: 2026-06-14

## Purpose

This experiment isolates the marginal contribution of each auxiliary loss.
It does not compare a newly trained model directly with the heavily tuned old
V9 strategy.

All portfolio comparisons below use epoch 6, 2024 only, the same `average_w3`
Alpha, the fixed 9.5% signal-day return filter, Top30, 10% hold threshold,
20% rebalance band, legacy market timing and the same execution constraints.

## Actual Experiment Definitions

| ID | Change from A0 |
|---|---|
| M0 | Remove separate multi-horizon IC loss; retain the weighted multi-horizon global target |
| A0 | Global IC plus multi-horizon IC baseline |
| A1 | Add industry IC, weight 0.10 |
| A2 | Add industry IC 0.10 and diversity 0.05 |
| A3 | Add diversity only, weight 0.05 |
| A4 | Add Top-focus only, weight 0.005 |
| A5 | Add spread only, weight 0.001 |

## Epoch 6 Training Metrics

| ID | Alpha IC | Raw Top30 h5 return | Raw Top30 stability |
|---|---:|---:|---:|
| A0 | 0.1474 | 1.663% | 0.2057 |
| A1 | 0.1468 | 1.629% | 0.2007 |
| A2 | 0.1452 | 1.586% | 0.1923 |
| A3 | 0.1457 | 1.647% | 0.1998 |
| A4 | 0.1467 | 1.670% | 0.2075 |
| A5 | 0.1430 | 1.690% | 0.2267 |

### Multi-horizon Isolation

M0 and A0 use the same seed, data split, six epochs and all other
hyperparameters. M0 sets `multi_loss_weight=0`, while A0 uses 0.3.

| ID | Alpha IC | h1 IC | h3 IC | h5 IC | h7 IC |
|---|---:|---:|---:|---:|---:|
| M0 | 0.1476 | 0.0121 | 0.0328 | 0.0231 | 0.0201 |
| A0 | 0.1474 | 0.1055 | 0.1258 | 0.1325 | 0.1350 |

The auxiliary loss clearly trains the individual horizon heads, but it does
not improve the weighted Alpha IC. Portfolio results are therefore required
to determine whether those heads help the trading objective.

## Executable 2024 Results

### Base Scenario After Fixed 9.5% Filter

| ID | CNY 500k annualized / Sharpe | CNY 1m annualized / Sharpe |
|---|---:|---:|
| A0 | 44.93% / 1.292 | 48.28% / 1.341 |
| A1 | 44.94% / 1.302 | 46.94% / 1.322 |
| A2 | 34.52% / 1.049 | 35.52% / 1.057 |
| A3 | 41.98% / 1.227 | 41.56% / 1.200 |
| A4 | **46.29% / 1.321** | **48.43% / 1.344** |
| A5 | 18.06% / 0.693 | 18.90% / 0.706 |

### Multi-horizon Loss: M0 versus A0

All figures use the fixed 9.5% signal-day return filter.

| Scenario | M0 CNY 500k annualized / Sharpe | A0 CNY 500k annualized / Sharpe | M0 CNY 1m annualized / Sharpe | A0 CNY 1m annualized / Sharpe |
|---|---:|---:|---:|---:|
| Base | **45.73% / 1.327** | 44.93% / 1.292 | 46.27% / 1.316 | **48.28% / 1.341** |
| Cost 2x | **40.19% / 1.207** | 39.68% / 1.179 | 40.83% / **1.203** | **40.88%** / 1.187 |
| Delay 1 day | **26.10% / 0.875** | 16.47% / 0.625 | **25.56% / 0.850** | 17.23% / 0.638 |

### One-Trading-Day Delay

| ID | CNY 500k annualized / Sharpe | CNY 1m annualized / Sharpe |
|---|---:|---:|
| A0 | 16.47% / 0.625 | 17.23% / 0.638 |
| A1 | **21.99% / 0.769** | **23.16% / 0.787** |
| A2 | 20.13% / 0.711 | 20.94% / 0.722 |
| A3 | 18.12% / 0.662 | 16.36% / 0.615 |
| A4 | 19.97% / 0.720 | 19.69% / 0.703 |
| A5 | 8.68% / 0.426 | 8.46% / 0.417 |

## Decisions

### Retain for the next baseline: Top-focus

A4 is the only auxiliary loss that improves base annualized return and Sharpe
for both account sizes versus A0. The improvement is modest, but it also
improves the delayed result. Retain weight 0.005 as the leading setting, then
test nearby weights without combining other auxiliary losses.

### Retest separately: Industry IC

A1 improves raw base performance and delayed execution, but after the 9.5%
filter its CNY 1m base result is slightly worse than A0. This is mixed rather
than a clear pass. Do not add weight 0.10 to the default model yet. A lower
weight such as 0.03 or 0.05 can be tested against A0 and A4.

### Remove: Diversity

A3 is the clean diversity-only comparison and is worse than A0 for both
account sizes after execution filtering. A2, which combines industry IC and
diversity, is worse again. Diversity weight 0.05 should be removed.

### Remove: Spread

A5 produces the highest raw Top30 stability metric, but its executable return
collapses. This loss is optimizing a proxy that does not align with the
tradable portfolio. Spread weight 0.001 should be removed and
`rawtopstable_h5_top0p6` must not be used alone for checkpoint selection.

### Stop extending epochs blindly

A0 epoch 9 falls from 44.93% / 48.28% annualized at epoch 6 to
10.51% / 12.85%, despite apparently respectable training metrics. More epochs
caused severe portfolio overfitting. Checkpoint selection must include the
fixed 2024 execution backtest.

### Remove the separate multi-horizon auxiliary loss

The separate multi-horizon loss improves the IC values of the individual
horizon heads, but does not improve weighted Alpha IC and substantially hurts
the one-day-delay portfolio. Base results are mixed: A0 is better for CNY 1m,
while M0 is better for CNY 500k. Under doubled costs they are effectively tied.

Use M0 as the next baseline because its execution-delay robustness is much
stronger. This does not remove the h1/h3/h5/h7 labels: the global Alpha target
still combines them with weights 0.15/0.25/0.35/0.25. It only removes the
extra loss that forces each horizon head to optimize IC independently.

### M0 plus Top-focus interaction test

M1 combines M0 with Top-focus weight 0.005. All other settings match M0,
including seed 42, six epochs and the same purged train/validation split.

| Scenario | M1 CNY 500k annualized / Sharpe | M0 CNY 500k annualized / Sharpe | M1 CNY 1m annualized / Sharpe | M0 CNY 1m annualized / Sharpe |
|---|---:|---:|---:|---:|
| Base | 42.17% / 1.252 | **45.73% / 1.327** | 41.13% / 1.212 | **46.27% / 1.316** |
| Cost 2x | 35.54% / 1.104 | **40.19% / 1.207** | 34.24% / 1.060 | **40.83% / 1.203** |
| Delay 1 day | 23.71% / 0.820 | **26.10% / 0.875** | 18.87% / 0.691 | **25.56% / 0.850** |

M1 Alpha IC is 0.14746 versus 0.14755 for M0, so aggregate IC does not
explain the portfolio deterioration. Top-focus helped A0 when the separate
multi-horizon loss was present, but does not combine cleanly with M0.
Reject weight 0.005 for the M0 baseline.

## Current Recommended Training Baseline

Use M0: the weighted multi-horizon global IC target with the separate
`multi_loss_weight` set to zero. Keep Top-focus, industry IC, diversity and
spread disabled. M0 plus Top-focus weight 0.005 has been tested and rejected.
Do not assume auxiliary-loss improvements are additive.

The 9.5% signal-day return rule remains useful as an execution filter, but it
is not a training loss and should remain outside the model objective.
