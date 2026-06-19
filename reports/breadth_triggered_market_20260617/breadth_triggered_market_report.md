# Breadth-Triggered Market Multiplier Test 2026-06-17

## Purpose

Prior breadth-triggered target shrink did not work. Weak breadth is still useful as a diagnostic risk signal, but reducing the number of names increased concentration and turnover.

This round tested a different action:

```text
when breadth is weak, reduce gross exposure / market multiplier
keep the stock list and ranking structure
```

## Implementation

Added:

```text
run/make_breadth_triggered_market_alpha.py
```

Added optional backtest support:

```text
run/backtest_retention_open_ledger.py --use-row-market-mult
```

Default behavior is unchanged unless `--use-row-market-mult` is passed.

## Tested Candidates

Candidate A:

```text
ma3_040_m070
up_ma3 <= 0.40 -> market_mult cap 0.70
```

Candidate B:

```text
ma3_040_m085
up_ma3 <= 0.40 -> market_mult cap 0.85
```

Candidate C:

```text
ma3_035_m085
up_ma3 <= 0.35 -> market_mult cap 0.85
```

## Validation

| Mode | Capital | Ann | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Official | 50w | 84.08% | 1.834 | 21.07% |
| ma3_040_m070 | 50w | 77.20% | 1.765 | 19.11% |
| ma3_040_m085 | 50w | 81.61% | 1.823 | 19.67% |
| ma3_035_m085 | 50w | 83.33% | 1.837 | 20.39% |
| Official | 100w | 89.31% | 1.879 | 20.79% |
| ma3_040_m070 | 100w | 83.57% | 1.834 | 18.66% |
| ma3_040_m085 | 100w | 86.65% | 1.864 | 19.87% |
| ma3_035_m085 | 100w | 88.24% | 1.878 | 20.43% |

Validation interpretation:

- aggressive exposure cut lowers MDD but sacrifices return;
- strict trigger `ma3_035_m085` is closest to official and slightly improves 50w Sharpe, but does not materially improve 100w.

## Test

Only the best validation balance, `ma3_035_m085`, was tested on historical test.

| Mode | Capital | Ann | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Official | 50w | 84.33% | 2.954 | 14.28% |
| ma3_035_m085 | 50w | 84.63% | 2.991 | 14.11% |
| Official | 100w | 88.23% | 2.888 | 15.24% |
| ma3_035_m085 | 100w | 87.07% | 2.890 | 15.21% |

Test interpretation:

- 50w improves slightly on annualized return, Sharpe, and MDD.
- 100w Sharpe is basically flat, MDD slightly better, annualized return lower.

## Forward

Forward 2026-05-19 to 2026-06-16:

| Mode | Capital | Ann | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Official | 50w | -61.03% | -2.997 | 11.23% |
| ma3_035_m085 | 50w | -55.25% | -2.711 | 10.31% |
| Official | 100w | -52.25% | -2.017 | 11.78% |
| ma3_035_m085 | 100w | -45.03% | -1.675 | 10.71% |

Forward interpretation:

- strict breadth market cap reduces recent loss and drawdown;
- it does not solve the forward problem, but it moves in the right risk-control direction.

## Decision

Do not replace the official baseline yet.

Add as observation candidate:

```text
breadth_market_ma3_035_m085
trigger: up_ma3 <= 0.35
action: cap market_mult at 0.85
```

Reason:

- improves 50w test slightly;
- keeps 100w test Sharpe essentially flat, with slightly lower MDD;
- reduces forward losses;
- validation is close to official, not clearly better.

This is the first risk-control overlay in this round that improves forward without clearly damaging 50w historical test. It still needs more stress checks before promotion:

1. cost2x;
2. lag1;
3. monthly/state stability;
4. compare with `risk_target_r004` and `negfilter_r030_100_drop3`.

## Stress Checks

Stress checks were added after the initial report. All checks use the same official open-price share-ledger settings:

```text
target_frac=0.006
hold_frac=0.10
max_new_names=5
rebalance_band=0.20
market_timing_mode=legacy
minADV=3M
ADV cap=5%
maxret095 signal filter
```

### Lag1 Execution

| Split | Mode | Capital | Ann | Sharpe | MDD |
|---|---|---:|---:|---:|---:|
| Validation | Official | 50w | 46.39% | 1.215 | 22.42% |
| Validation | ma3_035_m085 | 50w | 54.86% | 1.375 | 22.45% |
| Validation | Official | 100w | 47.36% | 1.215 | 22.89% |
| Validation | ma3_035_m085 | 100w | 57.61% | 1.401 | 22.66% |
| Test | Official | 50w | 78.59% | 2.818 | 13.81% |
| Test | ma3_035_m085 | 50w | 76.62% | 2.801 | 13.70% |
| Test | Official | 100w | 85.87% | 2.859 | 14.49% |
| Test | ma3_035_m085 | 100w | 83.42% | 2.823 | 14.56% |
| Forward | Official | 50w | -70.43% | -3.819 | 11.88% |
| Forward | ma3_035_m085 | 50w | -64.33% | -3.388 | 10.69% |
| Forward | Official | 100w | -62.63% | -2.700 | 11.28% |
| Forward | ma3_035_m085 | 100w | -55.20% | -2.259 | 10.22% |

Lag1 interpretation:

- validation improves materially;
- test is slightly worse on annualized return and Sharpe, but drawdown is similar or slightly better for 50w;
- forward remains negative, but loss and drawdown are lower than official.

### Double Cost

| Split | Mode | Capital | Ann | Sharpe | MDD |
|---|---|---:|---:|---:|---:|
| Validation | Official | 50w | 70.40% | 1.631 | 22.37% |
| Validation | ma3_035_m085 | 50w | 73.16% | 1.688 | 21.68% |
| Validation | Official | 100w | 74.55% | 1.668 | 22.22% |
| Validation | ma3_035_m085 | 100w | 77.22% | 1.719 | 21.83% |
| Test | Official | 50w | 72.91% | 2.655 | 14.61% |
| Test | ma3_035_m085 | 50w | 72.74% | 2.675 | 14.42% |
| Test | Official | 100w | 75.81% | 2.586 | 15.50% |
| Test | ma3_035_m085 | 100w | 75.26% | 2.598 | 15.45% |
| Forward | Official | 50w | -63.13% | -3.182 | 11.53% |
| Forward | ma3_035_m085 | 50w | -57.56% | -2.899 | 10.62% |
| Forward | Official | 100w | -55.82% | -2.249 | 12.15% |
| Forward | ma3_035_m085 | 100w | -47.69% | -1.829 | 10.94% |

Double-cost interpretation:

- validation improves on return, Sharpe, and drawdown;
- test Sharpe and drawdown are slightly better, while annualized return is nearly flat to slightly lower;
- forward loss and drawdown are lower than official.

## Updated Decision After Stress Checks

`breadth_market_ma3_035_m085` remains an observation candidate, not the official replacement yet.

The overlay has a real risk-control signal:

- it improves validation stress tests;
- it reduces forward losses under normal, lag1, and double-cost settings;
- it does not meaningfully damage test Sharpe or drawdown.

The reason it is not promoted immediately:

- normal test 100w annualized return is lower;
- lag1 test annualized return is lower for both 50w and 100w;
- the forward window is still short and cannot prove robustness by itself.

## Monthly And State Stability

Monthly stability was checked against clean validation, historical test, and the short forward window.

| Split | Mode | Capital | Months | Positive Months | Worst Month | Avg Month | Sum |
|---|---|---:|---:|---:|---:|---:|---:|
| Validation | Official | 50w | 12 | 7 | -9.19% | 5.41% | 64.88% |
| Validation | ma3_035_m085 | 50w | 12 | 7 | -8.50% | 5.36% | 64.37% |
| Validation | risk_target_r004 | 50w | 12 | 8 | -6.05% | 5.70% | 68.43% |
| Validation | Official | 100w | 12 | 7 | -9.20% | 5.65% | 67.84% |
| Validation | ma3_035_m085 | 100w | 12 | 7 | -8.82% | 5.60% | 67.17% |
| Validation | risk_target_r004 | 100w | 12 | 8 | -6.38% | 5.92% | 71.10% |
| Test | Official | 50w | 17 | 13 | -2.10% | 4.86% | 82.69% |
| Test | ma3_035_m085 | 50w | 17 | 14 | -1.81% | 4.87% | 82.85% |
| Test | risk_target_r004 | 50w | 17 | 13 | -1.98% | 4.77% | 81.02% |
| Test | Official | 100w | 17 | 13 | -2.07% | 5.05% | 85.81% |
| Test | ma3_035_m085 | 100w | 17 | 14 | -1.80% | 5.00% | 84.93% |
| Test | risk_target_r004 | 100w | 17 | 12 | -1.88% | 4.92% | 83.66% |
| Forward | Official | 50w | 2 | 0 | -6.29% | -3.55% | -7.11% |
| Forward | ma3_035_m085 | 50w | 2 | 0 | -4.87% | -3.03% | -6.05% |
| Forward | Official | 100w | 2 | 0 | -5.24% | -2.70% | -5.40% |
| Forward | ma3_035_m085 | 100w | 2 | 0 | -4.03% | -2.16% | -4.32% |

Monthly interpretation:

- `ma3_035_m085` does not materially improve clean validation monthly total return;
- it slightly improves test 50w monthly total and positive-month count;
- it slightly lowers test 100w monthly total, while improving positive-month count and worst month;
- `risk_target_r004` looks strongest on validation monthly stability, but its test and forward totals are worse than official, so it remains overfit/observation-only.

State-trigger interpretation:

| Split | Capital | Bucket | Days | Candidate - Official | Official Avg Ret | Candidate Avg Ret | Official Mult | Candidate Mult |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Validation | 50w | Triggered | 21 | +1.18% | -1.581% | -1.525% | 1.000 | 0.850 |
| Validation | 100w | Triggered | 21 | +1.03% | -1.610% | -1.561% | 1.000 | 0.850 |
| Test | 50w | Triggered | 20 | +0.64% | -0.854% | -0.822% | 1.000 | 0.850 |
| Test | 100w | Triggered | 20 | +0.49% | -0.881% | -0.856% | 1.000 | 0.850 |
| Forward | 50w | Triggered | 5 | +0.28% | -0.975% | -0.918% | 1.000 | 0.850 |
| Forward | 100w | Triggered | 5 | +0.44% | -1.080% | -0.992% | 1.000 | 0.850 |

This confirms the overlay behaves like risk control, not alpha enhancement. It helps mainly on weak-breadth days where the official strategy was already losing. On normal days it is roughly neutral to slightly negative.

Updated next check:

```text
compare full candidate set in one leaderboard:
official, breadth_market_ma3_035_m085, risk_target_r004, negfilter_r030_100_drop3, edge_r030_100

promotion rule:
do not promote unless validation/test/forward stress and monthly stability all remain acceptable
```

Output:

```text
reports/breadth_triggered_market_20260617/breadth_triggered_market_results.csv
```
