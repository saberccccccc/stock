# Target Fraction Validation 2026-06-17

## Purpose

Forward observation from 2026-05-19 to 2026-06-16 showed that the official open-price share-ledger setting:

```text
target_frac=0.006
hold_frac=0.10
max_new_names=5
rebalance_band=0.20
market_timing_mode=legacy
maxret095
```

performed poorly in the short forward window. Lower `target_frac` looked much better forward, so this round tested whether lower target is a real historical improvement or only a recent-window effect.

## Important Validation-Cut Note

Old validation single-diagnostic output included two cross-year return days:

```text
2025-01-02
2025-01-03
```

This round uses a cleaner validation cut:

```text
max_data_date=2024-12-31
```

Therefore the same `target_frac=0.006` validation baseline is slightly different from older notes. Test results are unchanged versus the previous baseline.

## Fixed Target Results

Same signal and execution rules; only `target_frac` changes.

### Validation 2024

| Capital | Target | Ann | Sharpe | MDD | Avg Names |
|---:|---:|---:|---:|---:|---:|
| 50w | 0.002 | 89.44% | 1.824 | 20.70% | 9.25 |
| 50w | 0.003 | 97.08% | 1.957 | 21.78% | 14.24 |
| 50w | 0.004 | 81.95% | 1.795 | 19.59% | 19.23 |
| 50w | 0.006 | 84.08% | 1.834 | 21.07% | 29.13 |
| 100w | 0.002 | 90.55% | 1.829 | 20.83% | 9.25 |
| 100w | 0.003 | 98.01% | 1.951 | 21.91% | 14.24 |
| 100w | 0.004 | 84.12% | 1.809 | 19.63% | 19.23 |
| 100w | 0.006 | 89.31% | 1.879 | 20.79% | 29.23 |

Validation prefers narrower portfolios, especially `target_frac=0.003`.

### Test 2025-01-01 to 2026-05-18

| Capital | Target | Ann | Sharpe | MDD | Avg Names |
|---:|---:|---:|---:|---:|---:|
| 50w | 0.002 | 64.89% | 1.982 | 19.06% | 9.63 |
| 50w | 0.003 | 67.02% | 2.238 | 18.39% | 14.65 |
| 50w | 0.004 | 76.49% | 2.574 | 16.08% | 19.66 |
| 50w | 0.006 | 84.33% | 2.954 | 14.28% | 29.57 |
| 100w | 0.002 | 67.78% | 2.011 | 19.43% | 9.63 |
| 100w | 0.003 | 69.58% | 2.221 | 19.45% | 14.67 |
| 100w | 0.004 | 78.46% | 2.519 | 17.15% | 19.73 |
| 100w | 0.006 | 88.23% | 2.888 | 15.24% | 29.70 |

Test still strongly prefers `target_frac=0.006`.

## Mid Target Results

Mid targets were tested to find a compromise:

```text
0.0045 / 0.0050 / 0.0055 / 0.0060
```

Result:

- `0.0055` slightly improves validation versus `0.006`.
- `0.0055` still hurts test, especially 100w annualized return.
- No fixed mid target is a clean replacement.

## Risk Target Switch

Added explicit optional arguments to `run/backtest_retention_open_ledger.py`:

```text
--risk-target-frac
--risk-target-market-mult-below
```

Default behavior is unchanged. When enabled, the backtest keeps the normal `target_frac` unless the computed market multiplier is below the threshold. In that risk state, it uses the lower target.

Tested:

```text
normal target=0.006
risk target=0.003 / 0.004 / 0.005
threshold market_mult < 1.0
```

### Risk Target Summary

| Split | Capital | Risk Target | Ann | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|
| Validation | 50w | 0.003 | 87.92% | 1.871 | 19.87% |
| Validation | 50w | 0.004 | 90.82% | 1.920 | 18.85% |
| Validation | 50w | 0.005 | 82.06% | 1.819 | 20.34% |
| Validation | 100w | 0.003 | 91.19% | 1.891 | 20.08% |
| Validation | 100w | 0.004 | 95.67% | 1.957 | 18.80% |
| Validation | 100w | 0.005 | 86.06% | 1.853 | 20.33% |
| Test | 50w | 0.003 | 78.81% | 2.637 | 17.25% |
| Test | 50w | 0.004 | 81.60% | 2.770 | 15.38% |
| Test | 50w | 0.005 | 83.05% | 2.852 | 14.02% |
| Test | 100w | 0.003 | 83.51% | 2.614 | 18.32% |
| Test | 100w | 0.004 | 84.84% | 2.727 | 16.52% |
| Test | 100w | 0.005 | 84.14% | 2.747 | 15.79% |

Interpretation:

- `risk_target=0.004` is best on validation.
- `risk_target=0.005` is least damaging on test.
- All risk-target versions still underperform the official fixed `0.006` on test Sharpe and annualized return.

## Forward Check

Forward 2026-05-19 to 2026-06-16:

| Capital | Mode | Ann | Sharpe | MDD |
|---:|---|---:|---:|---:|
| 50w | official 0.006 | -61.03% | -2.997 | 11.23% |
| 50w | fixed 0.002 | -23.16% | -0.402 | 9.44% |
| 50w | risk 0.004 | -54.11% | -2.384 | 10.91% |
| 50w | risk 0.005 | -55.28% | -2.463 | 10.67% |
| 100w | official 0.006 | -52.25% | -2.017 | 11.78% |
| 100w | fixed 0.003 | -1.24% | 0.170 | 9.09% |
| 100w | risk 0.004 | -39.05% | -1.271 | 11.41% |
| 100w | risk 0.005 | -47.67% | -1.747 | 11.18% |

The simple risk-target switch helps a little forward, but not enough. The forward problem is not captured fully by the current legacy market multiplier.

## Decision

Do not replace the official fixed `target_frac=0.006` baseline.

Reason:

- validation improves with lower or risk-switched target;
- forward also prefers narrower target;
- but historical test still strongly prefers fixed `0.006`;
- risk-target switch does not solve forward enough to justify the test sacrifice.

Current status:

```text
Official baseline remains:
V9 avgw3 + maxret095 + open-price share-ledger
target_frac=0.006
hold_frac=0.10
max_new_names=5
rebalance_band=0.20
legacy
```

New candidate for observation only:

```text
risk_target_r004
normal target=0.006
risk target=0.004 when market_mult < 1.0
```

Why `risk_target_r004` remains observation-only:

- best validation balance;
- less bad forward than official;
- still weaker than official on test.

## Next Work

The useful next direction is not simply lowering target. It is improving the risk trigger.

Candidate triggers to test:

1. recent strategy drawdown trigger;
2. recent Top30 executable return trigger;
3. market breadth deterioration trigger;
4. high turnover plus weak alpha spread trigger;
5. combined trigger that lowers target only when both market and strategy state are poor.

Output files:

```text
reports/target_fraction_validation_20260617/target_fraction_all_results.csv
reports/target_fraction_validation_20260617/target_fraction_compact_results.csv
reports/target_fraction_validation_20260617/target_fraction_top_by_sharpe.csv
```
