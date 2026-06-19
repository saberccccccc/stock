# Forward Observation 2026-05-19 to 2026-06-16

## Data Update

Updated on 2026-06-17:

- `data/raw` stock daily data: most stocks updated to 2026-06-17.
- `data/raw` broad/industry index data: updated to 2026-06-16.
- `data/forward_raw` stock daily data: incrementally updated from 2026-06-12 to 2026-06-17.
- `data/forward_raw` broad/industry index data: updated to 2026-06-16.

Forward evaluation uses:

```text
data_dir=data/forward_raw
max_data_date=2026-06-16
signal_dates=2026-05-19 to 2026-06-16
```

The extra 2026-06-17 stock rows are not used because broad index data is only current through 2026-06-16.

## Alpha

Generated frozen V9 avgw3 forward alpha:

```text
forward_results/frozen_v9_avgw3/alpha_20260519_20260616.jsonl
```

Applied `maxret095` signal-day chase filter:

```text
forward_results/frozen_v9_avgw3/alpha_20260519_20260616_maxret095.jsonl
```

Transform summary:

```text
dates=21
codes=5285
demoted_total=2464
```

## Official Parameter Check

Official open-price share-ledger parameters:

```text
target_frac=0.006
hold_frac=0.10
max_new_names=5
rebalance_band=0.20
market_timing_mode=legacy
minADV=3M
ADV cap=5%
capital=50w/100w
```

Forward result:

| Capital | Ann | Sharpe | MDD |
|---:|---:|---:|---:|
| 50w | -61.03% | -2.997 | 11.23% |
| 100w | -52.25% | -2.017 | 11.78% |

Observation:

- The official historical baseline is not working in this short forward window.
- The result is still based on only 20 return days, so it is not enough to replace the historical baseline, but it is a serious warning.

## Market Timing Check

Same parameters, but `market_timing_mode=dynamic`:

| Capital | Ann | Sharpe | MDD |
|---:|---:|---:|---:|
| 50w | -54.45% | -3.355 | 9.10% |
| 100w | -49.74% | -2.623 | 9.39% |

Observation:

- Dynamic market timing reduces drawdown.
- It does not improve Sharpe or solve the negative-return problem.
- Current dynamic timing is a risk-control candidate, not an alpha improvement.

## Target Fraction Check

Legacy market timing, `max_new_names=5`, `rebalance_band=0.20`:

| Capital | target | hold | Ann | Sharpe | MDD | Avg Names |
|---:|---:|---:|---:|---:|---:|---:|
| 50w | 0.002 | 0.06 | -23.16% | -0.402 | 9.44% | 9.85 |
| 50w | 0.003 | 0.06 | -27.41% | -0.695 | 9.37% | 14.70 |
| 50w | 0.004 | 0.06 | -49.12% | -1.981 | 10.50% | 19.40 |
| 50w | 0.006 | 0.06 | -61.03% | -2.997 | 11.23% | 28.95 |
| 100w | 0.002 | 0.06 | -11.95% | -0.058 | 9.45% | 9.90 |
| 100w | 0.003 | 0.06 | -1.24% | 0.170 | 9.09% | 14.80 |
| 100w | 0.004 | 0.06 | -28.98% | -0.812 | 9.62% | 19.80 |
| 100w | 0.006 | 0.06 | -52.25% | -2.017 | 11.78% | 29.85 |

Observation:

- Lower `target_frac` is clearly better in this forward window.
- For 100w, `target_frac=0.003` nearly flattens the forward loss and gives a positive Sharpe.
- For 50w, `target_frac=0.002` is the least bad.
- This suggests the forward drawdown is concentrated outside the very top names; the historical `target_frac=0.006` may be too broad under the current market regime.

## Dynamic Low-Target Check

Dynamic market timing with low target:

| Capital | target | hold | Ann | Sharpe | MDD |
|---:|---:|---:|---:|---:|---:|
| 50w | 0.002 | 0.06 | -26.28% | -0.796 | 7.22% |
| 50w | 0.003 | 0.06 | -26.98% | -1.012 | 7.14% |
| 100w | 0.002 | 0.06 | -16.02% | -0.311 | 7.49% |
| 100w | 0.003 | 0.06 | -12.27% | -0.264 | 7.13% |

Observation:

- Dynamic timing lowers MDD from about 9% to about 7%.
- It sacrifices return and Sharpe versus low-target legacy.

## Current Conclusion

Do not replace the official historical baseline from this short forward window alone.

However, the next optimization direction is now clearer:

1. Keep frozen V9 avgw3 + maxret095 as the reference signal.
2. Add a forward/risk mode that narrows `target_frac` when recent executable performance is weak.
3. Validate `target_frac=0.002/0.003/0.004` on historical validation/test with the same open-price share-ledger, not only on forward.
4. Treat dynamic market timing as a drawdown-control overlay, not a return enhancer.
5. Do not tune only to this 20-return-day forward window.

Output tables:

```text
reports/forward_observation_20260617/forward_open_ledger_summary_20260519_20260616.csv
reports/forward_observation_20260617/forward_open_ledger_top_by_sharpe_20260519_20260616.csv
```
