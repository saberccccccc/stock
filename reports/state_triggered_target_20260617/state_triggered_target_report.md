# State-Triggered Target Shrink Test 2026-06-17

## Correction

Later testing added `--use-row-target-frac` to `run/backtest_retention_open_ledger.py`.

The first state-triggered runs in this report should be interpreted as **rank-order demotion only**, not true target shrink, because the backtester did not yet read each row's `effective_target_frac`.

After enabling true row-level target shrink, the representative `state_w5_cumret_m3` result was:

| Split | Capital | Ann | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Validation | 50w | 86.64% | 1.879 | 20.39% |
| Validation | 100w | 89.34% | 1.894 | 20.99% |
| Test | 50w | 75.02% | 2.636 | 15.48% |
| Test | 100w | 77.38% | 2.556 | 16.96% |
| Forward | 50w | -58.70% | -2.561 | 11.99% |
| Forward | 100w | -40.64% | -1.249 | 11.30% |

The decision remains unchanged: do not promote state-triggered target shrink. True shrink improves forward 100w but materially hurts historical test.

## Purpose

The previous target-fraction validation showed:

- lower target helps validation and recent forward;
- lower target hurts historical test;
- market-mult based risk target helps validation but still hurts test.

This round tested a different risk trigger: shrink the alpha top list only after the strategy itself has recently lost money.

## Implementation

Added low-memory alpha transform:

```text
run/make_state_triggered_target_alpha.py
```

The script reads:

- saved alpha JSONL;
- prior official open-ledger returns CSV.

For each signal date, it computes prior-window executed strategy performance. If triggered, it keeps only the smaller top bucket and moves the rest of the original `base_target` bucket to the tail.

No model loading or feature cache loading is required.

## Tested Triggers

Candidate A:

```text
state_w5_cumret_m3
window=5 return days
trigger if prior 5-day cumulative return <= -3%
risk_target_frac=0.004
```

Candidate B:

```text
state_w3_cumret_m2
window=3 return days
trigger if prior 3-day cumulative return <= -2%
risk_target_frac=0.004
```

## Results

### Validation 2024

| Mode | Capital | Ann | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Official | 50w | 84.08% | 1.834 | 21.07% |
| state_w5_cumret_m3 | 50w | 86.42% | 1.872 | 21.43% |
| state_w3_cumret_m2 | 50w | 83.54% | 1.829 | 21.89% |
| Official | 100w | 89.31% | 1.879 | 20.79% |
| state_w5_cumret_m3 | 100w | 90.77% | 1.905 | 21.34% |
| state_w3_cumret_m2 | 100w | 89.17% | 1.878 | 21.55% |

Validation has only a small improvement for `state_w5_cumret_m3`, with slightly worse drawdown.

### Test 2025-01-01 to 2026-05-18

| Mode | Capital | Ann | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Official | 50w | 84.33% | 2.954 | 14.28% |
| state_w5_cumret_m3 | 50w | 81.60% | 2.889 | 14.45% |
| state_w3_cumret_m2 | 50w | 78.00% | 2.799 | 14.47% |
| Official | 100w | 88.23% | 2.888 | 15.24% |
| state_w5_cumret_m3 | 100w | 84.83% | 2.791 | 16.02% |
| state_w3_cumret_m2 | 100w | 83.76% | 2.793 | 15.70% |

Both state-triggered candidates hurt test.

### Forward 2026-05-19 to 2026-06-16

| Mode | Capital | Ann | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Official | 50w | -61.03% | -2.997 | 11.23% |
| state_w5_cumret_m3 | 50w | -61.98% | -3.096 | 11.48% |
| state_w3_cumret_m2 | 50w | -61.41% | -3.041 | 11.60% |
| Official | 100w | -52.25% | -2.017 | 11.78% |
| state_w5_cumret_m3 | 100w | -47.75% | -1.790 | 11.18% |
| state_w3_cumret_m2 | 100w | -47.54% | -1.755 | 11.34% |

Forward is mixed:

- 50w gets worse;
- 100w improves, but remains strongly negative.

## Decision

Do not promote state-triggered target shrink.

Reason:

- validation improvement is small and comes with worse drawdown;
- test is consistently weaker than official;
- forward does not solve the 50w problem and only partially reduces 100w loss.

The trigger is too reactive: after the strategy has already lost money, shrinking the next few days does not reliably avoid the harmful period.

## Implication

The next trigger should be more predictive, not purely reactive.

Better candidates:

1. market breadth deterioration before the loss;
2. alpha-spread compression before weak returns;
3. high turnover plus low spread;
4. signal quality diagnostics, e.g. high concentration of chase/stall names in the top bucket;
5. combining market state and signal quality rather than portfolio drawdown alone.

Output:

```text
reports/state_triggered_target_20260617/state_triggered_target_results.csv
```
