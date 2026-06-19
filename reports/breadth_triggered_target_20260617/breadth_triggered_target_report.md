# Breadth-Triggered Target Shrink Test 2026-06-17

## Purpose

The prior strategy-return trigger was too reactive. This round tested a more predictive signal known at signal-day close:

```text
market breadth = fraction of stocks with positive close-to-close return
```

If recent breadth is weak, the strategy shrinks target before next-open execution.

## Implementation

Added:

```text
run/make_breadth_triggered_target_alpha.py
```

Also added to open-ledger:

```text
--use-row-target-frac
```

This allows the backtester to read per-row `effective_target_frac` from alpha metadata. Without this flag, alpha transforms can only reorder names; they cannot truly shrink the target bucket.

## Candidates

Candidate A:

```text
breadth_ma3_040_r004
trigger: 3-day average up ratio <= 0.40
risk_target_frac=0.004
```

Candidate B:

```text
breadth_ma3_040_r003
trigger: 3-day average up ratio <= 0.40
risk_target_frac=0.003
```

Candidate C:

```text
breadth_ma5_040_r004
trigger: 5-day average up ratio <= 0.40
risk_target_frac=0.004
```

## Results

### Validation

| Mode | Capital | Ann | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Official | 50w | 84.08% | 1.834 | 21.07% |
| breadth_ma3_040_r004 | 50w | 82.97% | 1.811 | 19.84% |
| breadth_ma3_040_r003 | 50w | 68.63% | 1.579 | 21.72% |
| breadth_ma5_040_r004 | 50w | 81.60% | 1.778 | 21.75% |
| Official | 100w | 89.31% | 1.879 | 20.79% |
| breadth_ma3_040_r004 | 100w | 86.47% | 1.835 | 20.14% |
| breadth_ma3_040_r003 | 100w | 71.95% | 1.607 | 22.22% |
| breadth_ma5_040_r004 | 100w | 84.97% | 1.804 | 21.76% |

### Forward

Forward was not rerun for breadth candidates after validation failed, because the validation signal was already weaker than official. The earlier breadth diagnostic showed bad breadth aligns with poor returns, but shrinking the target on those dates did not improve the executable validation portfolio.

## Decision

Do not promote breadth-triggered target shrink.

Reason:

- `ma3<=0.40, target=0.004` lowers validation drawdown slightly, but also lowers annualized return and Sharpe.
- `target=0.003` is too aggressive.
- `ma5<=0.40` is weaker than ma3 and also worse than official.

The signal is informative diagnostically, but the action "shrink target" is not the right response by itself.

## Interpretation

Weak breadth days are dangerous, but reducing the number of names increases concentration and turnover. That can remove some bad names, but it also discards rebound candidates and forces extra rotation. The better response may be:

1. reduce gross exposure/market multiplier instead of shrinking names;
2. combine breadth with alpha-spread or turnover;
3. avoid new buys while allowing existing winners to stay;
4. use breadth to choose between official and a more conservative rebalance rule, not just target size.

Output:

```text
reports/breadth_triggered_target_20260617/breadth_and_true_state_target_results.csv
```
