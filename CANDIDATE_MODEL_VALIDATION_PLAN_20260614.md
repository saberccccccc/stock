# Candidate Model Validation Plan

Updated: 2026-06-14

## Objective

Determine whether the best purged V9 loss-ablation checkpoints improve the
executable 2024 strategy for CNY 500,000 and CNY 1,000,000 accounts.

Checkpoint selection metrics are screening signals only. Promotion is based on
the constrained portfolio backtest.

## Research Boundary

| Stage | Date range | Rule |
|---|---|---|
| Training labels | through 2023-12-31 | Already completed |
| Model and strategy validation | 2024-01-01 through 2024-12-31 | Used in this plan |
| Historical confirmation | 2025-01-01 through 2026-05-18 | Authorized after 2024 selection |
| Forward-only | from 2026-05-19 | Do not access in this plan |

No parameter may be changed after viewing the historical confirmation period.

## Candidate Checkpoints

| ID | Checkpoint | Reason |
|---|---|---|
| A5-E6 | `checkpoints_loss_ablation_A5/epochs/epoch_006.pt` | Best raw Top30 stability; spread loss |
| A0-E9 | `checkpoints_loss_ablation_A0_low_lr_e10/epochs/epoch_009.pt` | Best continued minimum-loss baseline |
| A4-E6 | `checkpoints_loss_ablation_A4/epochs/epoch_006.pt` | Best top-focus checkpoint and control |
| Frozen V9 | Existing 2024 Alpha | Operational reference only; split was contaminated |

## Fixed Alpha Construction

```text
predictor_mode=average
window=3
split=val
start=2024-01-01
end=2024-12-31
seeded model checkpoints as listed above
```

For each candidate, preserve both:

1. Raw `average_w3` ranking.
2. Ranking after demoting stocks with signal-day return at least 9.5%.

The surge-then-stall rule is excluded because its completed screen reduced
annualized return and Sharpe.

## Fixed Portfolio Configuration

```text
target_frac=0.006
hold_frac=0.10
rebalance_band=0.20
weight_mode=equal
max_weight=0.05
market_timing=legacy
legacy_bear_mult=0.70
legacy_crash_mult=0.30
minimum_ADV=CNY 3,000,000
lot_size=100
minimum_commission=CNY 5
limit_threshold=9.5%
```

Account sizes:

- CNY 500,000
- CNY 1,000,000

## Execution Scenarios

| Scenario | ADV cap | Cost multiplier | Extra lag |
|---|---:|---:|---:|
| Base | 5% | 1x | 0 |
| Capacity | 3% | 1x | 0 |
| Cost stress | 5% | 2x | 0 |
| Delay stress | 5% | 1x | 1 trading day |

Base costs are 1 bp commission, 5 bp stamp tax and 5 bp slippage.

## Metrics

- Annualized return, Sharpe and maximum drawdown.
- Executed and unfilled turnover.
- Blocked buys and sells.
- ADV-blocked and capped orders.
- Average number of holdings.
- Monthly return concentration.
- Rank overlap between candidates.

## Promotion Gate

A candidate can advance only when its 9.5%-filtered result satisfies:

| Metric | Gate |
|---|---:|
| CNY 500k base Sharpe | at least 1.50 |
| CNY 1m base Sharpe | at least 1.50 |
| Base annualized return | at least 45% |
| Maximum drawdown | no more than 20% |
| 2x-cost Sharpe | at least 1.20 |
| Extra-day-lag Sharpe | at least 0.95 |
| 3% ADV-cap degradation | no material collapse |

When candidates are close, prefer the simpler loss and require consistent
improvement for both account sizes. A validation win does not authorize use of
the 2025-2026 confirmation period automatically.

## Execution Order

1. Generate raw 2024 Alpha for A5-E6, A0-E9 and A4-E6.
2. Apply the fixed 9.5% signal-day return demotion.
3. Load market and ADV data once for all candidates.
4. Run base, 3% ADV, 2x-cost and one-day-lag scenarios.
5. Compare with the frozen V9 operational reference.
6. Produce a single CSV and Markdown decision report.
7. Lock the best new model using 2024 only.
8. Run one historical confirmation of that locked model and frozen V9 on
   2025-01-01 through 2026-05-18.

## Status

| Step | Status |
|---|---|
| Protocol and candidates frozen | Complete |
| Unified evaluator | Complete |
| Candidate Alpha generation | Complete |
| Constrained backtests | Complete |
| Stress tests | Complete |
| Promotion decision | Complete: no new candidate passed the standalone 2024 gate |
| Historical confirmation | Complete; old V9 comparison is contaminated and diagnostic only |

## Final Decision

No new candidate passed the standalone 2024 promotion gate. A4-E6 was the
strongest new candidate in the 2024 validation and was locked before the
historical confirmation period was accessed.

For 2025-01-01 through 2026-05-18, the base results were:

| Model | Capital | Annualized | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| A4-E6 | CNY 500,000 | 54.66% | 2.280 | 11.04% |
| Frozen V9 | CNY 500,000 | 58.15% | 2.378 | 11.90% |
| A4-E6 | CNY 1,000,000 | 59.68% | 2.372 | 10.97% |
| Frozen V9 | CNY 1,000,000 | 66.69% | 2.489 | 12.55% |

A4-E6 reduced drawdown modestly, but its annualized return and Sharpe were
lower for both account sizes. It also remained behind frozen V9 under the
2x-cost and one-trading-day-delay scenarios.

This is not a clean model comparison. The old V9 pipeline split valid dates
80%/20% by time. Its last 20% was approximately 2023-02-15 through 2026-04-29
and was used as `val` for checkpoint selection. The frozen V9 checkpoint was
selected at epoch 12 using `topic_h5_top10` on that period. Its 2025-2026
results therefore contain model-selection bias and must not be treated as
independent test evidence.

Frozen V9 may remain the already frozen operational strategy, but not because
this historical comparison proves that it generalizes better. A4-E6 is also
not promoted because it independently failed the 2024 Sharpe gate. These
confirmation results must not be used to retune A4-E6 or select another epoch.
