# V9 Loss Ablation and Trading-Objective Plan

> 状态：历史消融计划；V9 不进入当前正式候选，本文件不再决定下一步。

Updated: 2026-06-13

## Objective

Identify which V9 loss components improve executable Top30 performance for
CNY 500,000 and CNY 1,000,000 accounts. A loss is retained only when it
improves raw validation returns or strict portfolio results, rather than only
raising normalized-label IC.

## Fixed Research Boundary

| Use | Dates |
|---|---|
| Training labels | no later than 2023-12-31 |
| Model and strategy validation | 2024-01-01 through 2024-12-31 |
| One-time confirmation | 2025-01-01 through 2026-05-18 |
| Forward-only | 2026-05-19 onward |

All experiments use the purged split. A signal date is excluded when any
required future label crosses its boundary.

## Fixed Hardware Configuration

```text
batch_size=2
validation_batch_size=1
accumulation_steps=8
num_workers=0
memmap_trim_interval=32
seed=42
```

The effective training batch is 16. Stop a run when available system memory
remains below 2 GB, process memory remains above 13 GB, or total GPU memory
remains above 7.2 GB.

## Metrics Recorded Per Epoch

### Prediction metrics

- Alpha IC.
- h1, h3, h5 and h7 IC.
- Top30 normalized-label return and IC.
- Top30 raw h1, h3, h5 and h7 return.
- Mean divided by standard deviation of daily Top30 raw returns.

### Loss components

- Global IC loss.
- Within-industry IC loss.
- Multi-horizon IC loss.
- Alpha-head diversity loss.
- Top-focus loss.
- Top-bottom spread loss.
- Pairwise loss when enabled.

### Execution diagnostics

- Signal-day mean and median return.
- Share of Top30 names with signal-day return at least 7% and 9.5%.
- Prior-day Top30 overlap.
- Blocked buys and sells.
- Turnover, unfilled turnover and explicit costs.

## Phase 1: Baseline A

The currently running 12-epoch model is the reference:

```text
industry_weight=0.10
multi_weight=0.30
diversity_weight=0.05
top_focus_weight=0.005, starts epoch 6
spread_weight=0.001, starts epoch 11
pairwise_weight=0
horizon_weights=0.15,0.25,0.35,0.25
```

Every epoch is saved. Completion of Baseline A does not automatically promote
its best checkpoint. The best two or three epochs are selected for strict
2024 execution backtests.

## Phase 2: Minimal Loss Ablation

Each experiment runs six epochs using the full purged training period and
the same random seed. Auxiliary losses start at epoch 3 in these short runs,
so their effect is observable.

| ID | Global IC | Industry IC | Multi IC | Diversity | Top Focus | Spread | Question |
|---|---:|---:|---:|---:|---:|---:|---|
| A0 | on | 0 | 0.30 | 0 | 0 | 0 | Minimum ranking baseline |
| A1 | on | 0.10 | 0.30 | 0 | 0 | 0 | Does industry IC help? |
| A2 | on | 0.10 | 0.30 | 0.05 | 0 | 0 | Does head diversity help? |
| A3 | on | 0.10 | 0.30 | 0.05 | 0.005 | 0 | Does top focus help Top30? |
| A4 | on | 0.10 | 0.30 | 0.05 | 0.005 | 0.001 | Does spread add long-only value? |

Only one component changes between adjacent experiments.

## Phase 3: Component Decision Rules

A newly added component is retained only when:

- Best raw h5 Top30 stability improves.
- Raw h5 Top30 mean return does not decline.
- Alpha IC does not decline by more than 0.02.
- The improvement appears in at least two epochs after activation.
- Top30 9.5%-surge share does not increase.
- Performance is not produced by one calendar month.

If adjacent experiments are statistically indistinguishable, prefer the
simpler loss.

## Phase 4: Trading Objectives

These experiments begin only after A0-A4 identify the useful existing
components.

### R1: Raw Top30 return loss

Use raw future h5 returns in a soft long-only objective:

```text
L_raw_top = -mean(sum(softmax(alpha / temperature) * raw_h5_return))
```

Initial registered values:

```text
weight=0.01
temperature=0.75
activation_epoch=3
```

The raw return target must be winsorized per cross-section to reduce
high-volatility domination. The original unmodified raw return remains in
evaluation metrics.

### R2: Chase penalty

Penalize model weight assigned to stocks already up at least 7% on the signal
date:

```text
L_chase = mean(sum(top_weight * relu(signal_day_return - 0.07)))
```

Initial registered values:

```text
weight=0.005
threshold=0.07
activation_epoch=3
```

The signal-day return must come from same-day or earlier information. The
9.5% execution filter remains unchanged and is not tuned in this phase.

## Phase 5: Strict Validation

For each surviving configuration:

1. Select at most three epochs using 2024 raw metrics.
2. Generate `average_w3` Alpha.
3. Apply the fixed 9.5% signal-day filter as a separate comparison.
4. Backtest Top30, `hold_frac=0.10`, 20% rebalance band.
5. Test CNY 500,000 and CNY 1,000,000.
6. Test 5% and 3% ADV caps, CNY 3m minimum ADV.
7. Test base costs, 2x costs and one extra trading-day delay.

## Promotion Gate

The final candidate must satisfy all conditions:

| Metric | Minimum |
|---|---:|
| CNY 500k validation Sharpe | 1.50 |
| CNY 1m validation Sharpe | 1.50 |
| Validation annualized return | 45% |
| Maximum drawdown | 20% or less |
| 2x-cost Sharpe | 1.20 |
| Extra-day-lag Sharpe | 0.95 |
| Top30 signal-day return at least 9.5% | 3.5% or less |

The candidate must also exceed the strictly isolated baseline for both
account sizes. Comparison with the old contaminated V9 is informative but
is not the formal promotion gate.

## Execution Order

1. Finish Baseline A.
2. Produce per-epoch loss and metric diagnostics.
3. Strict-backtest the best Baseline A epochs.
4. Run A0, A1, A2, A3 and A4 sequentially.
5. Reject or retain each added component.
6. Implement and run R1.
7. Implement and run R2 only if R1 is stable.
8. Strict-backtest surviving configurations.
9. Freeze one model or ensemble.
10. Run the 2025-2026 confirmation once, without parameter revision.

## Status Table

| ID | Status | Output |
|---|---|---|
| Baseline A | Running | `checkpoints_exp_purged_rawmetric_A_20260613` |
| A0 | Pending | `checkpoints_loss_ablation_A0` |
| A1 | Pending | `checkpoints_loss_ablation_A1` |
| A2 | Pending | `checkpoints_loss_ablation_A2` |
| A3 | Pending | `checkpoints_loss_ablation_A3` |
| A4 | Pending | `checkpoints_loss_ablation_A4` |
| R1 | Pending implementation | `checkpoints_loss_rawtop_R1` |
| R2 | Pending implementation | `checkpoints_loss_chase_R2` |
