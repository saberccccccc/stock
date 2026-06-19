# Purged Alpha Optimization Plan

Updated: 2026-06-13

## Objective

Improve annualized return and Sharpe for CNY 500,000 and CNY 1,000,000
accounts without exceeding 16 GB RAM or 8 GB VRAM, while preserving a clean
out-of-sample boundary.

## Fixed Boundaries

- Training labels end on or before 2023-12-31.
- Model validation labels end on or before 2024-12-31.
- Strategy selection uses only 2024 validation data.
- Dates from 2025-01-01 through 2026-05-18 are reserved for one-time
  confirmation after a candidate is frozen. They are strictly independent
  only for checkpoints trained and selected with the purged split.
- Dates from 2026-05-19 onward remain forward-only.
- Tushare tokens already stored in project documents may remain unchanged.
- Training uses one process, `num_workers=0`, and memory-mapped data.
- Stop training if system RAM remains above 13 GB or VRAM remains above
  7.2 GB.

## Baselines

### Frozen practical baseline

- Checkpoint: `checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt`
- Signal: `average_w3`
- Selection: `target_frac=0.006`, `hold_frac=0.10`
- Execution: legacy market timing, 20% rebalance band, 100-share lots,
  CNY 5 minimum commission, 5% ADV cap, CNY 3m minimum ADV.

### Strictly isolated model

- Checkpoint:
  `C:\Users\x\Documents\股票预测\checkpoints_exp_purged_20260613\ultimate_v7_best.pt`
- Best epoch: 9
- Training dates: 2010-05-05 through 2023-12-15
- Validation dates: 2024-01-02 through 2024-12-17
- Result: safe resource usage, but strict 2024 execution performance was far
  below the practical baseline. It is not eligible for independent testing.

## Experiment Sequence

| Stage | Work | Data allowed | Pass condition |
|---|---|---|---|
| P01 | Audit split and resource use | Through 2024 | No label leakage; no OOM |
| P02 | Compare chase risk, blocked buys and rank stability | 2024 only | Explain the execution gap |
| P03 | Add signal-day tradability filter | 2024 only | No future price fields |
| P04 | Add lagged rank stabilization | 2024 only | Uses current and prior signals only |
| P05 | Screen a fixed small candidate set | 2024 only | Improve both account sizes |
| P06 | Stress costs, delay and capacity | 2024 only | Survive 2x cost and one-day delay |
| P07 | Freeze one candidate | No test access | Parameters and hashes recorded |
| P08 | Run one-time historical confirmation | 2025-01-01 to 2026-05-18 | No parameter revision |
| P09 | If P05 fails, redesign training objective | Through 2024 | New purged training run |

Current status:

- P01 through P07: complete.
- P08: complete as a one-time historical confirmation. Because the old V9
  checkpoint was selected with a contaminated validation split, this result
  is not a strictly independent model test.
- P09: required for the next model round because the strictly isolated
  epoch-9 model did not meet the validation gate.

## Pre-Registered Alpha Candidates

The screen is deliberately small:

1. Baseline Alpha with signal-day return at or above 9.5% demoted.
2. Baseline Alpha with signal-day return at or above 7.0% demoted.
3. Baseline Alpha with 75% current rank and 25% mean of the prior two ranks.
4. Candidate 1 plus candidate 3.
5. The same combined transform on the purged epoch-9 Alpha, for diagnosis
   only unless it closes the full performance gap.

No threshold is added after seeing candidate results.

## Promotion Gate

A candidate advances only if all conditions hold:

- Annualized return and Sharpe improve versus the matching 20% band baseline
  for both CNY 500,000 and CNY 1,000,000.
- Maximum drawdown is not worse by more than 2 percentage points.
- Blocked buys decrease or remain unchanged.
- Validation 2x-cost Sharpe does not fall below the baseline.
- One-extra-day execution does not show a material collapse relative to the
  baseline.

If no candidate passes, the execution transform is rejected and the next
work item is a purged retraining run with a more execution-aware objective
and checkpoint selection process.

## Completed Screen

The 9.5% signal-day return filter was the only candidate that passed all
validation gates. It demotes, rather than removes, stocks whose close-to-close
return on the signal date is at least 9.5%.

| Validation scenario | CNY 500k ann / Sharpe | CNY 1m ann / Sharpe |
|---|---:|---:|
| Matching baseline | 56.59% / 1.575 | 58.83% / 1.597 |
| 9.5% filter | 59.56% / 1.625 | 60.81% / 1.622 |
| Baseline, 2x cost | 48.36% / 1.406 | 49.80% / 1.415 |
| 9.5% filter, 2x cost | 51.09% / 1.452 | 51.50% / 1.439 |
| Baseline, extra-day lag | 34.12% / 1.083 | 33.14% / 1.047 |
| 9.5% filter, extra-day lag | 40.16% / 1.224 | 41.04% / 1.221 |

The 3% ADV-cap check produced the same results as the 5% cap for both account
sizes. Blocked buys fell from 78 to 70 for CNY 500k and from 80 to 68 for CNY
1m. Maximum drawdown increased by less than one percentage point.

The frozen one-time historical confirmation for 2025-01-01 through
2026-05-18 produced:

- CNY 500k: 58.15% annualized, 2.378 Sharpe, 11.90% maximum drawdown.
- CNY 1m: 66.69% annualized, 2.489 Sharpe, 12.55% maximum drawdown.

These confirmation numbers are useful for comparing execution rules but are
not clean model evidence because the underlying old checkpoint participated
in the previously contaminated selection process.

## Resource Policy

- Alpha transforms run on CPU and stream per-stock CSV files.
- Only required columns are loaded from raw price files.
- Candidate backtests run sequentially.
- No dataset cache rebuild is allowed during this round.
- GPU training remains paused until the validation screen is complete.
