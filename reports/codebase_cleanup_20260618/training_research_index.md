# Training Research Index 2026-06-18

This index consolidates the root-level training, loss-ablation, and candidate
validation documents that are still kept in manual review. It is an index only:
the source documents stay in place until their details are fully migrated or
explicitly archived.

## Source Documents

| Document | Role | Current cleanup decision |
|---|---|---|
| `LOSS_ABLATION_PLAN.md` | V9 loss and trading-objective protocol | Keep or consolidate |
| `PURGED_ALPHA_OPTIMIZATION_PLAN.md` | Purged Alpha execution-filter screen | Keep or consolidate |
| `CANDIDATE_MODEL_VALIDATION_PLAN_20260614.md` | Strict candidate checkpoint validation | Keep or consolidate |
| `EXPERIMENTS.md` | Early experiment-branch log | Archive only after consolidation |

## Research Boundaries

| Boundary | Rule |
|---|---|
| Strategy validation | 2024 only, unless a plan explicitly freezes a candidate first |
| One-time confirmation | 2025-01-01 through 2026-05-18, with no parameter revision |
| Forward period | After 2026-05-18, never used to tune checkpoints or filters |
| Account sizes | CNY 500,000 and CNY 1,000,000 |
| Formal checkpoint selection | Do not select solely by Alpha IC; require executable Top30 and stress checks |

## Key Conclusions

1. The 9.5% signal-day return filter is the strongest validated execution
   transform so far. It demotes names whose signal-day close-to-close return is
   at least 9.5%, reducing chase risk without using future information.

2. New purged checkpoints did not beat the standalone 2024 promotion gate. A4-E6
   was the strongest new candidate, but it was not promoted because its 2024
   result did not satisfy the formal Sharpe/return/stress requirements.

3. The old frozen V9 remains operationally useful, but its 2025-2026 historical
   confirmation is not clean model evidence. Its old 80%/20% time split selected
   the epoch using a validation range that overlaps the later confirmation
   window.

4. IC is useful as a floor, not as the final model-selection metric. The
   project records Alpha IC and horizon IC, but checkpoint selection must
   prioritize executable portfolio return, Sharpe, delay robustness, doubled
   cost robustness, drawdown, blocked buys, and capacity.

5. Loss additions did not yet prove a stable positive contribution. Top-focus
   and downside/chase-aware ideas remain plausible, but must be validated as
   separate, preregistered components rather than stacked until something looks
   good.

## Validated Execution Filter

From `PURGED_ALPHA_OPTIMIZATION_PLAN.md`:

| Validation scenario | CNY 500k annualized / Sharpe | CNY 1m annualized / Sharpe |
|---|---:|---:|
| Matching baseline | 56.59% / 1.575 | 58.83% / 1.597 |
| 9.5% filter | 59.56% / 1.625 | 60.81% / 1.622 |
| Baseline, 2x cost | 48.36% / 1.406 | 49.80% / 1.415 |
| 9.5% filter, 2x cost | 51.09% / 1.452 | 51.50% / 1.439 |
| Baseline, extra-day lag | 34.12% / 1.083 | 33.14% / 1.047 |
| 9.5% filter, extra-day lag | 40.16% / 1.224 | 41.04% / 1.221 |

Other notes:

- 3% ADV-cap produced the same result as 5% cap for both account sizes.
- Blocked buys fell from 78 to 70 for CNY 500k.
- Blocked buys fell from 80 to 68 for CNY 1m.
- Maximum drawdown increased by less than one percentage point.

## Candidate Validation Outcome

From `CANDIDATE_MODEL_VALIDATION_PLAN_20260614.md`:

| Model | Capital | Annualized | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| A4-E6 | CNY 500,000 | 54.66% | 2.280 | 11.04% |
| Frozen V9 | CNY 500,000 | 58.15% | 2.378 | 11.90% |
| A4-E6 | CNY 1,000,000 | 59.68% | 2.372 | 10.97% |
| Frozen V9 | CNY 1,000,000 | 66.69% | 2.489 | 12.55% |

Interpretation:

- A4-E6 reduced drawdown modestly.
- A4-E6 lagged frozen V9 in return and Sharpe for both account sizes.
- A4-E6 remained behind under doubled-cost and one-day-delay stress.
- The frozen V9 comparison is diagnostic, not clean evidence, because of the
  old checkpoint-selection contamination.

## Loss-Ablation Protocol

From `LOSS_ABLATION_PLAN.md`, every serious candidate should be checked on:

| Metric group | Required evidence |
|---|---|
| IC floor | Alpha IC plus h1/h3/h5/h7 IC |
| Top30 quality | Top30 normalized-label return and raw horizon returns |
| Stability | Mean divided by standard deviation of daily Top30 raw returns |
| Execution risk | Signal-day 7% and 9.5% surge shares, prior-day Top30 overlap |
| Portfolio realism | CNY 500k/1m, ADV caps, doubled cost, one-day delay |

Registered loss/component families:

| ID/family | Purpose | Current handling |
|---|---|---|
| A0 | Global IC plus baseline objective | Baseline/reference |
| A1 | Add within-industry IC | Validate only if executable metrics improve |
| A2 | Add diversity | Validate for Top30 stability, not IC alone |
| A3 | Add Top-focus | Plausible, but not yet proven as standalone positive |
| A4/A5 | Add spread-style objectives | Treat cautiously; spread can dominate IC |
| R1 | Raw Top30 return loss | Pending as preregistered trading-objective test |
| R2 | Chase penalty | Pending; should use same-day or earlier signal-day return only |

Promotion gate:

| Metric | Minimum |
|---|---:|
| CNY 500k validation Sharpe | 1.50 |
| CNY 1m validation Sharpe | 1.50 |
| Validation annualized return | 45% |
| Maximum drawdown | 20% or less |
| 2x-cost Sharpe | 1.20 |
| Extra-day-lag Sharpe | 0.95 |
| Top30 signal-day return at least 9.5% | 3.5% or less |

## Cleanup Decisions

| Document | Keep now? | Future action |
|---|---|---|
| `LOSS_ABLATION_PLAN.md` | Yes | Keep until R1/R2 status is resolved or fully indexed |
| `PURGED_ALPHA_OPTIMIZATION_PLAN.md` | Yes | Keep as source evidence for the 9.5% filter |
| `CANDIDATE_MODEL_VALIDATION_PLAN_20260614.md` | Yes | Keep as source evidence for A4-E6 and frozen V9 caveat |
| `EXPERIMENTS.md` | Temporarily | Archive after its early branch notes are either obsolete or copied into CLAUDE/this index |

## Next Research Cleanup Step

Create a checkpoint/loss result ledger that links each training output directory
to its decision:

```text
checkpoints_loss_ablation_A*
checkpoints_exp_purged_rawmetric_A_20260613
candidate_model_validation_20260614
loss_ablation_portfolio_validation_20260614
multi_loss_validation_20260614
```

Do this before moving checkpoint or experiment-output directories.
