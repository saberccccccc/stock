# M0 Candidate Reranker Implementation Plan

Date: 2026-06-14

## Objective

Build a leakage-controlled second-stage stock ranker without changing the M0
training loss. M0 remains the broad opportunity model. The second-stage model
only reranks M0 candidates, while market exposure, liquidity and execution
constraints remain outside the prediction model.

## Training Metrics versus Model Selection

Training and model selection serve different purposes and must not be mixed.

The M0 base models are trained with weighted cross-sectional IC. The
LightGBM reranker is trained with LambdaRank/NDCG inside each daily candidate
group. These objectives provide gradients and make training possible, but
neither IC nor NDCG is the final strategy-selection objective.

Candidate models and blend weights are promoted according to executable
portfolio evidence. Alpha IC is retained only as a model-health threshold.

## Formal Model-Selection Priority

Apply the following order when comparing M0 with reranked candidates on 2024.

### Priority 1: Executable portfolio performance

For both CNY 500,000 and CNY 1,000,000:

- Compare annualized return and Sharpe after all execution constraints.
- Neither base-scenario Sharpe may fall by more than 0.03 versus M0.
- At least one account size must improve annualized return.
- A candidate that improves IC or NDCG but reduces executable portfolio
  quality is rejected.

### Priority 2: Delay and cost robustness

- One-trading-day-delay Sharpe must improve for both account sizes.
- Doubled-cost Sharpe may not fall by more than 0.03.
- Results must not depend on same-day perfect execution.

### Priority 3: Risk, turnover and capacity

- Maximum drawdown may not increase by more than two percentage points.
- Average turnover may not increase by more than 20%.
- The 3% ADV capacity scenario must pass for both account sizes.
- Report blocked buys, blocked sells and unfilled turnover.

### Priority 4: Top-book quality and stability

- Report executable Top30 return rather than relying only on full-universe
  rank statistics.
- Report monthly performance, worst quarter and performance by market regime.
- At least seven of twelve 2024 months should be no worse than M0.
- Acute-rise/stall concentration in the final Top30 may not increase
  materially.
- Improvement should also be visible across multiple 2018-2023 OOF years,
  rather than coming from one exceptional year.

### Priority 5: Alpha IC health check

- Alpha IC must remain positive.
- The preferred minimum health threshold is 0.07.
- Alpha IC is not included in a weighted final model score.
- High IC cannot compensate for poor portfolio performance.
- Low but positive IC does not automatically reject a model if Top30 and
  executable portfolio results are robust.

The same treatment applies to LambdaRank NDCG: it is a diagnostic and training
metric, not sufficient evidence for strategy promotion.

The first production candidate is:

```text
M0 weighted-IC Alpha
    -> top 10% candidate pool
    -> LightGBM LambdaRank reranker
    -> blend M0 and reranker percentile ranks
    -> existing 9.5% chase filter and constrained portfolio engine
```

## Non-negotiable Data Protocol

| Purpose | Allowed dates |
|---|---|
| OOF reranker training rows | Historical folds ending no later than 2023-12-31 |
| Model and blend selection | 2024-01-01 through 2024-12-31 |
| Final historical confirmation | Frozen until the 2024 design is locked |
| Forward evidence | 2026-05-19 onward; never used for tuning |

M0 predictions used to train the reranker must be out of fold. A checkpoint
may not generate reranker training features for dates that were included in
that checkpoint's own training labels.

## Responsibilities

### M0

- Learn a broad cross-sectional return ranking.
- Reduce the full universe to a candidate pool.
- Supply Alpha level, rank and signal-history features.

### Reranker

- Improve ordering inside the M0 candidate pool.
- Concentrate true high-return names near the top.
- Prefer signals that remain useful after realistic execution delay.

### Portfolio and execution layer

- Apply the market exposure multiplier.
- Apply the 9.5% signal-day chase filter.
- Enforce lot size, minimum commission, ADV and rebalance-band constraints.
- Measure costs, blocked trades and drawdown.

The reranker must not learn future market-state or execution information.

## Label Definition

Use the same weighted future target as M0:

```text
target = 0.15*h1 + 0.25*h3 + 0.35*h5 + 0.25*h7
```

For each signal date, first use M0 to form the candidate pool. Compute the
target rank inside that candidate pool, then assign graded relevance:

| Candidate-pool target rank | Relevance |
|---|---:|
| Top 6% | 4 |
| 6% to 15% | 3 |
| 15% to 30% | 2 |
| 30% to 60% | 1 |
| Remaining | 0 |

Each trading date is one LambdaRank query group. Graded labels are preferred
to a sparse Top30 binary label. With an approximately 500-name Top10%
candidate pool, the Top6% relevance bucket contains roughly 30 names.

Full-universe future ranks are retained only as audit fields. They measure how
much of the unknowable future global Top30 was present in the candidate pool,
but are not appropriate as the reranker label because the reranker can only
order names it receives.

## Candidate Pool

Start with M0 Top 10%.

Before training, the candidate-pool audit must report full-universe Top 0.6%
and Top 2% recall and compare them with the random baselines implied by the
candidate fraction. Recall is diagnostic, not a hard promotion gate. Pool
sizes of 10%, 15% and 20% may be compared only after the OOF pipeline works.

## Initial Features

### M0 signal features

- Current smoothed M0 Alpha.
- Cross-sectional Alpha percentile.
- One-day and three-day Alpha change.
- One-day and three-day rank change.
- Three-day Alpha mean and distance from that mean.

### Existing point-in-time model inputs

- The 250 normalized V9 input features.
- The 59 stock/market/macro risk features.
- Industry identifier.

No future return, relevance label, future rank or post-signal execution result
may be included as a feature.

Horizon-head predictions and hidden embeddings are deferred. They are not
required for the first baseline and would add extraction and overfitting risk.

## Phase Plan

### Phase 0: Audit and specification

Deliverables:

- This plan.
- Confirmed LightGBM LambdaRank availability.
- Confirmed M0 checkpoint and data boundaries.
- Explicit feature and non-feature column lists.

Exit condition:

- No unresolved date-boundary or label-availability issue.

### Phase 1: Build the 2024 validation dataset

Purpose:

- Validate schema, label distribution, candidate recall, memory usage and
  daily group construction.
- Do not train and validate a model on this same dataset.

Deliverables:

- Candidate-level Parquet dataset.
- Dataset configuration JSON.
- Daily audit CSV.
- Summary JSON containing row count, date range, memory estimate, label
  distribution and candidate recall.

Exit conditions:

- Dates are restricted to 2024.
- Exactly one rank group exists per signal date.
- No forbidden feature column is present.
- Candidate Top 0.6% recall is measured.

### Phase 2: Generate historical OOF features

Proposed expanding folds:

| Fold | M0 training labels through | OOF prediction year |
|---|---|---|
| F1 | 2017-12-31 | 2018 |
| F2 | 2018-12-31 | 2019 |
| F3 | 2019-12-31 | 2020 |
| F4 | 2020-12-31 | 2021 |
| F5 | 2021-12-31 | 2022 |
| F6 | 2022-12-31 | 2023 |

Each fold reuses the M0 configuration, seed and six-epoch ceiling. Fold
checkpoints and predictions are stored separately. They must never access the
2024 selection period.

Resource controls for the 16 GB machine:

- Train folds sequentially.
- Batch size 4, validation batch size 2, accumulation 4.
- Write candidate rows per fold to Parquet.
- Do not concatenate all raw model tensors in memory.
- Train LightGBM from candidate rows only, not the full universe.

Exit conditions:

- Every training row has a checkpoint cutoff earlier than its signal date.
- No duplicate date/code rows.
- Fold-level candidate recall is stable enough to support reranking.

### Phase 3: Train the first LambdaRank baseline

Initial parameters:

```text
objective = lambdarank
metric = ndcg
eval_at = 30, 50, 100
n_estimators = 400
learning_rate = 0.03
num_leaves = 31
min_child_samples = 200
feature_fraction = 0.80
bagging_fraction = 0.80
lambda_l2 = 2.0
```

Train on 2018-2023 OOF rows. Use date groups and never randomly split rows
across dates.

Offline acceptance checks:

- NDCG@30 and NDCG@50 exceed M0 ordering.
- Top relevance bucket has monotonically higher realized target.
- Improvement is present in at least four of six OOF years.
- Feature importance is not dominated by a suspicious date or future field.

### Phase 4: Create 2024 reranked Alpha

Convert both scores to daily percentiles:

```text
final_score = w_alpha * alpha_percentile
            + (1 - w_alpha) * reranker_percentile
```

Test only the preregistered grid:

```text
w_alpha = 1.00, 0.75, 0.50, 0.25, 0.00
```

Do not tune arbitrary weights after inspecting individual months.

### Phase 5: Strict 2024 portfolio validation

For CNY 500,000 and CNY 1,000,000, run:

- Base costs.
- Doubled costs.
- One-trading-day delay.
- 3% ADV capacity.
- Existing 9.5% chase filter.

Report:

- Annualized return and Sharpe.
- Maximum drawdown and Calmar.
- Turnover and total cost.
- Blocked buys and unfilled turnover.
- Monthly win rate and worst quarter.
- Performance by market regime.
- Acute-rise/stall concentration in the final Top30.

Promotion rule:

- Apply the formal model-selection priority defined above.
- Portfolio performance and execution robustness take precedence over IC and
  NDCG.
- Alpha IC is checked only as a positive, preferably at least 0.07, health
  threshold.

### Phase 6: Decision

If the independent reranker passes, freeze its dataset, model and blend
weight. Only then consider a neural reranker or partial end-to-end fine-tuning.

After the 2024 design and blend weight are frozen, run one historical
confirmation covering 2025-01-01 through 2026-05-18. Do not change the model,
features, labels, blend weight or execution rules in response to that
confirmation result. Observations from 2026-05-19 onward remain forward-test
evidence and are never used for research tuning.

If it fails, retain M0 and analyze:

- Candidate-pool recall.
- Label instability by year.
- Feature importance and score buckets.
- Whether the failure is ranking, turnover or execution related.

Do not respond to failure by immediately adding more losses to M0.

## Estimated Runtime

| Work | Estimate |
|---|---:|
| 2024 validation dataset and audit | 5-15 minutes |
| One historical M0 fold | about 30-45 minutes |
| Six sequential OOF folds | about 3-5 hours |
| LambdaRank training and diagnostics | 10-30 minutes |
| 2024 blend and execution sweep | 10-30 minutes |

The OOF fold stage is the expensive but necessary part. A faster in-sample
shortcut may be used only as a software smoke test and cannot support a model
selection decision.

## Execution Status and 2024 Decision (2026-06-15)

Completed:

- Built six pure OOF folds covering 2018-2023.
- Built 555,579 candidate rows across 1,457 signal dates with 319 features.
- Trained the first LightGBM LambdaRank model.
- Selected iteration 88 on the 2023 temporal validation split and refit on
  all 2018-2023 OOF rows.
- Ran the preregistered 2024 blend grid for CNY 500,000 and CNY 1,000,000
  under base cost, doubled cost, one-day delay and 3% ADV capacity scenarios.

The ranker's 2023 offline NDCG improved over M0:

| Ordering | NDCG@30 | NDCG@50 | NDCG@100 |
|---|---:|---:|---:|
| M0 | 0.1718 | 0.1997 | 0.2552 |
| LambdaRank | 0.2235 | 0.2591 | 0.3200 |

However, the NDCG improvement did not translate into an executable portfolio
improvement. The following table uses the existing 9.5% chase filter and CNY
500,000:

| M0 weight | Base ann. | Base Sharpe | Cost 2x ann. | Lag 1 ann. | Base MDD |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 45.73% | 1.327 | 40.19% | 26.10% | 17.41% |
| 0.75 | 24.07% | 0.798 | 17.59% | 9.55% | 22.78% |
| 0.50 | 20.99% | 0.721 | 13.52% | 8.11% | 22.67% |
| 0.25 | 22.74% | 0.759 | 14.93% | 11.29% | 20.76% |
| 0.00 | 26.56% | 0.843 | 17.44% | 16.76% | 20.76% |

The CNY 1,000,000 results lead to the same decision. The 3% ADV scenario is
identical to the base scenario for this capital range, so capacity is not the
cause of failure.

Decision:

- Reject `lambdarank_v1` for promotion.
- Retain pure M0 with the 9.5% chase filter as the current baseline.
- Do not access 2025 or later data for this failed design.
- Treat the result as evidence that candidate-relative NDCG/relevance labels
  are not sufficiently aligned with the actual Top30 portfolio objective.
- Do not add this reranker to production and do not compensate by selecting an
  arbitrary post-hoc blend weight.

Artifacts:

- Model: `reranker_models_20260615/lambdarank_v1/reranker_model.pkl`
- Training report:
  `reranker_models_20260615/lambdarank_v1/training_summary.json`
- Portfolio report:
  `reranker_validation_20260615/reranker_validation_summary.csv`

Next diagnostic stage:

1. Measure the ranker's daily overlap with M0 Top30 and the realized return of
   promoted versus demoted names.
2. Break the loss down by turnover, score bucket, market regime and
   acute-rise/stall exposure.
3. Test whether the candidate-relative relevance label rewards broad
   cross-sectional ordering while misranking the very small executable Top30.
4. Only after diagnosis, define a second label based on executable net return,
   downside and one-day persistence. Keep this as a new preregistered
   experiment rather than tuning the failed 2024 result.

### Replacement diagnosis (2026-06-15)

The first diagnostic is complete. Compared with M0 Top30 after the 9.5% chase
filter, the ranker changes far too much of the portfolio:

| M0 weight | Mean names replaced | Top30 overlap | Promoted minus demoted target | Positive replacement days |
|---:|---:|---:|---:|---:|
| 0.75 | 19.25 | 35.83% | -0.0858 | 42.15% |
| 0.50 | 22.51 | 24.97% | -0.0826 | 43.39% |
| 0.25 | 24.16 | 19.46% | -0.0724 | 44.63% |
| 0.00 | 24.93 | 16.89% | -0.0615 | 44.21% |

For the least aggressive blend, the promoted names also underperform the
demoted names on the stored H5 target by -0.0949. Seven of twelve months have
a negative replacement delta. The discrete relevance grade is slightly higher
for promoted names even though their continuous future target is lower. This
confirms that the candidate-relative relevance buckets and broad NDCG metric
are misaligned with the narrow executable Top30 decision.

The `market_regime` feature is also unusable in this dataset: it is zero for
every row in all six OOF years and in 2024. The existing detector therefore
contributed a constant feature and must be replaced or removed before another
dataset is built.

Diagnostic artifacts:

- `run/diagnose_reranker_replacements.py`
- `reranker_validation_20260615/replacement_diagnostics/replacement_summary.csv`
- `reranker_validation_20260615/replacement_diagnostics/replacement_monthly.csv`
- `reranker_validation_20260615/replacement_diagnostics/replacement_details.csv`

### Preregistered direction for reranker v2

Do not train v2 until its dataset rules are implemented and audited:

1. Change the task from broad candidate ranking to conservative Top30 boundary
   correction.
2. Freeze the strongest M0 names and allow at most 3-6 replacements per day.
3. Use a continuous, date-normalized target instead of candidate-relative
   relevance buckets.
4. Define the target from executable forward return, with explicit penalties
   for downside, one-day signal decay and expected trading cost.
5. Replace the constant regime field with index-based trailing market features
   that are known at signal time.
6. Evaluate replacement delta and Top30 portfolio metrics during temporal
   validation; NDCG remains diagnostic only.
7. Run one preregistered 2024 validation after the design is frozen. Do not
   search post-hoc blend weights on 2024.

### Conservative boundary probe (2026-06-15)

Before rebuilding the label, the existing v1 model was tested as a controlled
boundary corrector. The preregistered probe protected the M0 core, limited the
competition set to approximately M0 ranks 25-80, and allowed at most 3 or 6
Top30 replacements.

With the 9.5% chase filter:

| Variant | Capital | Base ann. | Sharpe | Cost 2x ann. | Lag 1 ann. |
|---|---:|---:|---:|---:|---:|
| M0 | 500k | 45.73% | 1.327 | 40.19% | 26.10% |
| Max 3 replacements | 500k | 46.08% | 1.334 | 40.25% | 26.18% |
| M0 | 1m | 46.27% | 1.316 | 40.83% | 25.56% |
| Max 3 replacements | 1m | 48.61% | 1.365 | 40.39% | 25.94% |
| Max 6 replacements | 500k | 45.01% | 1.313 | 38.71% | 24.33% |
| Max 6 replacements | 1m | 46.46% | 1.323 | 40.25% | 24.45% |

Interpretation:

- Limiting the ranker to about three replacements removes nearly all of the
  catastrophic v1 degradation and produces a small base-case improvement.
- Six replacements are already too aggressive.
- The three-replacement improvement is not yet promotion quality: doubled-cost
  performance is mixed, most monthly differences are close to zero, and the
  500k gain is concentrated mainly in January and April.
- Promoted names still have a lower continuous future target than demoted
  names by -0.0426 on average, with a positive replacement delta on only
  44.80% of valid days. The old relevance label remains misaligned.

Decision:

- Keep this result as evidence that conservative boundary reranking is viable.
- Do not promote the v1 model or the max-3 probe.
- Build v2 with a continuous executable target and select it on OOF
  replacement delta plus portfolio robustness.

Artifacts:

- `run/validate_conservative_reranker_2024.py`
- `reranker_validation_20260615/conservative_v1/summary.csv`
- `reranker_validation_20260615/conservative_v1/boundary_audit.csv`
- `reranker_validation_20260615/conservative_v1/replacement_diagnostics/`

### Continuous executable-target V2 result (2026-06-15)

V2 was implemented without using 2024 for model fitting or iteration
selection:

- Training rows: M0 boundary ranks 28-80 from pure OOF years 2018-2023.
- Target: continuous realized 1/3/5/10-day return, one-day-delayed return,
  maximum downside, round-trip cost and next-day blocked-buy penalty.
- Market features: trailing HS300 returns, volatility, drawdown and moving
  average gaps known on the signal date.
- Temporal selection: train through 2022 and choose 340 trees on 2023
  replacement delta.
- Portfolio action: protect M0 ranks 1-27 and select only three names from
  ranks 28-80.
- The final twelve or so dates of each year have no label when a complete
  10-day outcome would cross into the next calendar year. In particular, no
  2025 price was used to construct the 2024 validation labels.

Strict 2024 result with the existing 9.5% chase filter:

| Model | Capital | Base ann. | Sharpe | Cost 2x ann. | Lag 1 ann. | MDD |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 500k | 45.73% | 1.327 | 40.19% | 26.10% | 17.41% |
| V2 | 500k | 45.23% | 1.316 | 40.28% | 27.07% | 17.41% |
| M0 | 1m | 46.27% | 1.316 | 40.83% | 25.56% | 17.53% |
| V2 | 1m | 46.65% | 1.325 | 40.49% | 26.09% | 17.58% |

The stock-level replacement target generalizes in the correct direction:

- 227 labelled 2024 replacement days.
- Mean promoted-minus-demoted executable target: +0.00660.
- Positive replacement days: 53.74%.
- Median replacement delta: +0.00752.

Decision:

- V2 is materially better aligned than V1, but it does not pass the portfolio
  promotion rule because base and doubled-cost results are mixed.
- Retain M0 as the production baseline.
- Preserve V2 as the first valid research candidate rather than discarding the
  reranking approach.
- The next version must align labels with the retention portfolio itself:
  current holdings, expected holding age, replacement threshold and marginal
  benefit after turnover. Do not tune the frozen V2 target weights on 2024.

Artifacts:

- `run/build_reranker_v2_dataset.py`
- `run/train_reranker_v2.py`
- `run/validate_reranker_v2_2024.py`
- `reranker_v2_data_20260615/`
- `reranker_models_20260615/regression_v2/`
- `reranker_validation_20260615/regression_v2/`

### State-aware marginal-fill V3 result (2026-06-15)

V3 aligns the reranker with the actual retention state machine:

- Existing holdings are retained while they remain inside the M0 hold pool
  (approximately the top 10%).
- The model acts only when the portfolio has vacancies.
- M0 keeps any high-priority fills beyond the final three vacancy slots.
- V3 selects at most the last three fills from M0 ranks below 80.
- The model was fit on 2018-2022 OOF rows and selected at 200 trees using 2023
  marginal fill improvement. No 2024 result was used to select the model.

2023 temporal selection:

- Mean V3-minus-M0 fill target: +0.00554.
- Positive fill days: 53.15%.

Strict 2024 result:

| Model | Capital | Base ann. | Sharpe | Cost 2x ann. | Lag 1 ann. | MDD |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 500k | 45.73% | 1.327 | 40.19% | 26.10% | 17.41% |
| V3 | 500k | 52.67% | 1.538 | 46.04% | 42.18% | 15.35% |
| M0 | 1m | 46.27% | 1.316 | 40.83% | 25.56% | 17.53% |
| V3 | 1m | 54.13% | 1.542 | 46.27% | 43.31% | 15.58% |

Execution checks:

- 3% ADV results equal the base results at both capital levels.
- Average turnover does not increase:
  - 500k: 0.2441 to 0.2413.
  - 1m: 0.2474 to 0.2466.
- Total normalized cost decreases slightly.
- Blocked buys fall from 52 to 38.
- The model changes an average of 2.43 actual fill decisions per signal date.

2024 label confirmation:

- 220 labelled fill dates.
- Mean V3-minus-M0 executable target: +0.00691.
- Median target improvement: +0.00174.
- Positive fill days: 53.18%.

Decision:

- Promote V3 from research experiment to frozen candidate.
- Keep M0 available as the fallback baseline.
- Freeze the V3 target formula, 200-tree model, feature set, state-machine
  logic, rank-80 candidate boundary and maximum three marginal fills.
- Run one untouched historical confirmation from 2025-01-01 through
  2026-05-18. Do not alter V3 in response to that confirmation.
- Observations from 2026-05-19 onward remain reserved for forward testing.

Artifacts:

- `run/build_reranker_v3_dataset.py`
- `run/train_reranker_v3.py`
- `run/validate_reranker_v3_2024.py`
- `reranker_v3_data_20260615/`
- `reranker_models_20260615/regression_v3/`
- `reranker_validation_20260615/regression_v3/`

### Frozen post-2024 historical confirmation

The frozen V3 was applied once to the untouched period after 2024. Because the
current cached cross-section pipeline requires complete ten-day labels, and no
data after 2026-05-18 may be accessed, the usable signal range is
2025-01-02 through 2026-04-29 (319 signal dates). No parameter was changed
after viewing this result.

| Model | Capital | Base ann. | Sharpe | Cost 2x ann. | Cost 2x Sharpe | Lag 1 ann. | Lag 1 Sharpe | Base MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 500k | 41.85% | 1.897 | 33.40% | 1.582 | 46.95% | 2.049 | 11.77% |
| V3 | 500k | 40.02% | 2.053 | 32.35% | 1.723 | 42.02% | 2.107 | 11.12% |
| M0 | 1m | 43.65% | 1.923 | 35.03% | 1.605 | 50.11% | 2.122 | 11.73% |
| V3 | 1m | 43.56% | 2.139 | 35.36% | 1.804 | 45.91% | 2.202 | 10.75% |

Additional confirmation evidence:

- Mean executable fill-target improvement: +0.00392 over 309 labelled dates.
- Positive fill-target days: 53.07%.
- Turnover falls from 0.2975 to 0.2725 at 500k and from 0.3056 to 0.2792
  at 1m.
- Blocked buys fall from 35 to 16 in the base scenario.
- Base-case drawdown and Sharpe improve at both capital levels.
- Raw annualized return is mixed: lower at 500k and effectively unchanged at
  1m. Lag-one raw return is lower despite higher lag-one Sharpe.

Final deployment interpretation:

- V3 is a valid risk-adjusted reranker, not a proven raw-return replacement for
  M0.
- Keep M0 as the live fallback and benchmark.
- Freeze V3 exactly as tested and run it in parallel/shadow mode on observations
  from 2026-05-19 onward.
- Do not create a capital-specific switch or retune V3 from this confirmation
  result.
- Promote V3 to the live decision path only after forward evidence confirms
  that its lower turnover, lower blocked-buy count and higher Sharpe persist
  without an unacceptable return sacrifice.

Confirmation artifacts:

- `run/confirm_reranker_v3_history.py`
- `reranker_data_20260614/m0_confirmation_2025_20260518/`
- `reranker_v3_data_20260615/m0_confirmation_2025_20260518/`
- `reranker_confirmation_20260615/regression_v3/`

### First true forward shadow result (through 2026-06-11)

The frozen M0 epoch-6 checkpoint and frozen V3 were regenerated strictly from
`data/forward_raw`. Signals from 2026-05-12 through 2026-05-18 were used only
to warm the retention state. Performance was measured on 17 realized return
days from 2026-05-20 through 2026-06-11.

| Model | Capital | Realized return | Sharpe | MDD | Avg turnover | Blocked buys |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 500k | -9.18% | -4.290 | 8.40% | 0.3827 | 9 |
| V3 | 500k | -9.34% | -6.244 | 9.02% | 0.3683 | 6 |
| M0 | 1m | -9.24% | -3.996 | 9.09% | 0.4032 | 10 |
| V3 | 1m | -10.32% | -6.395 | 9.99% | 0.3927 | 8 |

Interpretation:

- Both strategies performed poorly in this short forward market window.
- V3 continues to reduce turnover and blocked buys.
- V3 underperformed M0 by 0.15 percentage points at 500k and 1.08 percentage
  points at 1m.
- V3 also had worse forward Sharpe and drawdown.
- The largest relative losses occurred around 2026-06-02, 2026-06-03 and
  2026-06-09.
- Seventeen return days are not enough to retrain or permanently reject the
  model, but they are sufficient to block live promotion.

Current decision:

- Keep M0 as the live baseline.
- Keep V3 frozen in shadow mode.
- Do not retune V3 from this forward sample.
- Reassess only after materially more forward observations have accumulated.

Forward artifacts:

- `run/build_reranker_v3_forward.py`
- `run/evaluate_reranker_v3_forward.py`
- `forward_results/m0_v3_20260615/`

### Confidence-gated V4

V4 adds a regression model, a candidate-versus-M0 classifier and an explicit
abstention gate. The gate was calibrated only from 2019-2022 rolling OOF
predictions and checked on 2023 before later-period comparisons.

Key behavior:

- 2023: active on 6.31% of dates; active-date target delta +0.01964.
- 2024 comparison: active on one date and modestly improves all tested return
  scenarios.
- 2025-2026 historical confirmation: active on 8 of 319 dates; base return and
  Sharpe improve slightly at both capital levels.
- True forward through 2026-06-11: active on zero dates, so V4 exactly matches
  M0 and avoids V3's additional forward loss.

Decision:

- Keep M0 live.
- Prefer V4 over V3 for continued shadow observation.
- Do not promote V4 yet because its activation rate collapses outside the gate
  calibration years.
- Research a date-level OOF meta-gate as V4.1 without using 2024 or later data
  for calibration.

Full specification and results:

- `RERANKER_V4_PLAN_20260615.md`
- `run/train_reranker_v4.py`
- `run/validate_reranker_v4_2024.py`
- `run/confirm_reranker_v4_history.py`
- `run/evaluate_reranker_v4_forward.py`
- `reranker_models_20260615/gated_v4/`
- `reranker_validation_20260615/gated_v4/`
- `reranker_confirmation_20260615/gated_v4/`

### V4.1 meta-gate conclusion

The date-level meta gate produced strong 2023 and historical comparison
results, but underperformed M0 by 1.45 percentage points at 500k and 2.75
percentage points at 1m in the frozen 17-day forward return window. Its market
inputs were inside the historical distribution, pointing to date-level sample
size and relationship instability rather than simple out-of-range features.

Decision:

- Reject V4.1 for promotion.
- Do not tune it from the forward sample.
- Keep V4 as the preferred shadow candidate because its conservative gate
  abstained and reproduced M0 exactly in the same period.
- Keep M0 as the live baseline.
