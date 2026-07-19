# Reranker V4: Confidence-Gated Marginal Fill Plan

> 状态：历史实验计划与结论记录；不再决定当前执行顺序。

## Objective

V4 improves V3 by adding an explicit abstention decision. It does not replace
M0 fills unless the model predicts a sufficiently strong advantage over the
exact M0 baseline fills for that date.

The forward observations after 2026-05-18 are excluded from training, feature
selection, gate calibration and model selection.

## Architecture

V4 contains two LightGBM models trained on the V3 state-aware OOF dataset:

1. A Huber regression model predicts continuous executable return.
2. A binary classifier predicts whether a candidate beats the mean executable
   return of that date's M0 baseline fills.

Candidate score:

```text
V4 score = 0.60 * daily regression percentile
         + 0.40 * daily win-probability percentile
```

Daily gate:

```text
confidence =
    mean predicted win probability of V4 proposed fills
  - mean predicted win probability of M0 baseline fills
```

If confidence is below the frozen gate, the Alpha row is left exactly equal to
M0. If confidence passes, V4 changes only the final one to three vacancy fills.

## Leakage Controls

- All stock models feeding the dataset are pure OOF models.
- Gate calibration uses rolling OOF predictions from 2019-2022.
- 2023 is used only as the final temporal validation year.
- `baseline_target`, `beats_baseline`, `year`, and all future/label fields are
  explicitly excluded from features.
- An initial leaked run was detected from implausibly strong results and was
  discarded before any downstream validation.
- 2024 and later data never enter V4 fitting or gate calibration.

## Gate Calibration

The gate is selected from confidence quantiles observed only in 2019-2022.
Coverage is constrained to 15%-80% during calibration.

Selected gate:

```text
confidence >= 0.1453109242
```

Calibration result:

- Active share: 20.02%.
- Mean all-date marginal target improvement: +0.00308.
- Mean active-date improvement: +0.01537.

2023 temporal validation:

- Active share: 6.31%.
- Mean all-date improvement: +0.00124.
- Mean active-date improvement: +0.01964.

The lower 2023 coverage shows confidence-scale drift, but the abstention
behavior remains directionally correct.

## Comparative Results

### 2024 comparison

V4 activated on one of 242 dates, 2024-11-07.

| Capital | Model | Base ann. | Sharpe | Cost 2x ann. | Lag 1 ann. |
|---:|---|---:|---:|---:|---:|
| 500k | M0 | 45.73% | 1.327 | 40.19% | 26.10% |
| 500k | V4 | 47.09% | 1.356 | 41.25% | 26.95% |
| 1m | M0 | 46.27% | 1.316 | 40.83% | 25.56% |
| 1m | V4 | 47.39% | 1.340 | 41.85% | 26.33% |

Because earlier reranker work already inspected 2024, this is a comparison
result rather than a pristine blind test.

### Post-2024 historical confirmation

V4 activated on 8 of 319 signal dates.

| Capital | Model | Base ann. | Sharpe | Cost 2x ann. | Lag 1 ann. |
|---:|---|---:|---:|---:|---:|
| 500k | M0 | 41.85% | 1.897 | 33.40% | 46.95% |
| 500k | V4 | 42.15% | 1.922 | 33.76% | 46.83% |
| 1m | M0 | 43.65% | 1.923 | 35.03% | 50.11% |
| 1m | V4 | 45.45% | 2.000 | 36.85% | 50.21% |

### True forward shadow through 2026-06-11

V4 activated on zero of 23 warmed signal dates. It therefore reproduced M0
exactly during the 17 realized return days:

| Capital | M0 return | V4 return |
|---:|---:|---:|
| 500k | -9.18% | -9.18% |
| 1m | -9.24% | -9.24% |

## Decision

- M0 remains the live baseline.
- V3 remains a frozen research reference.
- V4 is the preferred shadow candidate because it can abstain and cannot harm
  M0 when inactive.
- V4 is not ready for promotion because its confidence gate activates too
  rarely outside the calibration years.
- Do not lower the gate using 2024, post-2024 confirmation, or forward results.

## Next Research Step

The next improvement should address confidence-scale drift using only
2018-2023 OOF data:

1. Generate rolling OOF predictions for every training date.
2. Train a date-level meta-gate to predict whether the proposed fill set beats
   M0, rather than thresholding raw probability differences.
3. Use market regime, vacancy count, prediction dispersion, model agreement
   and proposed-versus-baseline score margins as meta-gate inputs.
4. Require stable activation and positive marginal return in multiple OOF
   years, not just higher pooled average improvement.
5. Keep exact M0 fallback behavior whenever the gate abstains.

This next step is a V4.1 calibration improvement, not permission to tune from
the short forward sample.

## V4.1 Date-Level Meta Gate Result

V4.1 was implemented using only rolling OOF candidate predictions:

- Meta rows: 1,091 dates from 2019-2023.
- Meta features: proposal-versus-M0 prediction margins, score dispersion,
  regression/classifier agreement, vacancy state and trailing market features.
- Meta label: whether the proposed fill set actually beat the exact M0 fill
  set on continuous executable target.
- Threshold calibration: rolling meta OOF predictions for 2021-2022.
- Final temporal validation: 2023.

Selected meta threshold:

```text
predicted win probability >= 0.5267829833
```

2023 temporal validation:

- Active share: 75.23%.
- Mean all-date target improvement: +0.00624.
- Active-date target improvement: +0.00829.
- Active-date win rate: 58.68%.

Historical comparisons were strong:

- 2024 comparison:
  - 500k annualized return: 45.73% to 53.71%.
  - 1m annualized return: 46.27% to 55.84%.
- Post-2024 historical confirmation:
  - 500k annualized return: 41.85% to 45.84%.
  - 1m annualized return: 43.65% to 52.33%.
  - Sharpe, doubled-cost and delayed execution results also improved.

However, the frozen true forward application failed:

| Capital | M0 realized return | V4.1 realized return | Difference |
|---:|---:|---:|---:|
| 500k | -9.18% | -10.63% | -1.45 pp |
| 1m | -9.24% | -11.99% | -2.75 pp |

V4.1 activated on 16 of 23 warmed forward signal dates. Its market inputs
were not outside the historical range, so the failure cannot be dismissed as
simple feature extrapolation. The more likely explanation is instability and
overfitting in a date-level model trained on only about 1,100 observations.

V4.1 decision:

- Do not promote V4.1.
- Do not retune its probability threshold from the forward result.
- Preserve it as a documented failed experiment.
- Continue M0 as the live baseline.
- Continue V4, not V4.1, as the preferred safe shadow model because V4
  abstained throughout the same forward window and exactly matched M0.

V4.1 artifacts:

- `run/train_reranker_v41.py`
- `run/validate_reranker_v41.py`
- `reranker_models_20260615/meta_gate_v41/`
- `reranker_validation_20260615/meta_gate_v41/`
- `reranker_confirmation_20260615/meta_gate_v41/`
- `forward_results/m0_v41_20260615/`
