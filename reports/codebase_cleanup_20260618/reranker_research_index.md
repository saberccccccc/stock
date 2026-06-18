# Reranker Research Index 2026-06-18

This index consolidates the M0/V2/V3/V4/V4.1 reranker decision history from
the root-level reranker plans. It is an index only: source documents and
artifacts stay in place until their details are fully migrated or explicitly
archived.

## Source Documents

| Document | Role | Current cleanup decision |
|---|---|---|
| `RERANKER_IMPLEMENTATION_PLAN_20260614.md` | M0, V1/V2/V3, forward shadow, V4 summary | Keep or consolidate |
| `RERANKER_V4_PLAN_20260615.md` | V4 confidence gate and V4.1 meta-gate | Keep or consolidate |

## Non-Negotiable Selection Rule

Reranker work exists because broad Alpha IC and broad ranking metrics were not
enough for the actual Top30 execution decision. The selection priority is:

| Priority | Evidence |
|---|---|
| 1 | Executable portfolio return and Sharpe on the constrained Top30 strategy |
| 2 | Delay and doubled-cost robustness |
| 3 | Drawdown, turnover, blocked buys, ADV/capacity behavior |
| 4 | Top-book stability and replacement quality |
| 5 | Alpha IC only as a health check |

Forward observations after 2026-05-18 are not used for training, feature
selection, gate calibration, threshold tuning, or model selection.

## Decision Ledger

| Version | Design | Main result | Decision |
|---|---|---|---|
| M0 | Baseline Top30/retention signal | Live baseline and benchmark | Keep live |
| Conservative V1 probe | Freeze M0 core, allow max 3 or 6 replacements | Max-3 modestly improved 2024, but effect was too small and not enough to promote | Reject for live promotion |
| V2 | Continuous executable-target boundary reranker | Mixed 2024 result; slightly worse at 500k, tiny improvement at 1m | Reject; retain M0 |
| V3 | State-aware marginal-fill reranker, max final 3 vacancy fills | Strong 2024 and better risk-adjusted historical confirmation, but weaker short forward raw return | Freeze as research/shadow, not live |
| V4 | V3 plus conservative confidence gate and M0 fallback | Slight historical improvements; true forward activated zero times and matched M0 | Preferred safe shadow candidate |
| V4.1 | Date-level meta-gate for activation | Strong historical comparison, failed frozen forward by -1.45 pp at 500k and -2.75 pp at 1m | Reject; do not retune from forward |

## V3 Evidence

V3 aligned the model with the actual retention state machine:

- Existing holdings remain while inside the M0 hold pool.
- M0 keeps high-priority fills.
- V3 changes at most the final three vacancy fills from candidates below rank
  80.

Strict 2024 validation:

| Model | Capital | Base ann. | Sharpe | Cost 2x ann. | Lag 1 ann. | MDD |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 500k | 45.73% | 1.327 | 40.19% | 26.10% | 17.41% |
| V3 | 500k | 52.67% | 1.538 | 46.04% | 42.18% | 15.35% |
| M0 | 1m | 46.27% | 1.316 | 40.83% | 25.56% | 17.53% |
| V3 | 1m | 54.13% | 1.542 | 46.27% | 43.31% | 15.58% |

Frozen post-2024 historical confirmation:

| Model | Capital | Base ann. | Sharpe | Cost 2x ann. | Cost 2x Sharpe | Lag 1 ann. | Lag 1 Sharpe | Base MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 500k | 41.85% | 1.897 | 33.40% | 1.582 | 46.95% | 2.049 | 11.77% |
| V3 | 500k | 40.02% | 2.053 | 32.35% | 1.723 | 42.02% | 2.107 | 11.12% |
| M0 | 1m | 43.65% | 1.923 | 35.03% | 1.605 | 50.11% | 2.122 | 11.73% |
| V3 | 1m | 43.56% | 2.139 | 35.36% | 1.804 | 45.91% | 2.202 | 10.75% |

Interpretation:

- V3 improved Sharpe and drawdown in the historical confirmation.
- V3 reduced turnover and blocked buys.
- Raw annualized return was mixed: lower at 500k and roughly unchanged at 1m.
- V3 is a valid risk-adjusted reranker, not a proven raw-return replacement
  for M0.

First true forward shadow through 2026-06-11:

| Model | Capital | Realized return | Sharpe | MDD | Avg turnover | Blocked buys |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 500k | -9.18% | -4.290 | 8.40% | 0.3827 | 9 |
| V3 | 500k | -9.34% | -6.244 | 9.02% | 0.3683 | 6 |
| M0 | 1m | -9.24% | -3.996 | 9.09% | 0.4032 | 10 |
| V3 | 1m | -10.32% | -6.395 | 9.99% | 0.3927 | 8 |

Current V3 decision:

- Keep M0 as the live baseline.
- Keep V3 frozen in shadow mode.
- Do not retune V3 from the short forward sample.
- Reassess only after materially more forward observations accumulate.

## V4 Evidence

V4 adds a conservative abstention layer. If the frozen gate does not pass, the
Alpha row stays exactly equal to M0.

Selected gate:

```text
confidence >= 0.1453109242
```

V4 comparison summary:

| Period | Activation | Result |
|---|---:|---|
| 2023 temporal validation | 6.31% | Active-date target delta +0.01964 |
| 2024 comparison | 1 of 242 dates | Modest improvement in tested return scenarios |
| 2025-2026 historical confirmation | 8 of 319 signal dates | Slight base return and Sharpe improvement |
| True forward through 2026-06-11 | 0 of 23 warmed signal dates | Exactly matched M0 |

2024 comparison:

| Capital | Model | Base ann. | Sharpe | Cost 2x ann. | Lag 1 ann. |
|---:|---|---:|---:|---:|---:|
| 500k | M0 | 45.73% | 1.327 | 40.19% | 26.10% |
| 500k | V4 | 47.09% | 1.356 | 41.25% | 26.95% |
| 1m | M0 | 46.27% | 1.316 | 40.83% | 25.56% |
| 1m | V4 | 47.39% | 1.340 | 41.85% | 26.33% |

Post-2024 historical confirmation:

| Capital | Model | Base ann. | Sharpe | Cost 2x ann. | Lag 1 ann. |
|---:|---|---:|---:|---:|---:|
| 500k | M0 | 41.85% | 1.897 | 33.40% | 46.95% |
| 500k | V4 | 42.15% | 1.922 | 33.76% | 46.83% |
| 1m | M0 | 43.65% | 1.923 | 35.03% | 50.11% |
| 1m | V4 | 45.45% | 2.000 | 36.85% | 50.21% |

Current V4 decision:

- M0 remains the live baseline.
- V4 is the preferred shadow candidate because it can abstain and cannot harm
  M0 when inactive.
- V4 is not ready for promotion because the gate activates too rarely outside
  calibration years.
- Do not lower the gate using 2024, post-2024 confirmation, or forward results.

## V4.1 Evidence

V4.1 used a date-level meta-gate trained only from rolling OOF candidate
predictions. Historical comparisons looked strong, but frozen forward failed.

Selected threshold:

```text
predicted win probability >= 0.5267829833
```

Historical comparisons:

| Period | Capital | M0 annualized | V4.1 annualized |
|---|---:|---:|---:|
| 2024 comparison | 500k | 45.73% | 53.71% |
| 2024 comparison | 1m | 46.27% | 55.84% |
| Post-2024 confirmation | 500k | 41.85% | 45.84% |
| Post-2024 confirmation | 1m | 43.65% | 52.33% |

Frozen true forward result:

| Capital | M0 realized return | V4.1 realized return | Difference |
|---:|---:|---:|---:|
| 500k | -9.18% | -10.63% | -1.45 pp |
| 1m | -9.24% | -11.99% | -2.75 pp |

Current V4.1 decision:

- Do not promote V4.1.
- Do not retune its probability threshold from the forward result.
- Preserve it as a documented failed experiment.
- Continue V4, not V4.1, as the preferred safe shadow model.

## Artifact Map

| Area | Artifacts |
|---|---|
| V3 data/model/validation | `reranker_v3_data_20260615/`, `reranker_models_20260615/regression_v3/`, `reranker_validation_20260615/regression_v3/` |
| V3 historical confirmation | `reranker_confirmation_20260615/regression_v3/`, `run/confirm_reranker_v3_history.py` |
| V3 forward shadow | `forward_results/m0_v3_20260615/`, `run/build_reranker_v3_forward.py`, `run/evaluate_reranker_v3_forward.py` |
| V4 model/validation | `reranker_models_20260615/gated_v4/`, `reranker_validation_20260615/gated_v4/`, `reranker_confirmation_20260615/gated_v4/` |
| V4 scripts | `run/train_reranker_v4.py`, `run/validate_reranker_v4_2024.py`, `run/confirm_reranker_v4_history.py`, `run/evaluate_reranker_v4_forward.py` |
| V4.1 model/validation | `reranker_models_20260615/meta_gate_v41/`, `reranker_validation_20260615/meta_gate_v41/`, `reranker_confirmation_20260615/meta_gate_v41/`, `forward_results/m0_v41_20260615/` |
| V4.1 scripts | `run/train_reranker_v41.py`, `run/validate_reranker_v41.py` |

## Cleanup Decisions

| Document/artifact group | Keep now? | Future action |
|---|---|---|
| `RERANKER_IMPLEMENTATION_PLAN_20260614.md` | Yes | Keep until all M0/V2/V3/V4 details are fully indexed |
| `RERANKER_V4_PLAN_20260615.md` | Yes | Keep until V4/V4.1 details are fully indexed |
| V3/V4/V4.1 result directories | Yes | Do not archive until artifact ledger records exact status and dependencies |
| Failed V4.1 artifacts | Yes, for now | Archive only after a failure ledger maps scripts, models, and reports |

## Next Cleanup Step

Create a reranker artifact ledger that classifies:

```text
reranker_data_20260614
reranker_oof_20260614
reranker_training_20260615
reranker_v2_data_20260615
reranker_v3_data_20260615
reranker_models_20260615
reranker_validation_20260615
reranker_confirmation_20260615
forward_results/m0_v3_20260615
forward_results/m0_v41_20260615
```

The ledger should mark each as active evidence, frozen shadow artifact, failed
experiment evidence, or archive candidate before any move.
