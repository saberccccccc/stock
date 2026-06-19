# Reranker Artifact Ledger 2026-06-18

This ledger maps reranker-related artifact directories to cleanup decisions.
It does not move files. Its purpose is to prevent accidental archival of
evidence that is still needed to explain M0, V3, V4, and the rejected V4.1.

## Decision Summary

| Status | Meaning | Move now? |
|---|---|---|
| `active_evidence` | Required to explain or reproduce reranker provenance | No |
| `frozen_shadow_artifact` | Required for V3/V4 shadow comparison and future review | No |
| `forward_shadow_evidence` | True forward observation evidence | No |
| `failed_experiment_evidence` | Rejected experiment evidence, still needed for audit | Not yet |
| `failed_forward_evidence` | Failed forward evidence for a rejected model | Not yet |
| `mixed_*_artifacts` | Parent directory contains multiple statuses | Split or tag first |

## Top-Level Directories

| Path | Status | Decision | Reason |
|---|---|---|---|
| `reranker_data_20260614` | `active_evidence` | Keep for now | M0/reranker candidate data used by V3 historical confirmation and reranker index. |
| `reranker_oof_20260614` | `active_evidence` | Keep for now | OOF predictions are the leakage-control base for reranker training and validation. |
| `reranker_training_20260615` | `failed_or_superseded_evidence` | Keep for now | Early reranker training output; superseded by V2/V3/V4 but useful for audit trail. |
| `reranker_v2_data_20260615` | `failed_experiment_evidence` | Keep for now | V2 continuous executable-target reranker was rejected, but evidence explains why M0 was retained. |
| `reranker_v3_data_20260615` | `frozen_shadow_artifact` | Keep for now | Frozen V3 state-aware marginal-fill data; V3 remains a shadow reference. |
| `reranker_models_20260615` | `mixed_model_artifacts` | Keep for now | Contains V1, V2, V3, V4 and V4.1 model artifacts with different statuses. |
| `reranker_validation_20260615` | `mixed_validation_artifacts` | Keep for now | Contains validation CSVs and diagnostics across promoted, shadow and rejected paths. |
| `reranker_confirmation_20260615` | `mixed_confirmation_artifacts` | Keep for now | Contains post-2024 confirmation outputs for V3/V4/V4.1. |

## Model Artifact Status

| Path | Status | Decision | Reason |
|---|---|---|---|
| `reranker_models_20260615/lambdarank_v1` | `failed_experiment_evidence` | Keep for now | V1/LambdaRank path was not promoted. |
| `reranker_models_20260615/regression_v2` | `failed_experiment_evidence` | Keep for now | V2 was rejected after mixed 2024 execution result. |
| `reranker_models_20260615/regression_v3` | `frozen_shadow_artifact` | Keep | V3 is frozen shadow evidence and should remain reproducible. |
| `reranker_models_20260615/gated_v4` | `frozen_shadow_artifact` | Keep | V4 is the preferred safe shadow candidate because it can abstain to exact M0. |
| `reranker_models_20260615/meta_gate_v41` | `failed_experiment_evidence` | Keep for now | V4.1 failed frozen forward and must not be promoted or retuned. |

## Source Reproducibility

The V1-V4.1 historical reranker source family is maintained as a single
reproducibility bundle:

```text
build_reranker_dataset.py through build_reranker_v3_forward.py
train_reranker.py through train_reranker_v41.py
validate/confirm/evaluate reranker V1-V4.1 scripts
diagnose_reranker_replacements.py
```

The modules compile and can be imported through the `run.*` package. Import
compatibility is enforced by `tests/test_reranker_imports.py`; label helpers
remain covered by `tests/test_reranker_dataset.py`.

Tracking this source does not promote any candidate. V1, V2 and V4.1 remain
failed evidence; V3 remains a frozen research reference; V4 remains the safe
shadow candidate; M0 remains the live baseline.

## Validation And Confirmation Status

| Path | Status | Decision | Reason |
|---|---|---|---|
| `reranker_validation_20260615/conservative_v1` | `failed_experiment_evidence` | Keep for now | Conservative V1/max replacement probe was not promoted. |
| `reranker_validation_20260615/regression_v2` | `failed_experiment_evidence` | Keep for now | V2 validation evidence for rejection. |
| `reranker_validation_20260615/regression_v3` | `frozen_shadow_artifact` | Keep | V3 strict 2024 validation evidence. |
| `reranker_validation_20260615/gated_v4` | `frozen_shadow_artifact` | Keep | V4 validation evidence and gate behavior. |
| `reranker_validation_20260615/meta_gate_v41` | `failed_experiment_evidence` | Keep for now | V4.1 validation evidence needed to explain forward failure. |
| `reranker_confirmation_20260615/regression_v3` | `frozen_shadow_artifact` | Keep | V3 post-2024 confirmation: better Sharpe/drawdown but mixed raw return. |
| `reranker_confirmation_20260615/gated_v4` | `frozen_shadow_artifact` | Keep | V4 post-2024 confirmation: slight improvement and safe abstention behavior. |
| `reranker_confirmation_20260615/meta_gate_v41` | `failed_experiment_evidence` | Keep for now | V4.1 historical confirmation looked strong but contradicted forward evidence. |

## Forward Evidence

| Path | Status | Decision | Reason |
|---|---|---|---|
| `forward_results/m0_v3_20260615` | `forward_shadow_evidence` | Keep for now | Frozen true forward result through 2026-06-11 blocks V3 live promotion. |
| `forward_results/m0_v41_20260615` | `failed_forward_evidence` | Keep for now | Frozen true forward result shows V4.1 underperformed M0 by -1.45 pp and -2.75 pp. |

## Cleanup Rules

1. Do not move parent directories with mixed statuses.
2. Do not move V3/V4 model, validation, confirmation, or forward evidence
   while they remain active shadow references.
3. Failed V1/V2/V4.1 artifacts can be archived later, but only after the
   failure summaries are indexed and no scripts expect those paths directly.
4. Any move should be class-filtered and dry-run first, using the archive plan
   machinery rather than manual drag-and-drop.

## Next Step

Create a broader experiment-output ledger for non-reranker directories:

```text
candidate_model_validation_20260614
loss_ablation_portfolio_validation_20260614
multi_loss_validation_20260614
downside_topfocus_validation_20260616
lag1_checkpoint_sweep_m0_20260616
open_reranker_current_v9_*
```

Only after that ledger exists should experiment-output directories be moved to
`archive/experiments_202606`.
