# Training Validation Ledger 2026-06-19

This ledger covers the remaining training-validation archive candidates and
their matching checkpoint groups. It exists to prevent cleanup from removing
evidence that still explains the current M0/loss/lag1 decisions.

## Source Reports

| Source | Role |
|---|---|
| `reports/loss_ablation_decision_20260614.md` | Main loss-ablation decision record. |
| `reports/lag1_training_selection_plan_20260616.md` | Lag1 selection/training plan. |
| `reports/downside_topfocus_ablation_plan_20260615.md` | Downside and low Top-focus experiment plan. |
| `reports/codebase_cleanup_20260618/experiment_output_ledger.md` | Broader experiment-output classification. |

## Current Training Decision

| Topic | Decision |
|---|---|
| Baseline | Use M0: weighted multi-horizon global IC target with separate `multi_loss_weight=0`. |
| Top-focus | A4 helped A0, but M0 + Top-focus 0.005 was tested and rejected. Do not enable by default. |
| Industry IC | Mixed; retest lower weights only if needed. Not default. |
| Diversity | Rejected. |
| Spread | Rejected; raw stability alone selected the wrong behavior. |
| Separate multi-horizon auxiliary IC loss | Remove from baseline. It trained horizon heads but hurt delay robustness. |
| Lag1 loss | Experimental only. Accept only if executable base, lag1, and cost2x are all competitive with M0. |
| Downside loss | Experimental only; must be judged by executable portfolio, not IC. |

## Implementation Boundary

The training runtime now supports reproducible ablation parameters for:

```text
multi/diversity/industry/spread/top-focus weights
downside loss
lag1 IC and lag1 Top-focus loss
purged train/validation label boundaries
per-epoch checkpoints and raw Top-bucket metrics
external checkpoint continuation
```

This is capability preservation, not candidate promotion. Downside and lag1
weights remain zero by default, and the M0/loss decisions above remain
authoritative.

Safety properties:

- lag1 labels require explicit train/validation boundaries;
- lag1 purging extends one extra trading day beyond the normal horizon;
- lag1 IC and lag1 Top-focus delays activate independently;
- raw-return and lag1 batch tensors are loaded only when required;
- fresh per-epoch runs truncate stale metrics JSONL before writing.

## Reproducibility Tools

The following source utilities are retained as one reviewed bundle:

```text
run/analyze_alpha_execution_quality.py
run/confirm_locked_candidate.py
run/generate_v9_inference_alpha.py
run/run_loss_ablation.py
run/screen_stall_execution.py
run/summarize_loss_ablation.py
run/validate_candidate_models.py
```

`run/run_loss_ablation.py` now consumes the validated shared training-preset
API instead of maintaining a second JSON-to-CLI implementation. Import-smoke
coverage protects all seven tools. Keeping these sources preserves experiment
reproducibility; it does not promote any checkpoint or candidate.

## Experiment Output Candidates

| Path | Status | Cleanup action |
|---|---|---|
| `candidate_model_validation_20260614` | Active decision evidence | Keep until summarized into a candidate-validation ledger. |
| `locked_candidate_confirmation_20260614` | Active decision evidence | Keep with candidate validation evidence. |
| `loss_ablation_portfolio_validation_20260614` | Active decision evidence | Keep; source for M0/A0/A4/A5 executable comparisons. |
| `multi_loss_validation_20260614` | Active decision evidence | Keep; source for multi-horizon isolation. |
| `m0_topfocus_validation_20260614` | Active decision evidence | Keep; source for M0+Top-focus rejection. |
| `lag1_checkpoint_sweep_m0_20260616` | Active lag1 evidence | Keep until lag1 sweep best rows are indexed. |
| `downside_topfocus_validation_20260616` | Failed/unpromoted evidence | Archive only after downside result summary is indexed. |
| `resume_downside_topfocus_remaining_20260616.ps1` | Repro script | Archive with downside validation after summary. |
| `validate_downside_topfocus_candidates_20260616.ps1` | Repro script | Archive with downside validation after summary. |
| `validate_m0_epoch_lag1_sweep_20260616.ps1` | Repro script | Keep with lag1 sweep evidence. |
| `run_lag1_loss_ablation_after_sweep_20260616.ps1` | Repro script | Keep with lag1 evidence until summarized. |
| `run_unified_good_ops_validation_20260616.ps1` | Strategy validation script | Keep with V9 strategy evidence, not this ledger. |
| `run_forward_observation_candidates_20260617.ps1` | Forward observation script | Keep with forward/open-reranker evidence, not this ledger. |

## Checkpoint Candidates

| Group | Paths | Cleanup action |
|---|---|---|
| M0 protected references | `checkpoints_loss_ablation_M0_nomulti`, `checkpoints_loss_ablation_M1_nomulti_topfocus_w005` | Already protected by archive-plan rules. Keep. |
| A0/A1/A2/A3/A4/A5 loss ablation | `checkpoints_loss_ablation_A0*`, `A1`, `A2`, `A3`, `A4*`, `A5` | Keep until report references are audited; then archive rejected variants in small exact batches. |
| Downside/lag1/low Top-focus | `checkpoints_loss_ablation_D001`, `D003`, `D003_T001`, `D005`, `LAG005`, `LAG010`, `T001` | Keep until downside/lag1 results are summarized. |
| Older Top-focus experiments | `checkpoints_exp_topfocus*` | Do not move until checkpoint-reference audit confirms no current scripts/reports depend on them. |

## Safe Next Actions

1. Generate a checkpoint-reference audit for every `checkpoint_or_model`
   archive candidate.
2. Index best rows from `lag1_checkpoint_sweep_m0_20260616`.
3. Index downside/topfocus validation conclusions.
4. After those summaries exist, archive clearly rejected checkpoint directories
   in exact-prefix batches.

## Unsafe Actions

- Do not archive broad `checkpoints_loss_ablation_*` as one batch.
- Do not archive M0/M1 protected checkpoint directories.
- Do not promote any loss by IC alone; executable Top30, delay, costs and
  drawdown remain the selection criteria.
