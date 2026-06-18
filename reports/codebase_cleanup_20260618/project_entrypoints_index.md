# Project Entrypoints Index 2026-06-19

This is the high-level navigation map for the current experiment workspace. It
does not move files. Use it to choose the right script/report family before
training, validating, or cleaning artifacts.

## Start Here

| Need | Primary entrypoint |
|---|---|
| Current cleanup status | `reports/codebase_cleanup_20260618/progress_log.md` |
| Remaining cleanup queue | `reports/codebase_cleanup_20260618/remaining_cleanup_queue.md` |
| Official baseline and open-ledger decisions | `reports/codebase_cleanup_20260618/official_baselines.md` |
| Training/loss decisions | `reports/codebase_cleanup_20260618/training_validation_ledger.md` |
| Reranker decisions | `reports/codebase_cleanup_20260618/reranker_artifact_ledger.md` |
| Open-reranker and market-overlay decisions | `reports/codebase_cleanup_20260618/open_reranker_market_overlay_ledger.md` |
| Failed/unpromoted candidate summary | `reports/codebase_cleanup_20260618/rejected_candidate_summary.md` |
| Checkpoint/model references | `reports/codebase_cleanup_20260618/checkpoint_reference_audit.md` |
| Run script roles | `reports/codebase_cleanup_20260618/run_script_index.md` |
| Document/report roles | `reports/codebase_cleanup_20260618/document_report_index.md` |

## Current Cleanup Boundary

Remaining archive candidates are not obvious trash:

```text
checkpoint_or_model=30
experiment_output=36
```

Do not broad-move either class. The remaining checkpoint/model outputs are
referenced or high risk. The remaining experiment outputs are active evidence,
observation candidates, or referenced failed experiments.

## Core Runtime Entrypoints

Detailed script classification lives in:

```text
reports/codebase_cleanup_20260618/run_script_index.md
```

| Task | Entrypoint |
|---|---|
| Main training | `run/train.py` |
| Training preset dry-run | `run/render_training_commands.py` |
| Open-price share-ledger backtest | `run/backtest_retention_open_ledger.py` |
| Legacy execution-constrained share-ledger backtest | `run/backtest_retention_execution_constraints.py` |
| Open-ledger parameter sweep | `run/sweep_open_ledger_params.py` |
| Open-price ledger parameter sweep | `run/sweep_open_price_ledger_params.py` |
| Candidate validation | `run/validate_candidate_models.py` |
| Cleanup source inventory | `run/generate_source_inventory.py` |
| Cleanup archive plan | `run/generate_archive_plan.py` |
| Cleanup move executor | `run/archive_from_plan.py` |
| Checkpoint reference audit | `run/generate_checkpoint_reference_audit.py` |

## Training And Loss Scripts

| Script | Purpose |
|---|---|
| `run/run_loss_ablation.py` | Launch loss-ablation training families. |
| `run/summarize_loss_ablation.py` | Summarize loss-ablation outputs. |
| `run/confirm_locked_candidate.py` | Confirm locked candidate results. |
| `validate_m0_epoch_lag1_sweep_20260616.ps1` | Validate M0 epoch/lag1 sweep. |
| `validate_downside_topfocus_candidates_20260616.ps1` | Validate downside/top-focus candidates. |
| `resume_downside_topfocus_remaining_20260616.ps1` | Resume downside/top-focus validation. |
| `run_lag1_loss_ablation_after_sweep_20260616.ps1` | Run lag1 ablation after sweep. |

Decision reports:

```text
reports/loss_ablation_decision_20260614.md
reports/lag1_training_selection_plan_20260616.md
reports/downside_topfocus_ablation_plan_20260615.md
reports/codebase_cleanup_20260618/training_validation_ledger.md
```

## Reranker Scripts

| Script | Purpose |
|---|---|
| `run/build_reranker_dataset.py` | Build initial reranker dataset. |
| `run/build_reranker_v2_dataset.py` | Build V2 continuous-target dataset. |
| `run/build_reranker_v3_dataset.py` | Build V3 state-aware marginal-fill dataset. |
| `run/build_reranker_v3_forward.py` | Build V3 forward/shadow inputs. |
| `run/train_reranker.py` | Train initial reranker. |
| `run/train_reranker_v2.py` | Train V2. |
| `run/train_reranker_v3.py` | Train V3. |
| `run/train_reranker_v4.py` | Train confidence-gated V4. |
| `run/train_reranker_v41.py` | Train V4.1 meta-gate. |
| `run/validate_reranker_2024.py` | Validate initial reranker. |
| `run/validate_reranker_v2_2024.py` | Validate V2. |
| `run/validate_reranker_v3_2024.py` | Validate V3. |
| `run/validate_reranker_v4_2024.py` | Validate V4. |
| `run/validate_reranker_v41.py` | Validate V4.1. |
| `run/confirm_reranker_v3_history.py` | Confirm V3 historical result. |
| `run/confirm_reranker_v4_history.py` | Confirm V4 historical result. |
| `run/evaluate_reranker_v3_forward.py` | Evaluate V3 forward shadow. |
| `run/evaluate_reranker_v4_forward.py` | Evaluate V4 forward shadow. |

Decision reports:

```text
reports/codebase_cleanup_20260618/reranker_research_index.md
reports/codebase_cleanup_20260618/reranker_artifact_ledger.md
RERANKER_IMPLEMENTATION_PLAN_20260614.md
RERANKER_V4_PLAN_20260615.md
```

## Open-Reranker And Overlay Scripts

| Script | Purpose |
|---|---|
| `run/train_open_reranker_current_v9.py` | Train current V9 open-reranker. |
| `run/apply_open_reranker_forward.py` | Apply open-reranker to forward data. |
| `run/diagnose_negative_filter.py` | Diagnose negfilter candidate behavior. |
| `run/compare_open_ledger_diagnostics.py` | Compare open-ledger diagnostics. |
| `run/summarize_open_ledger_candidates.py` | Summarize open-ledger candidates. |
| `run/summarize_candidate_stability.py` | Summarize candidate stability. |
| `run/make_breadth_triggered_market_alpha.py` | Build breadth market-mult overlay alpha. |
| `run/make_breadth_triggered_target_alpha.py` | Build breadth target-shrink alpha. |
| `run/make_state_triggered_target_alpha.py` | Build state-triggered target-shrink alpha. |
| `run/switch_alpha_by_market_state.py` | Switch alpha by market state. |
| `run_forward_observation_candidates_20260617.ps1` | Forward observation candidate runner. |

Decision reports:

```text
reports/open_ledger_candidate_summary_20260617/candidate_summary.md
reports/forward_observation_plan_20260617.md
reports/forward_observation_20260617/forward_observation_20260519_20260616.md
reports/codebase_cleanup_20260618/open_reranker_market_overlay_ledger.md
reports/codebase_cleanup_20260618/rejected_candidate_summary.md
```

## Reports And Snapshots

| Area | Entry |
|---|---|
| Strategy archaeology | `reports/strategy_archaeology_review_20260616.md` |
| Open-price share-ledger | `reports/open_price_share_ledger_optimization_log_20260616.md` |
| Maxret095 report | `reports/open_price_share_ledger_maxret095_report_20260616.md` |
| Target fraction validation | `reports/target_fraction_validation_20260617/target_fraction_validation_report.md` |
| Breadth market overlay | `reports/breadth_triggered_market_20260617/breadth_triggered_market_report.md` |
| Breadth target overlay | `reports/breadth_triggered_target_20260617/breadth_triggered_target_report.md` |
| State target overlay | `reports/state_triggered_target_20260617/state_triggered_target_report.md` |
| Historical snapshots | `backtest_result_snapshots/` |

## Cleanup Rule Of Thumb

1. If a path is protected in `archive_plan.md`, do not move it.
2. If a path is in `checkpoint_reference_audit.md` as `hold`, do not move it.
3. If a path appears in `rejected_candidate_summary.md` but still has live
   references, do not move it.
4. If a move is allowed, dry-run with an exact name/prefix, regenerate reports,
   run focused tests, then commit only code/report/index changes.
