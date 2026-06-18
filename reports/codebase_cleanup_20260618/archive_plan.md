# Archive Plan 2026-06-18

This is a non-destructive plan. No files are moved by this report.

## Summary

| Action | Count |
|---|---:|
| archive_candidate | 74 |
| protect | 18 |
| review | 21 |

## Protected Paths

- `alpha`
- `archive`
- `backtest`
- `cache`
- `checkpoints_exp_topfocus_w005_topic`
- `checkpoints_loss_ablation_M0_nomulti`
- `checkpoints_loss_ablation_M1_nomulti_topfocus_w005`
- `configs`
- `core`
- `data`
- `experiments`
- `forward_results`
- `reports`
- `run`
- `scripts`
- `tests`
- `v9_avgw3_extend_to_20260518_20260616`
- `v9_avgw3_open_ledger_20260617`

## Archive Candidates

| Name | Kind | Class | Target |
|---|---|---|---|
| .pytest_cache | dir | archive_or_cache | archive/cache_202606 |
| __pycache__ | dir | archive_or_cache | archive/cache_202606 |
| _archive_models_data_20260604 | dir | archive_or_cache | archive/cache_202606 |
| _archive_results_20260604 | dir | archive_or_cache | archive/cache_202606 |
| breadth_triggered_market_20260617 | dir | experiment_output | archive/experiments_202606 |
| breadth_triggered_target_20260617 | dir | experiment_output | archive/experiments_202606 |
| candidate_model_validation_20260614 | dir | experiment_output | archive/experiments_202606 |
| checkpoints | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_batch4_benchmark_20260613 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_batch8_benchmark_20260613 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_exp | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_exp_pairwise_w003_20260530_011759 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_exp_purged_rawmetric_A_20260613 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_exp_top06_stable_20260612 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_exp_topfocus | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_exp_topfocus_w005_pairwise_w001_20260530_053300 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_exp_topfocus_w005_topret | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_A0 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_A0_low_lr_e10 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_A1 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_A2 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_A3 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_A4 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_A4_low_lr_e10 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_A5 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_D001 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_D003 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_D003_T001 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_D005 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_LAG005 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_LAG010 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_loss_ablation_T001 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_reranker_oof_F1_train2017_val2018 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_reranker_oof_F2_train2018_val2019 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_reranker_oof_F3_train2019_val2020 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_reranker_oof_F4_train2020_val2021 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_reranker_oof_F5_train2021_val2022 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_reranker_oof_F6_train2022_val2023 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| checkpoints_smoke_rawmetrics_20260613 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| conditional_negfilter_breadth_20260618 | dir | experiment_output | archive/experiments_202606 |
| diagnostics_negfilter_drop3_20260617 | dir | experiment_output | archive/experiments_202606 |
| downside_topfocus_validation_20260616 | dir | experiment_output | archive/experiments_202606 |
| FORWARD_TEST_LOG.md | file | experiment_output | archive/experiments_202606 |
| lag1_checkpoint_sweep_m0_20260616 | dir | experiment_output | archive/experiments_202606 |
| locked_candidate_confirmation_20260614 | dir | experiment_output | archive/experiments_202606 |
| loss_ablation_portfolio_validation_20260614 | dir | experiment_output | archive/experiments_202606 |
| m0_topfocus_validation_20260614 | dir | experiment_output | archive/experiments_202606 |
| models_multi_v9_tech_macro | dir | checkpoint_or_model | archive/checkpoints_202606 |
| multi_loss_validation_20260614 | dir | experiment_output | archive/experiments_202606 |
| open_reranker_current_v9_20260617 | dir | experiment_output | archive/experiments_202606 |
| open_reranker_current_v9_edge_20260617 | dir | experiment_output | archive/experiments_202606 |
| open_reranker_current_v9_forward_20260617 | dir | experiment_output | archive/experiments_202606 |
| open_reranker_current_v9_light_20260617 | dir | experiment_output | archive/experiments_202606 |
| open_reranker_current_v9_market_switch_20260617 | dir | experiment_output | archive/experiments_202606 |
| open_reranker_current_v9_negfilter_20260617 | dir | experiment_output | archive/experiments_202606 |
| reranker_confirmation_20260615 | dir | experiment_output | archive/experiments_202606 |
| reranker_data_20260614 | dir | experiment_output | archive/experiments_202606 |
| reranker_models_20260615 | dir | experiment_output | archive/experiments_202606 |
| reranker_oof_20260614 | dir | experiment_output | archive/experiments_202606 |
| reranker_training_20260615 | dir | experiment_output | archive/experiments_202606 |
| reranker_v2_data_20260615 | dir | experiment_output | archive/experiments_202606 |
| reranker_v3_data_20260615 | dir | experiment_output | archive/experiments_202606 |
| reranker_validation_20260615 | dir | experiment_output | archive/experiments_202606 |
| resume_downside_topfocus_remaining_20260616.ps1 | file | experiment_output | archive/experiments_202606 |
| run_forward_observation_candidates_20260617.ps1 | file | experiment_output | archive/experiments_202606 |
| run_lag1_loss_ablation_after_sweep_20260616.ps1 | file | experiment_output | archive/experiments_202606 |
| run_unified_good_ops_validation_20260616.ps1 | file | experiment_output | archive/experiments_202606 |
| state_triggered_target_20260617 | dir | experiment_output | archive/experiments_202606 |
| switch_value_models_20260604_top3_pv1m_raw_lgb_h5 | dir | checkpoint_or_model | archive/checkpoints_202606 |
| unified_good_ops_validation_20260616 | dir | experiment_output | archive/experiments_202606 |
| v9_avgw3_filter095_validation_20260616 | dir | experiment_output | archive/experiments_202606 |
| v9_avgw3_open_ledger_20260616 | dir | experiment_output | archive/experiments_202606 |
| validate_downside_topfocus_candidates_20260616.ps1 | file | experiment_output | archive/experiments_202606 |
| validate_m0_epoch_lag1_sweep_20260616.ps1 | file | experiment_output | archive/experiments_202606 |

## Manual Review

| Name | Kind | Class | Reason |
|---|---|---|---|
| .claude | dir | misc | manual review before any move |
| .gitignore | file | misc | manual review before any move |
| .vscode | dir | misc | manual review before any move |
| __init__.py | file | source_or_docs | manual review before any move |
| _sys_check.ps1 | file | misc | manual review before any move |
| backtest_result_snapshots | dir | misc | manual review before any move |
| CANDIDATE_MODEL_VALIDATION_PLAN_20260614.md | file | source_or_docs | manual review before any move |
| CLAUDE.md | file | source_or_docs | manual review before any move |
| EXPERIMENTS.md | file | source_or_docs | manual review before any move |
| FROZEN_FORWARD_STRATEGY.md | file | source_or_docs | manual review before any move |
| LOSS_ABLATION_PLAN.md | file | source_or_docs | manual review before any move |
| PURGED_ALPHA_OPTIMIZATION_PLAN.md | file | source_or_docs | manual review before any move |
| README.md | file | source_or_docs | manual review before any move |
| requirements.txt | file | source_or_docs | manual review before any move |
| RERANKER_IMPLEMENTATION_PLAN_20260614.md | file | source_or_docs | manual review before any move |
| RERANKER_V4_PLAN_20260615.md | file | source_or_docs | manual review before any move |
| rerun_v9_avgw3_open_to_open_20260616 | dir | misc | manual review before any move |
| RESEARCH_PROTOCOL.md | file | source_or_docs | manual review before any move |
| SHARPE_OPTIMIZATION_REPORT.md | file | source_or_docs | manual review before any move |
| switch_value_data_20260604_top3_pv1m_raw | dir | misc | manual review before any move |
| TEST_PLAN.md | file | source_or_docs | manual review before any move |
