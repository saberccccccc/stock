# Document And Report Index 2026-06-19

This index maps top-level plans and report files to canonical decision sources.
It does not move files. Use it to decide which document to read first and which
documents are supporting evidence or historical snapshots.

## Canonical Cleanup Navigation

| Topic | Canonical source |
|---|---|
| Overall cleanup status | `reports/codebase_cleanup_20260618/progress_log.md` |
| Project navigation | `reports/codebase_cleanup_20260618/project_entrypoints_index.md` |
| Script roles | `reports/codebase_cleanup_20260618/run_script_index.md` |
| Remaining queue | `reports/codebase_cleanup_20260618/remaining_cleanup_queue.md` |
| Source inventory | `reports/codebase_cleanup_20260618/source_inventory.md` |
| Archive plan | `reports/codebase_cleanup_20260618/archive_plan.md` |

## Top-Level Plans

| Document | Status | Canonical target | Notes |
|---|---|---|---|
| `CANDIDATE_MODEL_VALIDATION_PLAN_20260614.md` | Source evidence | `training_research_index.md` and `training_validation_ledger.md` | Keep until A4/frozen-V9 candidate validation is fully consolidated. |
| `LOSS_ABLATION_PLAN.md` | Source evidence | `training_research_index.md` and `rejected_candidate_summary.md` | Keep while R1/R2 and loss decisions remain active. |
| `PURGED_ALPHA_OPTIMIZATION_PLAN.md` | Source evidence | `training_research_index.md` | Source for 9.5% execution filter history. |
| `RERANKER_IMPLEMENTATION_PLAN_20260614.md` | Source evidence | `reranker_research_index.md` and `reranker_artifact_ledger.md` | Keep until reranker provenance is fully consolidated. |
| `RERANKER_V4_PLAN_20260615.md` | Source evidence | `reranker_research_index.md` and `rejected_candidate_summary.md` | Source for V4/V4.1 decisions. |

Already-tracked canonical docs such as `README.md`, `RESEARCH_PROTOCOL.md`,
`TEST_PLAN.md`, `FROZEN_FORWARD_STRATEGY.md`, and
`SHARPE_OPTIMIZATION_REPORT.md` remain governed by
`review_docs_index.md`.

## Reports: Training And Loss

| Report | Status | Notes |
|---|---|---|
| `reports/loss_ablation_decision_20260614.md` | Canonical training-loss decision | M0 baseline, rejected M0+Top-focus, rejected diversity/spread, multi-horizon decision. |
| `reports/loss_ablation_summary_20260614.csv` | Supporting data | Epoch/loss metrics for loss-ablation analysis. |
| `reports/loss_ablation_summary_20260614_correlations.csv` | Supporting data | Correlation diagnostics. |
| `reports/lag1_training_selection_plan_20260616.md` | Canonical lag1 plan | Defines L0-L3 lag1 evaluation sequence. |
| `reports/downside_topfocus_ablation_plan_20260615.md` | Source plan | Downside and low Top-focus experiment matrix. |
| `reports/result_csv_bestrow_audit_20260616.csv` | Supporting audit | Used by checkpoint/output decisions; do not archive alone. |
| `reports/result_csv_strategy_summary_audit_20260616.csv` | Supporting audit | Used by checkpoint/output decisions; do not archive alone. |

## Reports: Official Strategy And Open-Ledger

| Report | Status | Notes |
|---|---|---|
| `reports/open_price_share_ledger_maxret095_report_20260616.md` | Canonical maxret095 evidence | Open-price share-ledger plus maxret095 result. |
| `reports/open_price_share_ledger_optimization_log_20260616.md` | Canonical optimization log | Long-running open-ledger parameter and candidate log. |
| `reports/open_price_share_ledger_param_sweep_20260616.md` | Supporting sweep report | Parameter sweep details. |
| `reports/unified_good_ops_validation_plan_20260616.md` | Canonical validation protocol | Defines unified good-ops validation frameworks. |
| `reports/open_ledger_candidate_summary_20260617/` | Canonical candidate summary | Official baseline, first attack candidate, first stability candidate. |
| `reports/codebase_cleanup_20260618/official_baselines.md` | Cleanup-era baseline summary | Use as the quick baseline reference. |

## Reports: Forward And Market Overlay

| Report | Status | Notes |
|---|---|---|
| `reports/forward_observation_plan_20260617.md` | Canonical forward plan | Promotion gates and forward update. |
| `reports/forward_observation_20260617/` | Supporting forward results | Forward 2026-05-19 to 2026-06-16 summaries. |
| `reports/target_fraction_validation_20260617/` | Supporting risk-target evidence | Low-target and risk-target follow-up. |
| `reports/breadth_triggered_market_20260617/` | Observation candidate evidence | Breadth market-mult overlay remains observation-only. |
| `reports/breadth_triggered_target_20260617/` | Rejected overlay evidence | Target shrink rejected. |
| `reports/state_triggered_target_20260617/` | Rejected overlay evidence | State target shrink rejected. |
| `reports/codebase_cleanup_20260618/open_reranker_market_overlay_ledger.md` | Cleanup decision ledger | Use before any overlay artifact move. |
| `reports/codebase_cleanup_20260618/rejected_candidate_summary.md` | Compact rejection summary | Fast reference for failed/unpromoted candidates. |

## Reports: Data Quality And Forward Universe

| Report | Status | Notes |
|---|---|---|
| `reports/forward_data_quality_20260612.csv` | Supporting data-quality audit | Keep with forward validation evidence. |
| `reports/forward_universe_20260612.csv` | Supporting data-quality audit | Keep with forward validation evidence. |
| `reports/test_plan_raw_quality_20260612.csv` | Supporting data-quality audit | Keep with test-plan data-quality evidence. |
| `reports/test_plan_universe_quality_20260612.csv` | Supporting data-quality audit | Keep with test-plan data-quality evidence. |

## Reports: Temporal And Legacy Snapshots

| Report | Status | Notes |
|---|---|---|
| `reports/temporal_cross_alpha_probe_capped_20260531_val_full_eval.*` | Historical temporal probe evidence | Keep while temporal branch remains indexed. |
| `reports/temporal_cross_alpha_target_sampler_20260531_val_full_eval.*` | Historical temporal target-sampler evidence | Keep while temporal branch remains indexed. |
| `backtest_result_snapshots/` | Historical snapshot archive | Keep; it replaced many raw `backtest_results_*` directories as compact evidence. |
| `reports/strategy_archaeology_review_20260616.md` | Canonical archaeology review | Use before reviving old strategy ideas. |
| `reports/alpha_execution_filter_20260613.md` | Supporting execution-filter note | Keep with 9.5% filter evidence. |

## Cleanup Decisions

| Group | Move now? | Reason |
|---|---|---|
| Canonical decision reports | No | They are the current source of truth. |
| Supporting CSV/JSONL evidence | No | Many reports and ledgers reference them. |
| Historical snapshots | No | They are compact replacements for archived raw outputs. |
| Top-level source plans | No | Keep until their details are fully consolidated into canonical indexes. |

## Future Consolidation Order

1. Consolidate root-level training plans into `training_research_index.md` only
   after R1/R2 or successor loss plans are resolved.
2. Consolidate reranker plans only after V3/V4 shadow status changes or a new
   promoted reranker exists.
3. Consolidate open-ledger reports only after a new official baseline replaces
   `main_candidate`.
4. Keep historical snapshots until no current ledger/report points at them.
