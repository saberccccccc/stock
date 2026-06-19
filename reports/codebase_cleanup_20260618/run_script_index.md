# Run Script Index 2026-06-19

This index classifies the current `run/` scripts by role. It does not move or
rename files. The goal is to make the experiment workspace navigable without
guessing which script is a durable entrypoint and which one is tied to a
specific research branch.

## Durable Entrypoints

| Script | Role | Keep in `run/`? |
|---|---|---|
| `run/train.py` | Main model training entrypoint. | Yes |
| `run/backtest_retention_open_ledger.py` | Thin CLI wrapper for open-price share-ledger backtests. | Yes |
| `run/backtest_retention_execution_constraints.py` | Legacy constrained share-ledger backtest engine. | Yes |
| `run/render_training_commands.py` | Training preset dry-run renderer. | Yes |
| `run/generate_source_inventory.py` | Cleanup source inventory generator. | Yes |
| `run/generate_archive_plan.py` | Cleanup archive-plan generator. | Yes |
| `run/archive_from_plan.py` | Non-destructive archive-plan mover. | Yes |
| `run/generate_review_docs_index.py` | Review-doc index generator. | Yes |
| `run/generate_checkpoint_reference_audit.py` | Checkpoint/model reference audit generator. | Yes |

## Shared Support And Baseline Utilities

| Script | Purpose | Cleanup handling |
|---|---|---|
| `run/__init__.py` | Marks `run` as an importable package. | Keep. |
| `run/v9_cache_utils.py` | Shared V9 alpha and universe cache helpers. | Keep as reusable support code. |
| `run/v9_long_only_optimization.py` | Defines `V9RankPredictor` and the original long-only sweep. | Keep; many current and historical scripts import `V9RankPredictor`. |
| `run/baseline_eval.py` | Lightweight cross-sectional baseline sanity checks. | Keep as a diagnostic entrypoint. |
| `run/hyper_search.py` | Original backtest hyperparameter grid search. | Keep as legacy methodology evidence. |
| `run/backtest_layered_holdings.py` | Rolling sleeve holdings backtest. | Keep until layered-runner coverage is audited. |
| `run/track_backtest_holdings.py` | Tracks close-to-close PnL from saved holdings. | Keep as an operational diagnostic. |

## Saved Alpha And Execution Utilities

| Script | Purpose | Cleanup handling |
|---|---|---|
| `run/blend_alpha_jsonl.py` | Blend two saved alpha rankings by percentile score. | Keep as a low-memory alpha utility. |
| `run/combine_alpha_jsonl.py` | Combine multiple saved alpha files into an ensemble. | Keep as a low-memory alpha utility. |
| `run/transform_alpha_for_execution.py` | Apply signal-day executable transforms, including the 9.5% chase filter. | Keep as a frozen execution entrypoint. |
| `run/make_negative_filter_from_full.py` | Derive negative-filter alpha from a full rerank. | Keep while open-reranker evidence remains active. |
| `run/make_edge_rerank_from_full.py` | Derive an edge-only rerank alpha file. | Keep while open-reranker evidence remains active. |
| `run/make_conditional_negfilter_alpha.py` | Apply the negative filter only when a metadata trigger is active. | Keep with conditional-breadth evidence. |
| `run/analyze_retention_capacity.py` | Diagnose ADV participation and retention capacity. | Keep as a capacity diagnostic. |
| `run/sweep_execution_constraints.py` | Sweep strict execution assumptions over saved alpha. | Keep as historical robustness evidence. |
| `run/forward_frozen_strategy.py` | Generate frozen V9 average-w3 alpha on forward-only observations. | Keep for cutoff-safe forward reproducibility. |

## Historical V9 Long-Only Research

| Script | Purpose | Cleanup handling |
|---|---|---|
| `run/test_long_only_plan.py` | Focused long-only plan runner from the initial V9 search. | Keep with the V9 strategy snapshots. |
| `run/v9_long_only_topfrac_sweep.py` | Sweep V9 long-only top fractions. | Keep with the V9 strategy snapshots. |
| `run/v9_top5_refine.py` | Refine the best historical V9 top-5% result. | Keep with the V9 strategy snapshots. |
| `run/v9_portfolio_micro_sweep.py` | Micro-sweep V9 portfolio parameters. | Keep with the V9 strategy snapshots. |
| `run/v9_layer_diagnostics.py` | Diagnose return layers for a V9 checkpoint. | Keep as historical diagnostic evidence. |
| `run/v9_checkpoint_ensemble.py` | Backtest rank ensembles across V9 checkpoints. | Keep with checkpoint-selection evidence. |
| `run/v9_persistent_sweep.py` | Sweep persistent-signal settings. | Keep with the V9 retention lineage. |
| `run/v9_persistent_refine.py` | Refine the historical 3-day persistent signal. | Keep with the V9 retention lineage. |

## Training And Candidate Validation

| Script | Purpose | Cleanup handling |
|---|---|---|
| `run/run_loss_ablation.py` | Launch registered loss-ablation training families. | Keep while loss-ablation evidence remains active. |
| `run/summarize_loss_ablation.py` | Summarize epoch metrics and correlations for loss experiments. | Keep with loss-ablation reports. |
| `run/validate_candidate_models.py` | Generate and strictly validate selected V9 checkpoints on 2024. | Keep; referenced by training-validation evidence. |
| `run/confirm_locked_candidate.py` | Confirm locked A4 candidate on reserved historical period. | Keep until candidate validation is fully consolidated. |
| `run/analyze_alpha_execution_quality.py` | Compare alpha files for chase risk and rank stability. | Keep as diagnostic utility. |
| `run/screen_stall_execution.py` | Screen stall/chase execution behavior. | Keep as diagnostic utility. |
| `run/generate_v9_inference_alpha.py` | Generate V9 alpha rankings for label-free inference dates. | Keep; useful for forward/cutoff alpha regeneration. |

Related PowerShell helpers:

```text
validate_m0_epoch_lag1_sweep_20260616.ps1
validate_downside_topfocus_candidates_20260616.ps1
resume_downside_topfocus_remaining_20260616.ps1
run_lag1_loss_ablation_after_sweep_20260616.ps1
run_unified_good_ops_validation_20260616.ps1
```

## Reranker Research

| Script | Purpose | Cleanup handling |
|---|---|---|
| `run/build_reranker_dataset.py` | Build first-stage M0 candidate-level reranker data. | Keep while reranker artifacts remain active. |
| `run/build_reranker_v2_dataset.py` | Add continuous executable labels and market features. | Keep as V2 rejection evidence. |
| `run/build_reranker_v3_dataset.py` | Build state-aware marginal-fill candidate rows. | Keep for V3/V4 reproducibility. |
| `run/build_reranker_v3_forward.py` | Build frozen M0/V3 forward shadow inputs. | Keep for forward shadow evidence. |
| `run/train_reranker.py` | Train first LambdaRank reranker. | Keep until failed V1 artifacts are archived. |
| `run/train_reranker_v2.py` | Train V2 continuous executable-return reranker. | Keep until V2 failure evidence is archived. |
| `run/train_reranker_v3.py` | Train V3 state-aware marginal-fill reranker. | Keep for V3 shadow reproducibility. |
| `run/train_reranker_v4.py` | Train V4 confidence-gated reranker. | Keep for V4 shadow reproducibility. |
| `run/train_reranker_v41.py` | Train V4.1 date-level meta-gate. | Keep as failed-forward evidence. |
| `run/validate_reranker_2024.py` | Validate first reranker blends on 2024. | Keep with reranker validation evidence. |
| `run/validate_conservative_reranker_2024.py` | Validate conservative V1 boundary corrections. | Keep until V1 failure evidence is archived. |
| `run/validate_reranker_v2_2024.py` | Validate V2. | Keep until V2 failure evidence is archived. |
| `run/validate_reranker_v3_2024.py` | Validate frozen V3 on 2024. | Keep. |
| `run/validate_reranker_v4_2024.py` | Validate frozen V4 on 2024. | Keep. |
| `run/validate_reranker_v41.py` | Apply frozen V4.1 meta-gate to an alpha period. | Keep as failed-forward evidence. |
| `run/confirm_reranker_v3_history.py` | Confirm V3 on post-2024 historical period. | Keep. |
| `run/confirm_reranker_v4_history.py` | Confirm V4 on post-2024 historical period. | Keep. |
| `run/evaluate_reranker_v3_forward.py` | Evaluate V3 forward shadow. | Keep. |
| `run/evaluate_reranker_v4_forward.py` | Evaluate V4 forward shadow. | Keep. |
| `run/diagnose_reranker_replacements.py` | Diagnose promoted/demoted names. | Keep as diagnostic utility. |

## Open-Reranker And Overlay Research

| Script | Purpose | Cleanup handling |
|---|---|---|
| `run/train_open_reranker_current_v9.py` | Train/apply current V9 open-execution reranker. | Keep while open-reranker candidates are observed. |
| `run/apply_open_reranker_forward.py` | Apply saved open reranker to forward alpha rows. | Keep for forward observation. |
| `run/diagnose_negative_filter.py` | Diagnose what negative filter removes. | Keep as first-attack candidate diagnostic. |
| `run/compare_open_ledger_diagnostics.py` | Compare two open-ledger diagnostic directories. | Keep as candidate comparison utility. |
| `run/summarize_open_ledger_candidates.py` | Summarize selected open-ledger candidates. | Keep; produces official candidate summary. |
| `run/summarize_candidate_stability.py` | Summarize candidate stability by month/state. | Keep as diagnostic utility. |
| `run/make_breadth_triggered_market_alpha.py` | Add row-level market multiplier metadata for weak breadth. | Keep for observation candidate. |
| `run/make_breadth_triggered_target_alpha.py` | Build breadth-triggered target-shrink alpha. | Keep until rejected overlay is archived. |
| `run/make_state_triggered_target_alpha.py` | Build state-triggered target-shrink alpha. | Keep until rejected overlay is archived. |
| `run/switch_alpha_by_market_state.py` | Switch between alpha files by index state. | Keep as low-memory transform utility. |
| `run/sweep_open_ledger_params.py` | Sweep open-ledger execution parameters on val/test alpha files. | Keep. |
| `run/sweep_open_price_ledger_params.py` | Sweep open-price share-ledger alpha execution. | Keep. |

Related PowerShell helper:

```text
run_forward_observation_candidates_20260617.ps1
```

## Legacy Research Utilities

| Script | Purpose | Cleanup handling |
|---|---|---|
| `run/backtest.py` | Older backtest entrypoint. | Keep until replacement coverage is audited. |
| `run/backtest_v9_retention.py` | V9 retention backtest/generation utility. | Keep; many reports reference it. |
| `run/backtest_retention_open_execution.py` | Earlier open-execution backtest. | Keep while old reports reference it. |
| `run/backtest_temporal_retention.py` | Temporal retention backtest. | Keep while temporal snapshot evidence is retained. |
| `run/backtest_temporal_daily_top.py` | Temporal daily-top backtest. | Keep with temporal research snapshots. |
| `run/backtest_switch_value_policy.py` | Switch-value policy backtest. | Keep while switch-value snapshots are retained. |
| `run/backtest_trade_policy.py` | Trade-policy v1 backtest. | Legacy; candidate for later archive/index after references are audited. |
| `run/backtest_trade_policy_v2.py` | Trade-policy v2 backtest. | Legacy; candidate for later archive/index after references are audited. |
| `run/build_switch_value_dataset.py` | Build switch-value dataset. | Legacy; keep because snapshot reports reference methodology. |
| `run/train_switch_value_model.py` | Train switch-value LightGBM model. | Legacy; keep with switch-value report. |
| `run/build_trade_policy_dataset.py` | Build trade-policy dataset. | Legacy; keep until trade-policy artifacts are archived. |
| `run/train_trade_policy.py` | Train trade-policy model. | Legacy; keep until trade-policy artifacts are archived. |
| `run/build_temporal_dataset.py` | Build temporal dataset. | Keep while temporal branch remains indexed. |
| `run/build_temporal_switch_dataset.py` | Build switch-value samples from temporal alpha states. | Keep with temporal/switch-value methodology evidence. |
| `run/train_temporal.py` | Train temporal model. | Keep while temporal branch remains indexed. |
| `run/eval_temporal_full.py` | Evaluate temporal model. | Keep while temporal branch remains indexed. |

## Recommendation And Watchlist Utilities

| Script | Purpose | Cleanup handling |
|---|---|---|
| `run/recommend_daily.py` | Daily recommendation entrypoint. | Keep if recommendation workflow remains supported. |
| `run/recommend_persistent.py` | Persistent recommendation helper. | Keep if recommendation workflow remains supported. |
| `run/recommend_utils.py` | Shared recommendation utilities. | Keep. |
| `run/rank_watchlist.py` | Rank watchlist names. | Keep if watchlist workflow remains supported. |
| `run/daily_top10.py` | Daily top-10 output. | Keep if daily workflow remains supported. |

## Future Cleanup Rules

1. Do not move scripts that are referenced by active ledgers or reports.
2. Before moving legacy scripts to a separate location, run a text-reference
   search and update reports that point at them.
3. Prefer moving groups together with their artifact ledgers: for example, do
   not archive `train_reranker_v41.py` before the V4.1 failed-forward artifacts
   are archived.
4. Keep durable CLI entrypoints in `run/`; consolidate one-off scripts only
   after their outputs are indexed.
5. `tests/test_run_script_index.py` enforces that every Python file directly
   under `run/` appears in this index. A new script must be classified when it
   is added.

Current coverage: 93 of 93 `run/*.py` files indexed.
