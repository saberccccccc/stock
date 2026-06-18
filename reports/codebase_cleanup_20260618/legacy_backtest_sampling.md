# Legacy Backtest Results Sampling 2026-06-18

This sampling pass reviews archive candidates matching `backtest_results_*`.
It does not move files. Its purpose is to decide which legacy result groups can
be archived in small batches after their conclusions are represented elsewhere.

## Candidate Count

`archive_plan.csv` currently lists 98 archive-candidate experiment outputs
matching `backtest_results_*`.

| Group | Pattern | Count |
|---|---|---:|
| exp | `backtest_results_exp_*` | 52 |
| test_plan | `backtest_results_test_plan_*` | 26 |
| switch_value | `backtest_results_switch_value_*` | 9 |
| temporal | `backtest_results_temporal_*` | 2 |
| summary_file | `backtest_results_summary_*.txt` | 2 |
| other | smoke/cache/topstable/retention outputs | 7 |

## Snapshot Coverage

`backtest_result_snapshots/` contains historical reports and summaries that
cover much of the older result set:

```text
20260528_long_only_baseline_backtest_results_summary_20260528.txt
20260528_long_only_baseline_decision_summary.md
20260528_strategy_optimization_summary.md
20260529_v9_long_only_optimization_report.md
20260529_v9_top_metric_selection_report.md
20260530_switch_value_fixed_report.md
20260530_trade_policy_v2_report.md
20260530_trade_strategy_tests_report.md
20260531_backtest_results_summary.md
20260531_model_strategy_comparison_report.md
20260531_model_strategy_comparison_summary.csv
20260531_temporal_longonly_metric_report.md
20260531_temporal_market_timing_summary.csv
20260531_temporal_target_sampler_overfit_report.md
20260531_temporal_v10_probe_report.md
20260604_exp_cleanup_report.md
```

This means many raw output directories are likely safe to archive after a
small dry-run batch, but they should not be deleted or moved blindly.

## Representative Samples

| Group | Representative sample | Observed files | Coverage status | Archive readiness |
|---|---|---|---|---|
| exp | `backtest_results_exp_base_avgw3_val` | `diagnostics_*.csv`, `monthly_summary.csv`, `returns_*.csv`, `v9_daily_alpha_top_order.jsonl`, `v9_retention_summary.csv`, `yearly_summary.csv` | Covered by reports and snapshots | Ready after small dry-run |
| test_plan | `backtest_results_test_plan_share_ledger_primary_test` | `diagnostics_*.csv`, `execution_cost_*.csv`, `monthly_summary.csv`, `returns_*.csv`, `top_by_sharpe.csv`, `yearly_summary.csv` | Covered by `TEST_PLAN.md` and `official_baselines.md` | Ready after small dry-run |
| switch_value | `backtest_results_switch_value_20260604_v9_alpha_baseline_test` | switch-value diagnostics/summary CSV files | Covered by snapshot reports | Ready after snapshot check |
| temporal | `backtest_results_temporal_full_eval_20260604` | temporal V10/V9 warmup result files | Covered by temporal reports | Ready after snapshot check |
| summary_file | `backtest_results_summary_20260528.txt` | standalone summary text | Covered by snapshot index | Ready after index |
| other | `backtest_results_v9_retention_20260531` | mixed smoke/cache/topstable/retention outputs | Partially covered | Needs manual batching |

## Archive Recommendation

Start with the easiest low-risk groups:

1. `backtest_results_exp_*`
2. `backtest_results_test_plan_*`
3. `backtest_results_summary_*.txt`

Hold these until a second pass:

1. `backtest_results_switch_value_*`
2. `backtest_results_temporal_*`
3. `backtest_results_v9_retention_20260531`
4. `backtest_results_topstable_epoch9_val_avgw3`
5. `backtest_results_switch_cache_smoke_*`
6. `backtest_results_small_account_smoke_20260612`

## Batch Safety Rules

1. Use `run/archive_from_plan.py` with `--class experiment_output` and a small
   `--limit`.
2. Review the dry-run list before execution.
3. Do not move protected official roots:

```text
v9_avgw3_open_ledger_20260617
v9_avgw3_extend_to_20260518_20260616
forward_results
```

4. Stop immediately if a dry-run includes non-legacy directories such as
   `candidate_model_validation_20260614`, `open_reranker_current_v9_*`, or
   `v9_avgw3_filter095_validation_20260616`.

## Next Step

Add a filtered archive option or batch list for legacy backtest outputs only,
then run the first small dry-run batch. Do not use the broad
`--class experiment_output` filter by itself because it also includes active
candidate and strategy evidence.
