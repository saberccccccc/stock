# Workspace Cleanup - 2026-05-31

Conservative cleanup performed while preserving formal results, checkpoints, and
the running temporal training job.

## Deleted

### Temporary Python artifacts

- `backtest/__pycache__`
- `core/__pycache__`
- `data/__pycache__`
- `run/__pycache__`
- `scripts/__pycache__`

### Smoke-only artifacts

- `checkpoints_exp_smoke`
- `checkpoints_exp/temporal_smoke_best.pt`
- `checkpoints_exp/temporal_smoke_v2_best.pt`
- `reports/temporal_dataset_smoke.json`
- `reports/temporal_dataset_smoke_v2.json`
- `backtest_results_switch_value_smoke_20260530`
- `backtest_results_trade_policy_smoke_20260530`
- `backtest_results_trade_policy_v2_smoke_20260530`
- `models_trade_policy_smoke_20260530`
- `switch_value_data_smoke_20260530`
- `switch_value_models_smoke_20260530`
- `trade_policy_data_smoke_20260530`
- temporal smoke cache files matching `cache/temporal_cross_section_*max40*s20_lb20*`

### Large historical detailed backtest dumps

Deleted only `*_full_data.pkl` files under:

- `backtest_results_exp*`
- `backtest_results_layered`

Kept:

- `*_summary.csv`
- `*_summary.txt`
- `*_returns.csv`
- `*_diagnostics.csv`

This preserved comparison evidence while removing large regenerated detail
objects.

Deleted count:

- `162` full-data pickle files

Freed space from this step:

- about `21.78 GB`

### Broken junction

- Removed broken workspace junction `dataaw`, which pointed to missing target
  `F:\stock_prediction\deepseek_optimized\dataaw` and caused `git status`
  warnings.

## Preserved

- All formal checkpoint directories:
  - `checkpoints`
  - `checkpoints_exp`
  - `checkpoints_exp_topfocus*`
  - `checkpoints_exp_pairwise*`
- Formal switch value data/model outputs:
  - `switch_value_data_20260530_fixed`
  - `switch_value_models_20260530_fixed`
- Formal trade policy data/model outputs:
  - `trade_policy_data_20260530`
  - `models_trade_policy_20260530`
- Summary reports:
  - `backtest_result_snapshots`
  - `backtest_results_summary_*.txt`
- Running temporal training artifacts:
  - `logs/temporal_probe_20260531.*.log`
  - `cache/temporal_cross_section_*all_techn_marke*s40_lb60*`
  - `checkpoints_exp/temporal_cross_alpha_probe_20260531.pt` if/when saved

## Current Large Items

The largest remaining item is `cache`, currently dominated by the active full
temporal dataset build. It was intentionally left untouched.
