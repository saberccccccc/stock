# Cleanup Progress Log

## 2026-06-18 Phase 1 / Phase 2 Low-Risk Start

### Completed

Created alpha utility package:

```text
alpha/__init__.py
alpha/io.py
alpha/transforms.py
```

Converted these scripts to compatibility wrappers:

```text
run/transform_alpha_for_execution.py
run/make_edge_rerank_from_full.py
run/make_negative_filter_from_full.py
run/make_conditional_negfilter_alpha.py
```

Added tests:

```text
tests/test_alpha_io.py
tests/test_alpha_transforms.py
```

Installed `pytest` into the `torch` conda environment:

```text
C:\Users\x\miniconda3\envs\torch
```

Created backtest preset/stress modules:

```text
backtest/presets.py
backtest/stress.py
tests/test_backtest_presets.py
```

### Important Behavior Notes

- No official backtest parameter was changed.
- `run/backtest_retention_open_ledger.py` behavior was not changed.
- Old alpha transform CLI scripts remain callable directly with `python run/<script>.py`.
- JSONL output remains UTF-8 without BOM.
- JSONL input now tolerates UTF-8 BOM, which is more permissive and should not change normal outputs.
- `OFFICIAL_OPEN_PRICE_SHARE_LEDGER` preset records `max_new_names=5`, but this has not been wired into the legacy CLI default. This avoids silently changing old script behavior.

### Validation

Compiled:

```text
alpha/__init__.py
alpha/io.py
alpha/transforms.py
run/transform_alpha_for_execution.py
run/make_edge_rerank_from_full.py
run/make_negative_filter_from_full.py
run/make_conditional_negfilter_alpha.py
backtest/presets.py
backtest/stress.py
```

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_backtest_presets.py `
  tests\test_alpha_io.py `
  tests\test_alpha_transforms.py `
  tests\test_transform_alpha_for_execution.py -q
```

Result:

```text
21 passed
```

CLI smoke checked:

```text
run/make_edge_rerank_from_full.py
run/make_negative_filter_from_full.py
run/make_conditional_negfilter_alpha.py
```

### Next Step

Recommended next step:

```text
Phase 2 continued:
add an opt-in preset wrapper or helper for open-ledger commands,
then compare a tiny date-window summary before touching the full engine.
```

Do not yet:

- move large result directories;
- delete legacy scripts;
- change `run/backtest_retention_open_ledger.py` defaults;
- run heavy full backtests;
- retrain models.

## 2026-06-18 Phase 2 Opt-In Open-Ledger Presets

### Completed

Added explicit preset/stress CLI support to:

```text
run/backtest_retention_open_ledger.py
```

New options:

```text
--preset {official_open_price_share_ledger,legacy_close_based_top30,research_open_to_open_wide_book}
--stress {normal,lag1,cost2x,capacity_3pct}
```

Behavior:

- Without `--preset` or `--stress`, legacy defaults are unchanged.
- With `--preset official_open_price_share_ledger`, official parameters are applied.
- With `--stress lag1/cost2x/capacity_3pct`, official base parameters are used if no preset is supplied.
- Explicit CLI flags override preset values.

Example:

```text
--preset official_open_price_share_ledger --stress cost2x --max-new-names 2
```

uses official + cost2x, but keeps `max_new_names=2`.

### Validation

Compiled:

```text
backtest/presets.py
backtest/stress.py
run/backtest_retention_open_ledger.py
tests/test_backtest_presets.py
tests/test_open_ledger_preset_cli.py
```

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_backtest_presets.py `
  tests\test_open_ledger_preset_cli.py -q
```

Result:

```text
13 passed
```

CLI help smoke confirmed `--preset` and `--stress` are visible.

### Important Behavior Notes

- No full backtest was run.
- No result directory was moved.
- `run/backtest_retention_open_ledger.py` default `max_new_names=0` remains unchanged unless a preset/stress is explicitly used.

### Next Step

Recommended next step:

```text
Run a tiny date-window equivalence check between manual official parameters and
--preset official_open_price_share_ledger, then begin extracting open-ledger
execution helpers only after equivalence is proven.
```

## 2026-06-18 Open-Ledger Preset Equivalence Check

### Completed

Fixed alpha JSONL loading in:

```text
run/backtest_retention_open_ledger.py
```

to accept UTF-8 files with BOM. This is input compatibility only; normal UTF-8
outputs are unchanged.

Added test coverage in:

```text
tests/test_open_ledger_preset_cli.py
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_backtest_presets.py `
  tests\test_open_ledger_preset_cli.py -q
```

Result:

```text
14 passed
```

Tiny real CLI equivalence check:

```text
manual official params
vs
--preset official_open_price_share_ledger
```

Input:

```text
5 stock codes
2 signal rows
max_data_date=2025-01-10
portfolio_values=500000,1000000
```

Result:

```text
tiny open-ledger preset equivalence passed
```

The two generated `open_ledger_summary.csv` files were identical.

### Next Step

Now it is safe to start extracting the smallest open-ledger helper functions,
starting with pure argument/preset/report helpers before moving execution logic.

## 2026-06-18 Phase 3 Minimal Open-Ledger Helper Extraction

### Completed

Created:

```text
backtest/open_ledger.py
```

Extracted pure helpers from `run/backtest_retention_open_ledger.py`:

```text
parse_float_list
load_alpha_rows
```

Kept the legacy entrypoint compatible by importing those helpers back into:

```text
run/backtest_retention_open_ledger.py
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_backtest_presets.py `
  tests\test_open_ledger_preset_cli.py -q
```

Result:

```text
15 passed
```

Tiny CLI equivalence:

```text
manual official params
vs
--preset official_open_price_share_ledger
```

Result:

```text
tiny open-ledger helper extraction equivalence passed
```

### Next Step

Next safe extraction targets:

```text
limit_new_names
apply_open_ledger_constraints
summarize_result
```

Extract one at a time and rerun the tiny equivalence check after each step.

## 2026-06-18 Phase 3 limit_new_names Extraction

### Completed

Moved the portfolio name-retention helper into:

```text
backtest/open_ledger.py
```

Extracted function:

```text
limit_new_names
```

The legacy entrypoint now imports it from the shared module:

```text
run/backtest_retention_open_ledger.py
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_backtest_presets.py `
  tests\test_open_ledger_preset_cli.py -q
```

Result:

```text
19 passed
```

Tiny CLI equivalence:

```text
manual official params
vs
--preset official_open_price_share_ledger
```

Result:

```text
tiny open-ledger limit_new_names extraction equivalence passed
```

### Note

The tests document the existing behavior, including the current `exit_hold_frac`
and `switch_gap_frac` semantics. No strategy rule was changed.

### Next Step

Next extraction target:

```text
apply_open_ledger_constraints
```

This is higher risk because it touches cash/share execution and costs, so it
should be moved with focused unit tests plus the same tiny equivalence check.

## 2026-06-18 Phase 3 Execution Constraint Extraction

### Completed

Moved execution helpers into:

```text
backtest/open_ledger.py
```

Extracted:

```text
open_limit_trade_mask
apply_open_ledger_constraints
```

The legacy entrypoint now imports execution logic from the shared module:

```text
run/backtest_retention_open_ledger.py
```

### Validation

Added:

```text
tests/test_open_ledger_execution.py
```

Covered:

- open limit buy/sell masks;
- board-lot buying;
- cash and cost accounting smoke;
- limit-up buy blocking;
- limit-down sell blocking;
- rebalance-band skip behavior.

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_backtest_presets.py `
  tests\test_open_ledger_preset_cli.py `
  tests\test_open_ledger_execution.py -q
```

Result:

```text
24 passed
```

Tiny CLI equivalence:

```text
manual official params
vs
--preset official_open_price_share_ledger
```

Result:

```text
tiny open-ledger execution extraction equivalence passed
```

### Next Step

Next extraction target:

```text
summarize_result
```

After that, consider moving the remaining `run_open_ledger` orchestration only
if the tiny equivalence check keeps passing.

## 2026-06-18 Phase 3 Summary Extraction

### Completed

Moved summary-row construction into:

```text
backtest/open_ledger.py
```

Extracted:

```text
summarize_open_ledger_result
```

The legacy entrypoint now calls this helper from:

```text
run/backtest_retention_open_ledger.py
```

### Validation

Expanded:

```text
tests/test_open_ledger_execution.py
```

Covered:

- return-day count;
- turnover averages;
- executed/unfilled turnover averages;
- holding days;
- average name count;
- gross weight;
- market multiplier;
- costs and blocked/capped counters;
- effective target fraction.

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_backtest_presets.py `
  tests\test_open_ledger_preset_cli.py `
  tests\test_open_ledger_execution.py -q
```

Result:

```text
25 passed
```

Tiny CLI equivalence:

```text
manual official params
vs
--preset official_open_price_share_ledger
```

Result:

```text
tiny open-ledger summary extraction equivalence passed
```

### Next Step

At this point, the reusable `backtest/open_ledger.py` contains IO, name limiting,
execution constraints, and summary construction. The next major extraction is
the `run_open_ledger` orchestration loop itself. That should be done carefully
because it touches date alignment, market timing, row-level target/mult metadata,
and output diagnostics.

## 2026-06-18 Phase 3 Target Helper Extraction

### Completed

Moved target/weight helpers into:

```text
backtest/open_ledger.py
```

Extracted:

```text
build_desired_target
weights_from_selected
```

The open-ledger entrypoint now imports these from the shared module instead of
`run/backtest_retention_execution_constraints.py`.

### Validation

Expanded:

```text
tests/test_open_ledger_preset_cli.py
```

Covered:

- retaining current names inside the hold bucket;
- filling target from alpha order;
- max-weight and gross-weight behavior;
- empty selection behavior.

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_backtest_presets.py `
  tests\test_open_ledger_preset_cli.py `
  tests\test_open_ledger_execution.py -q
```

Result:

```text
29 passed
```

Tiny CLI equivalence:

```text
manual official params
vs
--preset official_open_price_share_ledger
```

Result:

```text
tiny open-ledger target helper extraction equivalence passed
```

### Next Step

Before moving the full `run_open_ledger` loop, extract or relocate market timing
helpers so `backtest/open_ledger.py` does not need to depend on `run/` modules.

## 2026-06-18 Phase 3 Market Helper Extraction

### Completed

Moved market-state helpers into:

```text
backtest/open_ledger.py
```

Extracted:

```text
load_index_returns
compute_market_multiplier
```

The open-ledger entrypoint no longer imports these from
`run/backtest_temporal_retention.py`.

### Validation

Expanded:

```text
tests/test_open_ledger_execution.py
```

Covered:

- missing index file fallback;
- index close/daily-return loading;
- `none` and short-history behavior;
- legacy bear/crash multiplier behavior;
- dynamic multiplier bounds;
- unknown mode error.

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_backtest_presets.py `
  tests\test_open_ledger_preset_cli.py `
  tests\test_open_ledger_execution.py -q
```

Result:

```text
35 passed
```

Tiny CLI equivalence:

```text
manual official params
vs
--preset official_open_price_share_ledger
```

Result:

```text
tiny open-ledger market helper extraction equivalence passed
```

### Next Step

`run_open_ledger` can now move into `backtest/open_ledger.py` with fewer `run/`
dependencies. After moving it, keep the legacy CLI as a wrapper and rerun the
same tiny equivalence check.

## 2026-06-18 Phase 3 Run Loop Extraction

### Completed

Moved the main portfolio loop into:

```text
backtest/open_ledger.py
```

Extracted:

```text
run_open_ledger
```

The legacy CLI entrypoint now keeps only:

- argument parsing;
- research/forward date guard;
- OHLC/money loading;
- ADV recomputation;
- output file writing;
- progress printing.

The strategy loop itself now lives in the reusable backtest module.

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_backtest_presets.py `
  tests\test_open_ledger_preset_cli.py `
  tests\test_open_ledger_execution.py -q
```

Result:

```text
35 passed
```

Tiny CLI equivalence:

```text
manual official params
vs
--preset official_open_price_share_ledger
```

Result:

```text
tiny open-ledger run loop extraction equivalence passed
```

### Next Step

The remaining open-ledger wrapper dependencies are:

```text
load_ohlc_money
recompute_adv
save_stage_breakdown
```

Recommended next cleanup is to move these remaining reusable IO/report helpers
out of `run/` so `run/backtest_retention_open_ledger.py` becomes a true thin CLI
wrapper.

## 2026-06-18 Phase 3 Thin Open-Ledger Wrapper

### Completed

Moved the remaining reusable IO/report helpers into:

```text
backtest/open_ledger.py
```

Extracted:

```text
load_ohlc_money
recompute_adv
save_stage_breakdown
```

`run/backtest_retention_open_ledger.py` no longer imports helper functions from
other `run/` backtest scripts. It now acts as a thin CLI wrapper around
`backtest/open_ledger.py`.

### Validation

Expanded:

```text
tests/test_open_ledger_execution.py
```

Covered:

- OHLC/money CSV loading;
- money scaling;
- shifted rolling ADV recomputation;
- yearly/monthly stage breakdown output.

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_backtest_presets.py `
  tests\test_open_ledger_preset_cli.py `
  tests\test_open_ledger_execution.py -q
```

Result:

```text
38 passed
```

Tiny CLI equivalence:

```text
manual official params
vs
--preset official_open_price_share_ledger
```

Result:

```text
tiny open-ledger IO helper extraction equivalence passed
```

### Next Step

Phase 3 is now functionally complete for open-price share-ledger extraction.
Next recommended phase:

```text
experiments/registry.py
experiments/leaderboard.py
```

This will prevent future candidate comparisons from being rebuilt manually and
will keep execution-family rankings separated.

## 2026-06-18 Phase 4 Experiment Registry / Leaderboard

### Completed

Created a reproducible experiment comparison layer:

```text
experiments/__init__.py
experiments/registry.py
experiments/leaderboard.py
tests/test_experiment_leaderboard.py
```

The registry now records current open-price share-ledger candidates:

```text
official
breadth_m085
edge_r030_100
negfilter_drop3
risk_target_r004
```

Each result source carries explicit metadata:

```text
split
stress
source_type
target_frac
hold_frac
portfolio_value
```

This prevents grid/sweep summaries from leaking unrelated target or hold
settings into the leaderboard. The official validation source is pinned to the
historical grid row:

```text
target_frac=0.006
hold_frac=0.10
```

Generated:

```text
reports/candidate_leaderboard_20260617/candidate_leaderboard_from_registry.csv
```

### Validation

Compiled:

```text
experiments/__init__.py
experiments/registry.py
experiments/leaderboard.py
tests/test_experiment_leaderboard.py
```

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_experiment_leaderboard.py -q
```

Result:

```text
7 passed
```

Registry leaderboard generation:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m experiments.leaderboard `
  --output-csv reports\candidate_leaderboard_20260617\candidate_leaderboard_from_registry.csv `
  --fail-on-missing
```

Result:

```text
rows=78
missing=0
```

Compared against the existing manual leaderboard:

```text
old_rows=78
new_rows=78
missing_in_new=0
extra_in_new=0
```

Maximum numeric differences were only floating-point noise:

```text
ann: 5.68e-14
sharpe: 4.88e-15
mdd: 5.00e-16
exec_to: 5.00e-16
blocked_buy: 0
```

### Next Step

Next recommended phase:

```text
Phase 5: training/config organization
```

Start by extracting stable training experiment definitions for M0/V9/loss
ablation runs, but do not change training behavior or checkpoint-selection
rules until each config wrapper has a dry-run or smoke test.

## 2026-06-18 Phase 5 Training Presets Start

### Completed

Created:

```text
core/training_presets.py
tests/test_training_presets.py
```

Added a lightweight parser for existing training experiment JSON files under:

```text
configs/
```

The parser expands:

```text
common + experiments[] -> TrainingExperiment
```

and can render each experiment into explicit `run/train.py` CLI arguments.

This does not change `run/train.py` behavior. It only makes existing training
definitions inspectable and testable before wiring them into any CLI wrapper.

### Validation

Compiled:

```text
core/training_presets.py
tests/test_training_presets.py
```

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_training_presets.py `
  tests\test_experiment_leaderboard.py -q
```

Result:

```text
10 passed
```

### Next Step

Continue Phase 5 by adding a dry-run command renderer for training suites, so
loss-ablation and OOF fold commands can be printed and reviewed without starting
training.

## 2026-06-18 Phase 5 Training Command Dry-Run

### Completed

Extended:

```text
core/training_presets.py
```

Added:

```text
run/render_training_commands.py
```

The new CLI renders `run/train.py` commands from config suites without starting
training. It supports:

```text
--config
--config-dir
--experiment-id
--python-exe
--train-script
```

Example:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\render_training_commands.py `
  --config configs\m0_topfocus_validation_20260614.json `
  --experiment-id M1 `
  --python-exe C:\Users\x\miniconda3\envs\torch\python.exe
```

Output starts with:

```text
# m0_topfocus_validation_20260614:M1
```

and prints the fully expanded `run/train.py` command.

### Validation

Compiled:

```text
core/training_presets.py
run/render_training_commands.py
tests/test_training_presets.py
```

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_training_presets.py -q
```

Result:

```text
5 passed
```

Manual dry-run smoke:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\render_training_commands.py `
  --config configs\m0_topfocus_validation_20260614.json `
  --experiment-id M1 `
  --python-exe C:\Users\x\miniconda3\envs\torch\python.exe
```

Result:

```text
printed one M1 command; no training launched
```

### Next Step

The next Phase 5 step is to add validation around config keys, so typos in JSON
presets fail before a long training run starts.

## 2026-06-18 Phase 5 Training Config Key Validation

### Completed

Extended:

```text
core/training_presets.py
tests/test_training_presets.py
```

Added an explicit whitelist for `run/train.py` parameters and validation during
training suite loading. Unknown keys now fail early with the config path and
experiment id in the error message.

This catches mistakes such as:

```text
epochz
top_fokus_loss_weight
```

before any training run starts.

### Validation

Compiled:

```text
core/training_presets.py
tests/test_training_presets.py
run/render_training_commands.py
```

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_training_presets.py -q
```

Result:

```text
8 passed
```

Dry-run all configs with experiment filter:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\render_training_commands.py `
  --config-dir configs `
  --experiment-id A0 `
  --python-exe python
```

Result:

```text
rendered loss_ablation_20260613:A0
```

No unknown config keys were found in the current `configs/*.json` files.

### Next Step

The next Phase 5 step is to add a small checkpoint-selection report helper that
can rank saved epoch metrics by explicit gates instead of relying on IC alone.

## 2026-06-18 Phase 5 Checkpoint Selection Helper

### Completed

Created:

```text
experiments/checkpoint_selection.py
tests/test_checkpoint_selection.py
```

The helper reads:

```text
epochs/epoch_metrics.jsonl
```

and flattens each epoch into a table with:

```text
epoch
train_loss
train_components
val_metrics
checkpoint
```

Default rule:

```text
gates:
  alpha >= 0.07
  rawtopret_h5_top0p6 >= 0

rank:
  rawtopstable_h5_top0p6 weight 1.0
  rawtopret_h5_top0p6 weight 0.5
  alpha weight 0.1
```

This keeps Alpha IC as a minimum gate while making top-book stability and
top-book return the primary ranking inputs. It is intentionally separate from
the training loop and does not change checkpoint saving behavior.

Generated example report:

```text
reports/checkpoint_selection_20260618/a0_top_first_selection.csv
```

For `checkpoints_loss_ablation_A0`, the top-first rule ranked epoch 6 first.

### Validation

Compiled:

```text
experiments/checkpoint_selection.py
tests/test_checkpoint_selection.py
```

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_checkpoint_selection.py `
  tests\test_training_presets.py -q
```

Result:

```text
11 passed
```

Manual report smoke:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m experiments.checkpoint_selection `
  checkpoints_loss_ablation_A0\epochs\epoch_metrics.jsonl `
  --output-csv reports\checkpoint_selection_20260618\a0_top_first_selection.csv `
  --top 6
```

Result:

```text
epoch 6 ranked first under the top-first rule
```

### Next Step

The next checkpoint-selection step should join this epoch-level report with
open-ledger validation summaries when those per-epoch backtests exist. Until
then, this helper is a safer IC-gated training-metrics screen, not a full
portfolio-selection replacement.

## 2026-06-18 Phase 6 Preparation: Regenerable Source Inventory

### Completed

Created:

```text
experiments/source_inventory.py
run/generate_source_inventory.py
tests/test_source_inventory.py
```

Regenerated:

```text
reports/codebase_cleanup_20260618/source_inventory.csv
reports/codebase_cleanup_20260618/source_inventory.md
```

The inventory generator scans only top-level project entries and classifies
them into cleanup groups:

```text
source_or_docs
checkpoint_or_model
experiment_output
runtime_log_or_pid
archive_or_cache
misc
```

Current regenerated inventory:

```text
rows=263
```

Key classification improvements:

- PID/stdout/stderr/log files are now consistently `runtime_log_or_pid`.
- `checkpoints_loss_ablation_*` and OOF checkpoint folders are now
  `checkpoint_or_model`.
- source directories added during cleanup, including `alpha/`, `experiments/`,
  and `configs/`, are included as `source_or_docs`.

No files or output directories were moved.

### Validation

Compiled:

```text
experiments/source_inventory.py
run/generate_source_inventory.py
tests/test_source_inventory.py
```

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_source_inventory.py `
  tests\test_checkpoint_selection.py -q
```

Result:

```text
6 passed
```

Inventory generation smoke:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\generate_source_inventory.py
```

Result:

```text
wrote reports\codebase_cleanup_20260618\source_inventory.csv rows=263
wrote reports\codebase_cleanup_20260618\source_inventory.md
```

### Next Step

Before moving any root-level outputs, add an archive plan that maps each class
to a target archive directory and explicitly lists protected active paths that
must not move.

## 2026-06-18 Phase 6 Archive Plan

### Completed

Created:

```text
experiments/archive_plan.py
run/generate_archive_plan.py
tests/test_archive_plan.py
```

Generated non-destructive archive plan:

```text
reports/codebase_cleanup_20260618/archive_plan.csv
reports/codebase_cleanup_20260618/archive_plan.md
```

The plan maps inventory classes to target archive folders:

```text
runtime_log_or_pid -> archive/logs_202606
archive_or_cache -> archive/cache_202606
experiment_output -> archive/experiments_202606
checkpoint_or_model -> archive/checkpoints_202606
```

Protected active paths include:

```text
forward_results
v9_avgw3_open_ledger_20260617
v9_avgw3_extend_to_20260518_20260616
checkpoints_exp_topfocus_w005_topic
checkpoints_loss_ablation_M0_nomulti
checkpoints_loss_ablation_M1_nomulti_topfocus_w005
alpha/backtest/configs/core/data/experiments/reports/run/scripts/tests
```

Archive plan summary:

```text
archive_candidate=224
protect=17
review=22
```

No files or directories were moved.

### Validation

Compiled:

```text
experiments/archive_plan.py
run/generate_archive_plan.py
tests/test_archive_plan.py
```

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
7 passed
```

Archive plan generation smoke:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\generate_archive_plan.py
```

Result:

```text
wrote reports\codebase_cleanup_20260618\archive_plan.csv rows=263
wrote reports\codebase_cleanup_20260618\archive_plan.md
```

Spot checks:

```text
forward_results -> protect
v9_avgw3_open_ledger_20260617 -> protect
checkpoints_exp_topfocus_w005_topic -> protect
errors.log -> archive/logs_202606
backtest_results_exp_base_avgw3_val -> archive/experiments_202606
```

### Next Step

Do not move archive candidates yet. The next safe step is to add a dry-run mover
that prints planned moves and refuses to move protected/review paths.

## 2026-06-18 Phase 6 Archive Dry-Run Mover

### Completed

Extended:

```text
experiments/archive_plan.py
tests/test_archive_plan.py
```

Added:

```text
run/archive_from_plan.py
```

The new CLI reads:

```text
reports/codebase_cleanup_20260618/archive_plan.csv
```

and prints planned archive moves. By default it is dry-run only. It only builds
moves for rows with:

```text
action=archive_candidate
```

Protected and manual-review rows are skipped. Execution is opt-in via
`--execute`, and the core mover still refuses protected paths.

### Validation

Compiled:

```text
experiments/archive_plan.py
run/archive_from_plan.py
tests/test_archive_plan.py
```

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py -q
```

Result:

```text
7 passed
```

Dry-run smoke:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py --limit 5
```

Result:

```text
DRY-RUN archive moves: 5
.pytest_cache -> archive\cache_202606\.pytest_cache
__pycache__ -> archive\cache_202606\__pycache__
_archive_models_data_20260604 -> archive\cache_202606\_archive_models_data_20260604
_archive_results_20260604 -> archive\cache_202606\_archive_results_20260604
a5_recovery.pid -> archive\logs_202606\a5_recovery.pid
```

No files were moved.

### Next Step

If root cleanup is desired, first run the dry-run mover without `--limit` and
review the full printed move list. Only then consider a very small `--execute`
batch, starting with harmless log/pid files.

## 2026-06-18 Phase 6 Archive Mover Filters

### Completed

Extended:

```text
experiments/archive_plan.py
run/archive_from_plan.py
tests/test_archive_plan.py
```

Added dry-run filters:

```text
--class
--target
```

This allows reviewing only one inventory class or target archive folder before
any move. Example:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 10
```

### Validation

Compiled:

```text
experiments/archive_plan.py
run/archive_from_plan.py
tests/test_archive_plan.py
```

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py -q
```

Result:

```text
8 passed
```

Filtered dry-run smoke:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 10
```

Result:

```text
DRY-RUN archive moves: 10
a5_recovery.pid -> archive\logs_202606\a5_recovery.pid
a5_recovery_stderr.log -> archive\logs_202606\a5_recovery_stderr.log
a5_recovery_stdout.log -> archive\logs_202606\a5_recovery_stdout.log
...
candidate_validation.pid -> archive\logs_202606\candidate_validation.pid
```

No files were moved.

### Next Step

The first real cleanup batch, if desired, should be limited to:

```text
--class runtime_log_or_pid
```

and preferably a small `--limit` value after reviewing the full dry-run output.

## 2026-06-18 Phase 6 First Log/PID Archive Batch

### Completed

Updated `.gitignore` so archived logs and pid files do not pollute git status:

```text
*.pid
archive/logs_202606/
archive/cache_202606/
```

Executed the first very small archive batch:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 10 `
  --execute
```

Moved 10 low-risk runtime files into:

```text
archive/logs_202606/
```

Moved files:

```text
a5_recovery.pid
a5_recovery_stderr.log
a5_recovery_stdout.log
batch4_timing_stderr.log
batch4_timing_stdout.log
batch8_benchmark_stderr.log
batch8_benchmark_stdout.log
batch8_timing_stderr.log
batch8_timing_stdout.log
candidate_validation.pid
```

Then regenerated:

```text
reports/codebase_cleanup_20260618/source_inventory.csv
reports/codebase_cleanup_20260618/source_inventory.md
reports/codebase_cleanup_20260618/archive_plan.csv
reports/codebase_cleanup_20260618/archive_plan.md
```

Current archive plan summary after the move:

```text
archive_candidate=214
protect=18
review=22
```

`archive/` itself is now protected so future archive plans cannot recursively
archive the archive directory.

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
11 passed
```

Spot checks:

```text
archive -> protect
candidate_validation_stderr.log -> archive_candidate
archive/logs_202606 contains the 10 moved files
```

### Next Step

If continuing root cleanup, run another dry-run with:

```text
--class runtime_log_or_pid --limit 10
```

and only execute after reviewing the printed list.

## 2026-06-18 Phase 6 Second Log/PID Archive Batch

### Completed

Dry-run reviewed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 10
```

Then executed the second small runtime-only batch:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 10 `
  --execute
```

Moved 10 additional low-risk runtime files into:

```text
archive/logs_202606/
```

Moved files:

```text
candidate_validation_stderr.log
candidate_validation_stdout.log
downside_topfocus_ablation_20260615.err.log
downside_topfocus_ablation_20260615.out.log
errors.log
formal_train.pid
formal_train_stderr.log
formal_train_stdout.log
loss_ablation_A1_active.pid
loss_ablation_A2_active.pid
```

Regenerated:

```text
reports/codebase_cleanup_20260618/source_inventory.csv
reports/codebase_cleanup_20260618/source_inventory.md
reports/codebase_cleanup_20260618/archive_plan.csv
reports/codebase_cleanup_20260618/archive_plan.md
```

Current archive plan summary after the move:

```text
archive_candidate=204
protect=18
review=22
runtime_log_or_pid remaining=32
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
11 passed
```

Spot checks:

```text
archive -> protect
archive/logs_202606 contains 20 moved files
loss_ablation_queue.pid remains an archive_candidate for a later batch
```

### Next Step

Continue only with small `runtime_log_or_pid` batches after dry-run review. Do
not move checkpoints or experiment output directories until the source/report
references are audited.

## 2026-06-18 Phase 6 Third Log/PID Archive Batch

### Completed

Dry-run reviewed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 10
```

Then executed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 10 `
  --execute
```

Moved 10 additional runtime files into:

```text
archive/logs_202606/
```

Moved files:

```text
loss_ablation_queue.pid
loss_ablation_queue_batch4_stderr.log
loss_ablation_queue_batch4_stdout.log
loss_ablation_queue_stderr.log
loss_ablation_queue_stdout.log
loss_ablation_resume_stderr.log
loss_ablation_resume_stdout.log
loss_ablation_singlefactor_stderr.log
loss_ablation_singlefactor_stdout.log
low_lr_continuation_queue.pid
```

Regenerated source inventory and archive plan.

Current archive plan summary after the move:

```text
archive_candidate=194
protect=18
review=22
runtime_log_or_pid remaining=22
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
11 passed
```

Spot checks:

```text
archive -> protect
low_lr_continuation_stderr.log remains an archive_candidate for a later batch
archive/logs_202606 contains 30 moved files
```

## Phase 6 Fourth Log/PID Archive Batch

### Completed

Dry-run reviewed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 10
```

Then executed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 10 `
  --execute
```

Moved 10 additional runtime files into:

```text
archive/logs_202606/
```

Moved files:

```text
low_lr_continuation_stderr.log
low_lr_continuation_stdout.log
purged_rawmetric_A.pid
purged_rawmetric_A_stderr.log
purged_rawmetric_A_stdout.log
ram_smoke_stderr.log
ram_smoke_stdout.log
resume_downside_topfocus_remaining_20260616.err.log
resume_downside_topfocus_remaining_20260616.out.log
run_lag1_loss_ablation_20260616.err.log
```

Regenerated source inventory and archive plan.

Current archive plan summary after the move:

```text
archive_candidate=184
protect=18
review=22
runtime_log_or_pid remaining=12
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
11 passed
```

Spot checks:

```text
archive/logs_202606 contains 40 moved files
```

### Next Step

Finish the remaining 12 `runtime_log_or_pid` candidates in one or two small
batches. Stop before moving checkpoint or experiment-output directories.

## Phase 6 Final Log/PID Archive Batch

### Completed

Dry-run reviewed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 20
```

Then executed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 20 `
  --execute
```

Moved the final 12 top-level runtime files into:

```text
archive/logs_202606/
```

Moved files:

```text
run_lag1_loss_ablation_20260616.out.log
run_unified_good_ops_validation_20260616.err.log
run_unified_good_ops_validation_20260616.out.log
stall_execution_queue.pid
stall_execution_queue_stderr.log
stall_execution_queue_stdout.log
train_gat.log
train_v9.log
validate_downside_topfocus_candidates_20260616.err.log
validate_downside_topfocus_candidates_20260616.out.log
validate_m0_epoch_lag1_sweep_20260616.err.log
validate_m0_epoch_lag1_sweep_20260616.out.log
```

Regenerated source inventory and archive plan.

Current archive plan summary after the move:

```text
archive_candidate=172
protect=18
review=22
runtime_log_or_pid remaining=0
top-level inventory rows=212
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
11 passed
```

Spot checks:

```text
archive/logs_202606 contains 52 moved files
```

### Next Step

Review the remaining 22 `review` entries and decide whether docs/scripts should
be kept, indexed, or archived. Avoid moving checkpoints, experiment outputs, or
backtest artifacts until each class has a narrower plan.

## Phase 6 Legacy Logs Directory Archive

### Completed

Improved source inventory classification so the top-level `logs/` directory is
classified as `runtime_log_or_pid` instead of generic `misc`.

Dry-run reviewed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 5
```

Then executed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class runtime_log_or_pid `
  --limit 5 `
  --execute
```

Moved:

```text
logs -> archive/logs_202606/logs
```

Regenerated source inventory and archive plan.

Current archive plan summary after the move:

```text
archive_candidate=172
protect=18
review=21
runtime_log_or_pid remaining=0
top-level inventory rows=211
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
11 passed
```

Spot checks:

```text
logs exists=False
archive/logs_202606/logs exists=True
archive/logs_202606 contains 53 top-level entries
```

### Next Step

Create a separate review plan for the remaining 21 manual-review entries:
keep active docs in place, archive local IDE/settings/cache separately, and
handle historical backtest/report snapshots only after mapping which reports
are still referenced by the current strategy documents.

## Phase 7 Manual-Review Document Index

### Completed

Added a reproducible review-document index generator:

```text
experiments/review_docs.py
run/generate_review_docs_index.py
```

Generated:

```text
reports/codebase_cleanup_20260618/review_docs_index.csv
reports/codebase_cleanup_20260618/review_docs_index.md
```

The index covers the 12 root-level markdown files still marked as manual
review in the archive plan. It does not move those files. It records whether
each file should be kept, consolidated into a cleanup index, or archived only
after consolidation.

Current review-document summary:

```text
keep=4
keep_or_consolidate=7
archive_after_consolidation=1
```

Important decisions captured:

```text
README.md -> keep
CLAUDE.md -> keep
RESEARCH_PROTOCOL.md -> keep
FROZEN_FORWARD_STRATEGY.md -> keep
TEST_PLAN.md and SHARPE_OPTIMIZATION_REPORT.md -> consolidate toward official_baselines
LOSS_ABLATION/PURGED_ALPHA/CANDIDATE_MODEL plans -> consolidate toward training research index
RERANKER plans -> consolidate toward reranker research index
EXPERIMENTS.md -> archive only after key points are consolidated
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_review_docs.py `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
14 passed
```

### Next Step

Create the first consolidation index, starting with either:

```text
reports/codebase_cleanup_20260618/training_research_index.md
reports/codebase_cleanup_20260618/reranker_research_index.md
```

Do not move the source markdown files until the consolidation index has been
created and checked.

## Phase 7 Training Research Consolidation Index

### Completed

Created:

```text
reports/codebase_cleanup_20260618/training_research_index.md
```

This index consolidates the root-level training and loss-ablation documents:

```text
LOSS_ABLATION_PLAN.md
PURGED_ALPHA_OPTIMIZATION_PLAN.md
CANDIDATE_MODEL_VALIDATION_PLAN_20260614.md
EXPERIMENTS.md
```

Key conclusions captured:

```text
9.5% signal-day return filter remains the strongest validated execution transform.
A4-E6 was the strongest new purged candidate but did not pass the standalone 2024 promotion gate.
Frozen V9 remains operationally useful, but its 2025-2026 confirmation is not clean model evidence.
Alpha IC is a floor, not the final checkpoint-selection criterion.
Top-focus, R1 raw Top30 return loss and R2 chase penalty remain research candidates, not proven production improvements.
```

No source markdown files were moved.

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_review_docs.py `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
14 passed
```

Content spot-check:

```text
rg "9.5%|A4-E6|old frozen V9|Top-focus|R1|R2|Promotion gate" `
  reports\codebase_cleanup_20260618\training_research_index.md
```

### Next Step

Create `reports/codebase_cleanup_20260618/reranker_research_index.md`, then
the root reranker plans can be consolidated in the same way without losing the
M0/V3/V4/V4.1 decision history.

## Phase 7 Reranker Research Consolidation Index

### Completed

Created:

```text
reports/codebase_cleanup_20260618/reranker_research_index.md
```

This index consolidates the root-level reranker documents:

```text
RERANKER_IMPLEMENTATION_PLAN_20260614.md
RERANKER_V4_PLAN_20260615.md
```

Key decisions captured:

```text
M0 remains the live baseline.
V3 is a frozen risk-adjusted shadow reference, not a proven raw-return replacement.
V4 is the preferred safe shadow candidate because it abstains to exact M0 when inactive.
V4 is not promoted yet because activation collapses outside calibration years.
V4.1 is rejected and must not be retuned from the forward result.
```

No source markdown files or reranker artifacts were moved.

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_review_docs.py `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
14 passed
```

Content spot-check:

```text
rg "V3|V4|V4\.1|M0 remains|preferred shadow|Do not promote|45\.73%|52\.67%|-10\.63%|reranker_models" `
  reports\codebase_cleanup_20260618\reranker_research_index.md
```

### Next Step

Create a reranker artifact ledger that maps each `reranker_*`,
`forward_results/m0_v3_20260615`, and `forward_results/m0_v41_20260615`
artifact to one of:

```text
active evidence
frozen shadow artifact
failed experiment evidence
archive candidate
```

Do not move reranker artifact directories until this ledger exists.

## Phase 8 Reranker Artifact Ledger

### Completed

Created:

```text
reports/codebase_cleanup_20260618/reranker_artifact_ledger.csv
reports/codebase_cleanup_20260618/reranker_artifact_ledger.md
```

The ledger maps 23 reranker-related artifact paths, including top-level
`reranker_*` directories, nested V1/V2/V3/V4/V4.1 model and validation
subdirectories, and forward evidence directories:

```text
forward_results/m0_v3_20260615
forward_results/m0_v41_20260615
```

Status summary:

```text
active_evidence=2
frozen_shadow_artifact=7
forward_shadow_evidence=1
failed_experiment_evidence=8
failed_forward_evidence=1
failed_or_superseded_evidence=1
mixed_confirmation_artifacts=1
mixed_model_artifacts=1
mixed_validation_artifacts=1
```

Important cleanup rules captured:

```text
Do not move parent directories with mixed statuses.
Do not move V3/V4 evidence while they remain active shadow references.
Failed V1/V2/V4.1 artifacts can be archived later only after failure summaries are indexed.
Any move should be class-filtered and dry-run first.
```

No reranker artifact directories were moved.

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_review_docs.py `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
14 passed
```

Coverage spot-check:

```text
ledger rows=23
required paths covered:
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

### Next Step

Create a broader experiment-output ledger for non-reranker result directories
before moving any experiment outputs into `archive/experiments_202606`.

## Phase 8 Experiment Output Ledger

### Completed

Created:

```text
reports/codebase_cleanup_20260618/experiment_output_ledger.csv
reports/codebase_cleanup_20260618/experiment_output_ledger.md
```

The ledger classifies non-reranker experiment outputs before any batch move to
`archive/experiments_202606`.

Archive-plan context for non-reranker `experiment_output` entries:

```text
archive_candidate=126
protect=3
```

Protected roots captured:

```text
forward_results
v9_avgw3_open_ledger_20260617
v9_avgw3_extend_to_20260518_20260616
```

Ledger coverage:

```text
rows=37
candidate_validation=2
loss_validation=8
unified_validation=2
official_candidate=2
open_reranker=6
market_overlay=5
reports=3
legacy_backtests=5
protected=3
misc_experiment_scripts=1
```

Important decisions captured:

```text
Do not move protected roots.
Keep official V9 filter/open-ledger evidence.
Keep open-reranker and negfilter attack-candidate evidence until forward observation is resolved.
Keep candidate/loss validation directories until checkpoint/loss result ledgers exist.
Treat legacy backtest_results_* as archive candidates only after representative sampling.
```

No experiment-output directories were moved.

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_review_docs.py `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
14 passed
```

Coverage spot-check:

```text
required paths covered:
  forward_results
  v9_avgw3_open_ledger_20260617
  v9_avgw3_extend_to_20260518_20260616
  candidate_model_validation_20260614
  loss_ablation_portfolio_validation_20260614
  multi_loss_validation_20260614
  downside_topfocus_validation_20260616
  lag1_checkpoint_sweep_m0_20260616
  v9_avgw3_filter095_validation_20260616
  v9_avgw3_open_ledger_20260616
  open_reranker_current_v9_negfilter_20260617
  diagnostics_negfilter_drop3_20260617
  conditional_negfilter_breadth_20260618
  state_triggered_target_20260617
  breadth_triggered_target_20260617
  breadth_triggered_market_20260617
```

### Next Step

Run representative sampling for legacy `backtest_results_*` directories, then
archive a first small batch of clearly superseded legacy backtest outputs.

## Phase 8 Legacy Backtest Results Sampling

### Completed

Created:

```text
reports/codebase_cleanup_20260618/legacy_backtest_sampling.csv
reports/codebase_cleanup_20260618/legacy_backtest_sampling.md
```

The sampling pass reviewed archive candidates matching `backtest_results_*`.
No directories were moved.

Candidate count:

```text
backtest_results_* archive candidates=98
```

Group counts:

```text
backtest_results_exp_*=52
backtest_results_test_plan_*=26
backtest_results_switch_value_*=9
backtest_results_temporal_*=2
backtest_results_summary_*.txt=2
other smoke/cache/topstable/retention outputs=7
```

Representative samples checked:

```text
backtest_results_exp_base_avgw3_val
backtest_results_test_plan_share_ledger_primary_test
backtest_results_switch_value_20260604_v9_alpha_baseline_test
backtest_results_temporal_full_eval_20260604
backtest_results_summary_20260528.txt
backtest_results_v9_retention_20260531
```

Observed evidence:

```text
exp/test_plan samples contain diagnostics, monthly/yearly summaries, returns, and execution-cost CSVs.
backtest_result_snapshots contains historical reports for long-only, switch-value, temporal, and model-strategy comparisons.
```

Recommendation:

```text
Start small-batch archives with backtest_results_exp_* and backtest_results_test_plan_*.
Hold switch_value, temporal, retention/topstable, and smoke/cache outputs for a second pass.
Do not use broad --class experiment_output alone, because it includes active candidate evidence.
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_review_docs.py `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py -q
```

Result:

```text
14 passed
```

Content spot-check:

```text
rg "98|backtest_results_exp_\*|backtest_results_test_plan_\*|Snapshot Coverage|Do not move protected|candidate_model_validation_20260614" `
  reports\codebase_cleanup_20260618\legacy_backtest_sampling.md
```

### Next Step

Add a filtered archive option or explicit batch list for legacy backtest outputs
only, then dry-run the first small `backtest_results_exp_*` batch.

## Phase 8 Archive Filter Refinement

### Completed

Added name-based filtering to the archive move builder and CLI:

```text
experiments/archive_plan.py
run/archive_from_plan.py
```

New dry-run filters:

```text
--name-prefix <prefix>
--name-glob <glob>
```

This allows a safe legacy backtest batch such as:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_exp_ `
  --limit 10
```

without including active candidate evidence such as:

```text
candidate_model_validation_20260614
open_reranker_current_v9_*
v9_avgw3_filter095_validation_20260616
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

Dry-run checks:

```text
run\archive_from_plan.py --class experiment_output --name-prefix backtest_results_exp_ --limit 10
```

Result: 10 moves, all `backtest_results_exp_*`.

```text
run\archive_from_plan.py --class experiment_output --name-glob "backtest_results_summary_*.txt"
```

Result: 2 moves, both legacy summary txt files.

No directories were moved.

### Next Step

Run one small execute batch for `backtest_results_exp_*` after reviewing the
dry-run list again.

## Phase 8 First Legacy Backtest Archive Batch

### Completed

Dry-run reviewed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_exp_ `
  --limit 10
```

Then executed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_exp_ `
  --limit 10 `
  --execute
```

Moved 10 legacy backtest result directories:

```text
backtest_results_exp_ablate_fundamental_avgw3_val
backtest_results_exp_ablate_fundamental_share_ledger_val
backtest_results_exp_band20_mintrade_val
backtest_results_exp_band20_rankshrink_val
backtest_results_exp_band20_ranktilt_val
backtest_results_exp_band20_volpen_val
backtest_results_exp_base_avgw3_val
backtest_results_exp_base_band20_val
backtest_results_exp_blend_raw25_avg75_val
backtest_results_exp_blend_raw50_avg50_val
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=201
archive_candidate=162
protect=18
review=21
remaining backtest_results_exp_* candidates=42
archived backtest_results_exp_* directories=10
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

Spot checks:

```text
backtest_results_exp_base_avgw3_val exists=False
archive/experiments_202606/backtest_results_exp_base_avgw3_val exists=True
```

### Next Step

Continue `backtest_results_exp_*` in small batches, or pause to inspect the
remaining 42 names before the next execute batch.

## Phase 8 Second Legacy Backtest Archive Batch

### Completed

Executed the next already-reviewed small batch:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_exp_ `
  --limit 10 `
  --execute
```

Moved 10 legacy backtest result directories:

```text
backtest_results_exp_blend_raw75_avg25_hold100_stress_2x
backtest_results_exp_blend_raw75_avg25_hold100_stress_3x
backtest_results_exp_blend_raw75_avg25_hold100_stress_lag1
backtest_results_exp_blend_raw75_avg25_hold100_test
backtest_results_exp_blend_raw75_avg25_stress_2x
backtest_results_exp_blend_raw75_avg25_stress_3x
backtest_results_exp_blend_raw75_avg25_stress_lag1
backtest_results_exp_blend_raw75_avg25_test
backtest_results_exp_blend_raw75_avg25_val
backtest_results_exp_consensus_geo_test
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=191
archive_candidate=152
protect=18
review=21
remaining backtest_results_exp_* candidates=32
archived backtest_results_exp_* directories=20
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

Spot checks:

```text
backtest_results_exp_blend_raw75_avg25_test exists=False
archive/experiments_202606/backtest_results_exp_blend_raw75_avg25_test exists=True
```

### Next Step

Dry-run and review the next `backtest_results_exp_*` batch before executing.

## Phase 8 Final Legacy `backtest_results_test_plan_*` Archive Batch

### Completed

Executed the final already-reviewed `backtest_results_test_plan_*` batch:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_test_plan_ `
  --limit 10 `
  --execute
```

Moved 6 legacy test-plan backtest result directories:

```text
backtest_results_test_plan_stress_lag1_val
backtest_results_test_plan_v9_avgw3_test
backtest_results_test_plan_v9_avgw3_val
backtest_results_test_plan_v9_raw_test
backtest_results_test_plan_v9_raw_val
backtest_results_test_plan_v9_smoke5
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=133
archive_candidate=94
protect=18
review=21
remaining backtest_results_test_plan_* candidates=0
archived backtest_results_test_plan_* directories=26
remaining backtest_results_* candidates=20
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

### Next Step

Dry-run and review the remaining `backtest_results_*` candidate groups. The
lowest-risk next batch is likely the summary text files:

```text
run\archive_from_plan.py --class experiment_output --name-glob "backtest_results_summary_*.txt"
```

## Phase 9 Legacy Backtest Summary Text Archive

### Completed

Dry-ran and executed the summary-text batch:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-glob "backtest_results_summary_*.txt" `
  --execute
```

Moved 2 legacy summary text files:

```text
backtest_results_summary_20260516.txt
backtest_results_summary_20260528.txt
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=131
archive_candidate=92
protect=18
review=21
experiment_output candidates=54
checkpoint_or_model candidates=34
archive_or_cache candidates=4
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

### Next Step

Review the remaining `backtest_results_switch_*` and small smoke outputs before
moving them. Do not move active V9/open-reranker/loss validation evidence until
their ledgers prove the conclusions are represented elsewhere.

## Phase 10 Legacy Switch-Value Backtest Archive

### Completed

Reviewed switch-value coverage in:

```text
backtest_result_snapshots/20260530_switch_value_fixed_report.md
```

Representative directories were small raw-output bundles containing
`switch_value_config.json`, diagnostics, returns, and summary CSV files. The
historical conclusions are represented in the snapshot report, so the raw
`backtest_results_switch_value_*` directories were archived.

Executed:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_switch_value_ `
  --execute
```

Moved 9 legacy switch-value backtest result directories:

```text
backtest_results_switch_value_20260604_top3_pv1m_raw_alpha_val
backtest_results_switch_value_20260604_top3_pv1m_raw_newmodel_val
backtest_results_switch_value_20260604_v9_alpha_baseline_test
backtest_results_switch_value_20260604_v9_alpha_baseline_val
backtest_results_switch_value_20260604_v9_avgw3_alpha_val_pv1m
backtest_results_switch_value_20260604_v9_avgw3_switch_val_pv1m
backtest_results_switch_value_20260604_v9_baseline_layer_smoke
backtest_results_switch_value_20260604_v9_baseline_layer_test
backtest_results_switch_value_20260604_v9_baseline_layer_val
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=122
archive_candidate=83
protect=18
review=21
experiment_output candidates=45
checkpoint_or_model candidates=34
archive_or_cache candidates=4
remaining backtest_results_* candidates=9
archived backtest_results_switch_value_* directories=9
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

### Next Step

Review the last 9 scattered `backtest_results_*` outputs. The cache/smoke
directories are likely low risk, while temporal/retention/topstable outputs
should be checked against the temporal and V9-retention snapshot reports before
moving.

## Phase 11 Legacy Smoke/Cache Backtest Archive

### Completed

Dry-ran and executed the low-risk smoke/cache backtest outputs:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-glob "backtest_results_switch_cache_smoke*" `
  --execute

C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_small_account_smoke_ `
  --execute
```

Moved 5 legacy smoke/cache directories:

```text
backtest_results_switch_cache_smoke_fast2
backtest_results_switch_cache_smoke_read
backtest_results_switch_cache_smoke_read5
backtest_results_switch_cache_smoke_write
backtest_results_small_account_smoke_20260612
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=117
archive_candidate=78
protect=18
review=21
experiment_output candidates=40
checkpoint_or_model candidates=34
archive_or_cache candidates=4
remaining backtest_results_* candidates=4
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

### Next Step

Check the remaining temporal/retention/topstable backtest directories against
the snapshot reports before moving them.

## Phase 12 Final Legacy `backtest_results_*` Archive Batch

### Completed

Checked the final temporal/retention/topstable directories against the existing
snapshot reports, especially:

```text
backtest_result_snapshots/20260531_temporal_longonly_metric_report.md
backtest_result_snapshots/20260531_model_strategy_comparison_report.md
```

The current official open-ledger and cutoff evidence remains protected in:

```text
v9_avgw3_open_ledger_20260617
v9_avgw3_extend_to_20260518_20260616
```

Dry-ran and executed exact-prefix archive moves for:

```text
backtest_results_temporal_full_eval_20260604
backtest_results_temporal_retention_20260604_v10_v9warm_toploss
backtest_results_topstable_epoch9_val_avgw3
backtest_results_v9_retention_20260531
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=113
archive_candidate=74
protect=18
review=21
experiment_output candidates=36
checkpoint_or_model candidates=34
archive_or_cache candidates=4
remaining backtest_results_* candidates=0
top-level backtest_results_* entries=0
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

### Next Step

Move from legacy backtest outputs to the remaining experiment-output categories:
candidate/loss validation, open-reranker attack candidates, market overlays, and
reranker datasets. These need category-specific ledgers before any broad move.

## Phase 13 Remaining Cleanup Queue

### Completed

Created a post-backtest cleanup queue:

```text
reports/codebase_cleanup_20260618/remaining_cleanup_queue.md
```

Current remaining archive candidates:

```text
experiment_output=36
checkpoint_or_model=34
archive_or_cache=4
```

The remaining experiment outputs are now grouped into:

```text
training_validation=13
reranker_artifact=8
open_reranker_attack=7
market_overlay=4
v9_strategy_evidence=3
other=1
```

The remaining checkpoint/model outputs are now grouped into:

```text
loss_ablation_checkpoints=15
alpha_checkpoints=8
reranker_checkpoints=6
other_models=5
```

### Next Step

Build the training-validation ledger before moving any M0/A0/A4/loss/lag1
directories or their matching checkpoints.

## Phase 8 Fourth Legacy Backtest Archive Batch

### Completed

Executed the next already-reviewed small batch:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_exp_ `
  --limit 10 `
  --execute
```

Moved 10 legacy backtest result directories:

```text
backtest_results_exp_market_dynamic_val
backtest_results_exp_market_none_val
backtest_results_exp_pairwise_w001_avgw3_val
backtest_results_exp_pairwise_w001_band20_val
backtest_results_exp_rank_smooth_w2_val
backtest_results_exp_rank_smooth_w3_val
backtest_results_exp_rank_smooth_w4_val
backtest_results_exp_rank_smooth_w5_val
backtest_results_exp_rebalance_band10_pv1m_stress_2x
backtest_results_exp_rebalance_band10_pv1m_stress_3x
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=171
archive_candidate=132
protect=18
review=21
remaining backtest_results_exp_* candidates=12
archived backtest_results_exp_* directories=40
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

Spot checks:

```text
backtest_results_exp_market_dynamic_val exists=False
archive/experiments_202606/backtest_results_exp_market_dynamic_val exists=True
```

### Next Step

Dry-run the remaining 12 `backtest_results_exp_*` candidates, then decide
whether to archive them in one final batch or split 10 + 2.

## Phase 8 Final Legacy `backtest_results_exp_*` Archive Batch

### Completed

Executed the final reviewed batch:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_exp_ `
  --limit 20 `
  --execute
```

Moved the remaining 12 legacy `backtest_results_exp_*` directories:

```text
backtest_results_exp_rebalance_band10_pv1m_stress_lag1
backtest_results_exp_rebalance_band10_pv1m_test
backtest_results_exp_rebalance_band20_stress_2x
backtest_results_exp_rebalance_band20_stress_3x
backtest_results_exp_rebalance_band20_stress_lag1
backtest_results_exp_rebalance_band20_test
backtest_results_exp_rebalance_band_val
backtest_results_exp_topic_pairwise_blend_70_val
backtest_results_exp_topic_pairwise_blend_80_val
backtest_results_exp_topic_pairwise_blend_90_val
backtest_results_exp_topret_avgw3_val
backtest_results_exp_topret_band20_val
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=159
archive_candidate=120
protect=18
review=21
remaining backtest_results_exp_* candidates=0
archived backtest_results_exp_* directories=52
remaining backtest_results_* candidates=46
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

Spot checks:

```text
backtest_results_exp_* archive candidates=0
archive/experiments_202606 backtest_results_exp_* directories=52
```

### Next Step

Proceed to the next legacy group: `backtest_results_test_plan_*`, using the
same name-filtered dry-run and small-batch archive workflow.

## Phase 8 First Legacy `backtest_results_test_plan_*` Archive Batch

### Completed

Executed the first reviewed `backtest_results_test_plan_*` batch:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_test_plan_ `
  --limit 10 `
  --execute
```

Moved 10 legacy test-plan backtest result directories:

```text
backtest_results_test_plan_fixed_names_avgw3_test
backtest_results_test_plan_fixed_names_avgw3_val
backtest_results_test_plan_share_ledger_avgw3_test
backtest_results_test_plan_share_ledger_avgw3_val
backtest_results_test_plan_share_ledger_candidate_sweep_val
backtest_results_test_plan_share_ledger_primary_stress_2x
backtest_results_test_plan_share_ledger_primary_stress_3x
backtest_results_test_plan_share_ledger_primary_stress_lag1
backtest_results_test_plan_share_ledger_primary_test
backtest_results_test_plan_share_ledger_smoke
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=149
archive_candidate=110
protect=18
review=21
remaining backtest_results_test_plan_* candidates=16
archived backtest_results_test_plan_* directories=10
remaining backtest_results_* candidates=36
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

### Next Step

Dry-run and review the next `backtest_results_test_plan_*` batch.

## Phase 8 Second Legacy `backtest_results_test_plan_*` Archive Batch

### Completed

Executed the second reviewed `backtest_results_test_plan_*` batch:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_test_plan_ `
  --limit 10 `
  --execute
```

Moved 10 legacy test-plan backtest result directories:

```text
backtest_results_test_plan_share_ledger_stress_2x
backtest_results_test_plan_share_ledger_stress_3x
backtest_results_test_plan_share_ledger_stress_lag1
backtest_results_test_plan_small_account_avgw3_test
backtest_results_test_plan_small_account_avgw3_val
backtest_results_test_plan_small_account_lots_avgw3_val
backtest_results_test_plan_small_account_raw_test
backtest_results_test_plan_small_account_raw_val
backtest_results_test_plan_stress_2x_val
backtest_results_test_plan_stress_3x_val
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=139
archive_candidate=100
protect=18
review=21
remaining backtest_results_test_plan_* candidates=6
archived backtest_results_test_plan_* directories=20
remaining backtest_results_* candidates=26
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

### Next Step

Dry-run the remaining 6 `backtest_results_test_plan_*` candidates, then archive
them in one final batch if the list is clean.

## Phase 8 Third Legacy Backtest Archive Batch

### Completed

Executed the next already-reviewed small batch:

```text
C:\Users\x\miniconda3\envs\torch\python.exe run\archive_from_plan.py `
  --class experiment_output `
  --name-prefix backtest_results_exp_ `
  --limit 10 `
  --execute
```

Moved 10 legacy backtest result directories:

```text
backtest_results_exp_consensus_geo_val
backtest_results_exp_consensus_min_val
backtest_results_exp_legacy_bear40_test
backtest_results_exp_legacy_bear40_val
backtest_results_exp_legacy_bear50_val
backtest_results_exp_legacy_bear60_val
backtest_results_exp_legacy_bear80_val
backtest_results_exp_market_dynamic_min35_val
backtest_results_exp_market_dynamic_min50_val
backtest_results_exp_market_dynamic_min65_val
```

Destination:

```text
archive/experiments_202606/
```

Regenerated source inventory and archive plan.

Current summary after the move:

```text
top-level inventory rows=181
archive_candidate=142
protect=18
review=21
remaining backtest_results_exp_* candidates=22
archived backtest_results_exp_* directories=30
```

### Validation

Focused pytest:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

Result:

```text
15 passed
```

Spot checks:

```text
backtest_results_exp_legacy_bear40_test exists=False
archive/experiments_202606/backtest_results_exp_legacy_bear40_test exists=True
```

### Next Step

Dry-run and review the next `backtest_results_exp_*` batch before executing.
