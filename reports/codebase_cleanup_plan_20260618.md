# Codebase Cleanup Plan 2026-06-18

## Purpose

The project has reached a point where more ad-hoc scripts make results harder
to trust.  The cleanup goal is not cosmetic.  It is to make training,
alpha-generation, execution backtests, stress tests, and reports reusable and
less error-prone.

Primary constraints:

- preserve current working results and checkpoint compatibility;
- do not move large data/checkpoint/result folders in the first pass;
- avoid retraining or heavy backtests during refactor;
- keep old `run/*.py` scripts runnable as wrappers while extracting shared code;
- add small smoke tests for extracted utilities.

## Current Diagnosis

### Stable Core

These are reusable core areas and should stay:

```text
core/
data/
backtest/
tests/
configs/
reports/
```

Important existing reusable modules:

```text
core/model.py
core/train_utils.py
core/config.py
data/pipeline.py
backtest/runtime.py
backtest/engine.py
backtest/reports.py
run/backtest_retention_open_ledger.py
```

### Main Pain Points

1. `run/` contains many one-off experiment scripts.
2. Alpha JSONL transforms are duplicated across scripts.
3. Open-price share-ledger backtest parameters are repeated manually.
4. Forward / validation / test date windows are passed ad hoc.
5. Candidate summary tables are rebuilt with one-off PowerShell snippets.
6. Result directories are mixed with source directories at repo root.
7. PID/log files from old runs remain at repo root.

## Target Structure

Do not create all of this at once.  This is the target shape after gradual cleanup:

```text
core/
  config.py
  model.py
  train_utils.py
  research_protocol.py

data/
  pipeline.py
  update.py
  factors...

backtest/
  runtime.py
  engine.py
  reports.py
  open_ledger.py              # extracted open-price share-ledger engine
  execution.py                # cost, lot, ADV, limit rules
  presets.py                  # official parameter presets
  stress.py                   # lag1 / cost2x / capacity wrappers

alpha/
  io.py                       # JSONL read/write, date checks
  transforms.py               # maxret095, edge rerank, negfilter
  market_overlays.py          # breadth/state triggered target/mult overlays
  diagnostics.py              # trigger counts, top bucket diagnostics

experiments/
  registry.py                 # candidate names, alpha paths, output paths
  leaderboard.py              # unified candidate summary

run/
  backtest_open_ledger.py     # thin CLI wrapper
  generate_alpha.py           # thin CLI wrapper
  apply_alpha_transform.py    # thin CLI wrapper
  train.py                    # existing training CLI
  train_temporal.py           # existing temporal CLI

reports/
  ...
```

## Cleanup Order

### Phase 0: Safety Snapshot And Inventory

Status: next step.

Tasks:

1. Confirm no active Python process.
2. Create a source inventory table:
   - core modules;
   - reusable backtest scripts;
   - alpha transform scripts;
   - one-off experiment scripts;
   - result/checkpoint/log folders.
3. Save inventory to:

```text
reports/codebase_cleanup_20260618/source_inventory.csv
reports/codebase_cleanup_20260618/source_inventory.md
```

No code behavior changes in this phase.

### Phase 1: Extract Alpha JSONL Utilities

Risk: low.

Reason:

Many scripts duplicate JSONL loading, date normalization, rank alpha creation,
and date mismatch checks.

Create:

```text
alpha/io.py
alpha/transforms.py
```

Move/reuse logic from:

```text
run/make_edge_rerank_from_full.py
run/make_negative_filter_from_full.py
run/make_conditional_negfilter_alpha.py
run/make_breadth_triggered_market_alpha.py
run/make_breadth_triggered_target_alpha.py
run/make_state_triggered_target_alpha.py
run/transform_alpha_for_execution.py
```

Keep the old scripts, but turn them into thin CLI wrappers calling `alpha/*`.

Tests:

```text
tests/test_alpha_io.py
tests/test_alpha_transforms.py
```

Acceptance:

- existing transform scripts still run;
- output JSONL is byte-identical or row-equivalent for a small fixture;
- date mismatch still raises;
- `py_compile` passes.

### Phase 2: Extract Open-Ledger Presets And Stress Runner

Risk: low to medium.

Reason:

Official backtest parameters are repeated often.  This has already caused
comparison-risk across validation, test, and forward.

Create:

```text
backtest/presets.py
backtest/stress.py
```

Put official parameters in one place:

```text
target_frac=0.006
hold_frac=0.10
max_new_names=5
rebalance_band=0.20
market_timing_mode=legacy
min_adv_cny=3_000_000
adv_participation_cap=0.05
limit_threshold=0.095
portfolio_values=500000,1000000
```

Expose named stress presets:

```text
normal
lag1
cost2x
capacity_3pct
```

Update:

```text
run/backtest_retention_open_ledger.py
run/sweep_open_ledger_params.py
run/summarize_open_ledger_candidates.py
```

Acceptance:

- official baseline summary matches current saved baseline;
- stress output path and metrics are unchanged for one fixture run;
- no heavy full rerun required.

### Phase 3: Extract Open-Price Share-Ledger Engine

Risk: medium.

Reason:

`run/backtest_retention_open_ledger.py` has become a real production-style
engine but still lives as a script.

Create:

```text
backtest/open_ledger.py
backtest/execution.py
```

Move reusable pieces:

- OHLC loading;
- ADV recomputation;
- open limit mask;
- desired target construction;
- lot/min commission/cash ledger execution;
- monthly/yearly/diagnostic output.

Keep:

```text
run/backtest_retention_open_ledger.py
```

as CLI wrapper.

Acceptance:

- old CLI command still works;
- `open_ledger_summary.csv` matches current output on a small 5-day fixture;
- no change in default behavior unless a new flag is explicitly passed.

### Phase 4: Candidate Registry And Leaderboard

Risk: low.

Reason:

Candidate comparison is now too manual.

Create:

```text
experiments/registry.py
experiments/leaderboard.py
```

Registry entries should include:

```text
candidate_name
alpha_path_val
alpha_path_test
alpha_path_forward
uses_row_market_mult
uses_row_target_frac
output_dir
notes
```

This replaces manual PowerShell table building.

Acceptance:

- can regenerate:

```text
reports/candidate_leaderboard_20260617/candidate_leaderboard.csv
```

from registry and saved summary files.

### Phase 5: Training Presets And Checkpoint Selection

Risk: medium.

Reason:

Training and checkpoint selection need to be separated from ad-hoc loss tests.

Create:

```text
core/training_presets.py
experiments/checkpoint_selection.py
```

Presets:

```text
v9_official_arch
m0_nomulti
topfocus_small
lag_aux_small
temporal_v10_warmstart
```

Checkpoint selection should support:

- Alpha IC minimum gate;
- Top30 return;
- open-price share-ledger validation;
- lag1;
- cost2x;
- drawdown;
- turnover;
- 50w and 100w separately.

Acceptance:

- no training behavior change;
- existing training configs map to explicit named presets;
- checkpoint report can explain why a checkpoint wins.

### Phase 6: Archive Root-Level Experiment Outputs

Risk: medium, because paths in scripts/reports may reference these folders.

Only after Phases 1-5:

Move old root clutter into:

```text
archive/experiments_202606/
archive/logs_202606/
archive/pids_202606/
archive/old_backtest_results/
```

Do not move:

```text
data/
cache/
checkpoints_loss_ablation_M0_nomulti/
checkpoints_exp/
v9_avgw3_open_ledger_20260617/
forward_results/
reports/
```

until all active reports are updated.

Acceptance:

- active scripts still work;
- active reports link to current locations;
- old outputs remain recoverable.

## Proposed Immediate Work

Start with Phase 0 and Phase 1 only.

Why:

- they reduce duplication immediately;
- they are low risk;
- they do not change model or backtest behavior;
- they make future experiments less error-prone.

Immediate task list:

1. Build source inventory.
2. Create `alpha/io.py`.
3. Create `alpha/transforms.py`.
4. Convert three simplest scripts to wrappers:

```text
run/make_edge_rerank_from_full.py
run/make_negative_filter_from_full.py
run/make_conditional_negfilter_alpha.py
```

5. Add tests for those transforms.
6. Run `py_compile` and focused tests.

## Non-Goals For The First Pass

Do not do these yet:

- move large result directories;
- delete old scripts;
- rename core checkpoints;
- change official backtest parameters;
- rewrite training loop;
- merge V9/M0/V10 model code;
- run heavy retraining.

The first pass should make the project easier to operate, not change strategy
results.
