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
