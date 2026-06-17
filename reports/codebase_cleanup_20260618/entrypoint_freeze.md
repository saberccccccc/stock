# Entrypoint Freeze List 2026-06-18

## Purpose

This file defines which scripts are treated as stable user-facing entrypoints during cleanup.

These scripts may be internally refactored to call shared modules, but their default behavior, parameter meaning, and output schema should remain stable unless a new explicit flag is introduced.

## Frozen Entrypoints

### Official Backtest

```text
run/backtest_retention_open_ledger.py
```

Current role:

- official open-price share-ledger backtest;
- closest current proxy for small-account live execution;
- writes `open_ledger_summary.csv`, returns files, diagnostics files.

Freeze rules:

- do not change default execution semantics;
- do not change summary column names;
- do not change cost, ADV, limit, lot, or rebalance interpretation;
- when extracting modules, keep this script as a CLI wrapper.

### Alpha Execution Transform

```text
run/transform_alpha_for_execution.py
```

Current role:

- applies maxret095 and related execution-aware alpha transforms;
- used by official baseline pipeline.

Freeze rules:

- do not change max signal return behavior;
- preserve JSONL input/output structure;
- preserve date and code handling;
- extracted functions should live in `alpha/transforms.py`.

### V9 Inference Alpha Generation

```text
run/generate_v9_inference_alpha.py
```

Current role:

- generates tail-date V9 inference alpha without future labels;
- used to extend signal through 2026-05-15 while keeping returns capped at 2026-05-18.

Freeze rules:

- do not change checkpoint loading behavior;
- do not change `average` / `avgw3` signal meaning;
- do not change no-future-label behavior.

### Main Training Entrypoints

```text
run/train.py
run/train_temporal.py
```

Current role:

- main model training scripts.

Freeze rules:

- do not change default architecture or loss behavior during cleanup;
- new presets must be opt-in;
- checkpoint selection logic should be documented before becoming default.

### Candidate Validation

```text
run/validate_candidate_models.py
```

Current role:

- recent candidate validation and loss/reranker comparison.

Freeze rules:

- preserve existing output format;
- avoid using it as the only final decision tool;
- future unified validation should route through named execution families.

## Candidate Or Experimental Entrypoints

These can be refactored earlier because they are not the official baseline, but they still need compatibility wrappers.

```text
run/make_edge_rerank_from_full.py
run/make_negative_filter_from_full.py
run/make_conditional_negfilter_alpha.py
run/make_breadth_triggered_market_alpha.py
run/make_breadth_triggered_target_alpha.py
run/make_state_triggered_target_alpha.py
run/sweep_open_ledger_params.py
run/sweep_open_price_ledger_params.py
run/summarize_open_ledger_candidates.py
run/compare_open_ledger_diagnostics.py
```

Refactor target:

- move reusable logic into `alpha/`, `backtest/`, and `experiments/`;
- keep these files as thin CLI wrappers;
- add tests around extracted logic, not around every old script.

## Legacy Or One-Off Scripts

These should not be deleted yet. Later they can move to `run/legacy/` after registry and reports are stable.

Examples:

```text
run/backtest.py
run/backtest_trade_policy.py
run/backtest_trade_policy_v2.py
run/test_long_only_plan.py
run/v9_*_sweep.py
run/train_reranker*.py
run/validate_reranker*.py
```

Move only after:

- source inventory is complete;
- reports no longer reference them as active commands;
- official baseline can be reproduced from frozen entrypoints.

## Output Schema Freeze

The following file names and schemas should be treated as stable where possible:

```text
open_ledger_summary.csv
returns_*.csv
diagnostics_*.csv
candidate_leaderboard.csv
*_daily_alpha_top_order.jsonl
```

If a schema must change:

- create a new output file name or version;
- document the change in `reports/codebase_cleanup_20260618/`;
- do not silently overwrite older comparable outputs.

## Refactor Acceptance Rules

For any frozen entrypoint refactor:

1. Run `python -m py_compile` on the touched files.
2. Run focused tests if available.
3. For alpha transforms, compare small JSONL fixture output.
4. For backtest wrappers, compare a small date-window summary.
5. Do not run heavy full backtests unless explicitly needed.
6. Do not change official parameters during cleanup.
