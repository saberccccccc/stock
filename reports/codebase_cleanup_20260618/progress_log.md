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
