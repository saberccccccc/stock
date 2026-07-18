# Performance Architecture - 2026-07-10

Purpose: keep training, signal generation, and realistic open-ledger
backtests fast, reproducible, and understandable on the current 16 GB RAM /
8 GB GPU workstation.

## Required Python Environment

All Torch/CUDA training, inference, and Torch-dependent tests must use:

```powershell
$env:PYTHON = "$env:USERPROFILE\miniconda3\envs\torch\python.exe"
& $env:PYTHON -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Verified environment: PyTorch `2.11.0+cu128`, CUDA enabled, NVIDIA GeForce
RTX 5070 Laptop GPU. The default `C:\Users\x\miniconda3\python.exe` is not
the project training environment and does not have Torch installed.

## Design Principles

The relevant lessons from `AI System: Principles and Architecture` are applied
at the workflow level:

1. Profile and remove repeated work before changing model numerics.
2. Separate immutable input preparation from parameter-dependent execution.
3. Reuse compact, contiguous representations for repeated inference/sweeps.
4. Batch work and append results incrementally instead of rebuilding outputs.
5. Keep optimization choices explicit and reproducible.

## Standard Pipeline

```text
v14 memmap dataset
  -> PyTorch train / checkpoint
  -> batched signal generation
  -> registry candidate
  -> sweep_open_price_ledger_params
       -> shared OHLC + ADV + realistic masks + prepared context
       -> parameter grid execution
       -> append-only summary
  -> registry scorecard / attribution / decision
```

## Current Optimizations

### Training

- Precomputed int16 memmap datasets avoid repeated feature normalization.
- Training uses one DataLoader worker because Windows multiprocess memmap file
  descriptors are unreliable in this project.
- CUDA runs now use pinned host memory and non-blocking host-to-device copies.
- AMP remains disabled: this project has observed Loss NaN with AMP. Pinned
  memory and non-blocking copies preserve the safe transfer-speed benefit.

### Signal Inference

- M0 raw/`avgN` variants retain the existing per-date `V9RankPredictor` score
  cache. `alpha/persistence.py` now centralizes the code-aligned rolling
  average and ranking on CPU, preserving original code order for parity and
  removing duplicated signal-generation orchestration. This is a structural
  cleanup; the prior V9 cache already avoided repeated GPU forwards.
- `DLPredictor` has one shared input-preparation path for Alpha, raw-Alpha,
  and horizon heads, and executes each model call under `torch.inference_mode()`.

### Backtest

- Prefer `run/sweep_open_price_ledger_params.py` over one-process-per-command
  manifest runs.
- The sweep loads OHLC, ADV, index returns, and realistic execution masks once.
- `prepare_open_ledger_context` caches immutable NumPy price/return matrices,
  code indexes, and per-day 60-day risk estimates across the sweep grid.
- Realistic execution masks have a disk cache under
  `cache/open_ledger_execution_masks/`. The key includes matrix-cache version,
  selected codes/date window, ST/listing metadata, and constraint settings.
  It is automatically invalidated by those inputs changing.
- Sweep summaries are appended in chunks, preserving resume behavior without
  repeatedly rewriting the entire CSV.

### Measured Baseline

On 2026-07-11, one realistic 2024 validation grid cell (5,108 stocks, 349
OHLC days, CNY 500k) measured 31.95 seconds cold and 6.82 seconds warm with
identical summary output. Cold constraint preparation was 22.35 seconds;
warm cached constraint loading was 0.24 seconds. This makes constraint-mask
caching the highest-value current backtest acceleration.

A warm two-capital grid over the same validation signal completed in 6.45
seconds: 0.22 seconds for constraint-mask loading, 0.11 seconds for prepared
context construction, and 4.77 seconds for both ledger cells. The shared
preparation remains outside the capital/parameter loops.

## Non-Negotiable Correctness Rules

- Use realistic open-price share-ledger execution for official evidence.
- Keep `val_2024` and `test_2025` as selection evidence only.
- Treat `forward_2026` as observation-only.
- Do not cache values that depend on candidate signals, portfolio state,
  capital, stress, or execution parameters.
- Keep all output rows stamped with signal and backtest date bounds.

## Operational Commands

```powershell
# Official registry-driven backtest (shared OHLC path)
python run/official_backtest_from_registry.py --candidate-id <candidate>

# Registry scorecard and attribution coverage
python run/attribution_from_registry.py
python run/scorecard_from_registry.py

# Training in the required Torch environment
& $env:PYTHON run/train.py --model v9 --label-family oo_lag1
```

## Next Performance Work

1. Use the built-in `--performance-report` JSON before considering deeper
   vectorization of the daily
   ledger loop; the loop contains stateful execution logic and must remain
   correctness-first.
2. Run CUDA training tests with `$env:PYTHON`; never use the default Conda
   base interpreter for Torch-dependent commands.
