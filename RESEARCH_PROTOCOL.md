# Research and Forward-Test Protocol

## Frozen boundary

- Research cutoff: `2026-05-18`, inclusive.
- Forward-test start: `2026-05-19`.
- Training, validation, test selection, hyperparameter tuning, and checkpoint
  selection may only use observations through the research cutoff.
- Data after the cutoff is forward-test evidence. Do not use it to retrain,
  choose checkpoints, tune strategy parameters, or revise filters.

## Data directories

- `data/raw`: frozen research snapshot through `2026-05-18`.
- `data/forward_raw`: copy of the research snapshot plus observations from
  `2026-05-19` onward.
- `data/tracking_raw`: optional operational tracking data; it is not a
  research dataset.

Initialize and update forward data:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/init_forward_data.ps1
C:/Users/x/miniconda3/envs/torch/python data/update_daily.py --data-dir data/forward_raw
C:/Users/x/miniconda3/envs/torch/python scripts/update_forward_market_data.py --data-dir data/forward_raw
```

Generate the frozen forward Alpha only after both update commands succeed:

```powershell
C:/Users/x/miniconda3/envs/torch/python run/forward_frozen_strategy.py `
  --data-dir data/forward_raw `
  --end-date YYYY-MM-DD `
  --output forward_results/frozen_v9_avgw3/alpha.jsonl `
  --device cuda
```

## Account size

Capacity and execution sweeps use two account values:

- CNY 500,000
- CNY 1,000,000

Default liquidity checks are calibrated for this range rather than an
institutional CNY 100 million portfolio.

## Model-selection rule

The existing validation and test periods through `2026-05-18` remain research
periods. Results after `2026-05-18` are reported chronologically and are never
used to change the active model during the same forward-test campaign.
