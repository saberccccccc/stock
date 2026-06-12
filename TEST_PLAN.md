# V9 Small-Account Test Plan

Updated: 2026-06-12

## Fixed constraints

| Item | Rule |
|---|---|
| Research data | `data/raw`, no observation after `2026-05-18` |
| Forward data | `data/forward_raw`, beginning `2026-05-19` |
| Account values | CNY 500,000 and CNY 1,000,000 |
| Hardware | 16 GB RAM, RTX 5070 Laptop GPU with 8 GB VRAM |
| Primary checkpoint | `checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt` |
| Main signals | V9 raw and V9 3-day average |
| Market defense | legacy |
| Selection rule | Validation first; test is confirmation, not a tuning target |

## Execution table

| ID | Phase | Action | Resource limit | Output | Pass condition | Status |
|---|---|---|---|---|---|---|
| T01 | Environment | Verify Python, CUDA, RAM, checkpoint and worktree | Read-only | Environment record | Python/CUDA/checkpoint available | Done |
| T02 | Data boundary | Verify all research stock rows end on or before 2026-05-18 | Under 4 GB RAM | Data quality report | No row after cutoff | Done |
| T03 | Cache | Validate metadata and referenced memmap files | No rebuild | Cache inventory | Required files exist and dates respect cutoff | Done |
| T04 | Pipeline smoke | Load cached dataset and score 5 dates | 16 GB RAM, no rebuild | Smoke alpha JSONL | No OOM, dates <= cutoff | Done |
| T05 | Frozen alpha | Generate full V9 raw validation/test alpha | One process, cached data | Two alpha JSONL files | Complete date coverage, cutoff respected | Done |
| T06 | Frozen alpha | Generate full V9 average_w3 validation/test alpha | Reuse base scores where possible | Two alpha JSONL files | Complete date coverage | Done |
| T07 | Ratio baseline | Test target3/hold30 and target3/hold40 | Existing retention engine | Baseline CSV | Results reproduce prior range | Done |
| T08 | Small account | Test CNY 500k and 1m with execution constraints | 5% ADV, min ADV grid | Capacity CSV | Low unfilled turnover; no hidden leverage | Done |
| T09 | Realistic costs | Add 100-share lots and minimum commission | No future data | Realistic execution CSV | Accounting reconciliation passes | Done |
| T10 | Fixed names | Compare Top 20/30/50 against 3% selection | Same signal/costs | Fixed-count comparison | Better practical risk/return than excessive diversification | Done |
| T11 | Stress | Run 1x/2x/3x costs and one-day execution delay | Selected candidates only | Stress report | No collapse under 2x costs | Done |
| T12 | Stability | Report yearly, quarterly and weak-regime performance | No parameter search | Stability report | No dependence on one short period | Done |
| T13 | Decision | Lock one primary and one fallback strategy | Validation-led | Frozen strategy manifest | Parameters and checkpoint hashes recorded | Done |
| T14 | Forward test | Update only `data/forward_raw` and run from 2026-05-19 | No retraining/tuning | Forward ledger | Chronological, untouched evidence | In progress: first ledger through 2026-06-11 |

## Candidate order

1. V9 raw, target 3%, hold 40%.
2. V9 average_w3, target 3%, hold 30%.
3. V9 average_w3, target 3%, hold 40%.
4. Fixed Top 20, Top 30 and Top 50 variants derived from the best signal.

## Memory rules

- Never use `--force-rebuild` during baseline tests.
- Keep data loading and training at `num_workers=0`.
- Run one model or backtest process at a time.
- Stop a run if process working set exceeds 13 GB for more than two minutes.
- Preserve at least 20 GB free disk before generating new memmaps.
- If a full score run fails, retry by date chunks; do not rebuild the dataset.

## Decision gates

- Do not start new model architecture experiments before T08 is complete.
- Do not inspect post-2026-05-18 performance while choosing research parameters.
- V10 and switch-value work remain paused unless the locked V9 baseline is
  reproducible and the realistic execution layer is complete.

## Initial findings

- Research daily data: 5,332 stocks, `2010-01-04` through `2026-05-18`.
- Raw price validation: zero bad close rows and zero bad volume rows.
- PIT coverage: fundamentals 99.81%, shareholder 98.42%, restricted release 97.96%.
- Existing cross-section and temporal memmaps are complete and stop at the cutoff.
- `pyarrow` was added because the copied environment could not read PIT Parquet caches.
- Five-date V9 CUDA smoke completed without rebuilding caches or exhausting memory.

## Locked research result

The execution simulator now uses a cash and whole-share ledger. It marks
positions to market daily, sells before buys, enforces available cash, and
charges per-trade minimum commission. This removed the previous sub-lot
residual-position artifact.

Primary strategy:

- Signal: V9 `average_w3`.
- Entry: `target_frac=0.006`, approximately Top 30.
- Exit: `hold_frac=0.10`.
- Account values: CNY 500,000 and CNY 1,000,000.
- Validation: annualized 53.27% / 56.39%, Sharpe 1.507 / 1.543, maximum
  drawdown 15.39% / 15.69%.
- Independent test: annualized 44.55% / 54.25%, Sharpe 1.912 / 2.113,
  maximum drawdown 11.70% / 12.20%.
- Average live names: 28.9-29.5.
- Average unfilled turnover: 0.7%-1.3%.

Stress results on validation:

| Scenario | CNY 500k annualized / Sharpe | CNY 1m annualized / Sharpe |
|---|---:|---:|
| Base costs | 53.27% / 1.507 | 56.39% / 1.543 |
| 2x costs | 39.30% / 1.207 | 44.41% / 1.298 |
| 3x costs | 26.56% / 0.904 | 33.00% / 1.044 |
| One extra trading-day delay | 29.99% / 0.983 | 30.87% / 0.991 |

Execution remains time-sensitive, but the selected strategy does not collapse
under the required 2x-cost gate. Full locked parameters and hashes are in
`FROZEN_FORWARD_STRATEGY.md`.
