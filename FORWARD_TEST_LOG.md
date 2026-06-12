# Frozen Forward Test Log

## Campaign

- Frozen strategy: `FROZEN_FORWARD_STRATEGY.md`
- Forward start: 2026-05-19
- Retraining or parameter changes: none
- First complete data date: 2026-06-11

## Data Audit

- Individual stocks updated: 5,310
- Stocks without new rows: 22, primarily stale, suspended, or delisted
- Latest complete stock date: 2026-06-11
- Broad indices updated through: 2026-06-11
- All 31 Shenwan industry indices updated through: 2026-06-11
- Valid forward Alpha dates: 18, from 2026-05-19 through 2026-06-11
- Realized ledger dates: 17, from 2026-05-20 through 2026-06-11

An initial calculation was discarded because the copied broad-index files
ended on 2026-05-15 and the industry-index files ended on 2026-05-08. The
valid results below were regenerated after those files were updated and the
forward inference cache was rebuilt.

## First Ledger

| Account | Cumulative return | Maximum drawdown | Positive days | Average names |
|---|---:|---:|---:|---:|
| CNY 500,000 | -6.24% | 7.93% | 7 / 17 | 28.9 |
| CNY 1,000,000 | -6.43% | 8.04% | 7 / 17 | 29.7 |
| CSI 300 benchmark | -2.69% | n/a | n/a | n/a |

The strategy underperformed the CSI 300 by approximately 3.55 percentage
points for CNY 500,000 and 3.74 percentage points for CNY 1,000,000.
Annualized figures are not decision-useful for this 17-day sample.

## Evidence

| Artifact | SHA256 |
|---|---|
| Forward Alpha JSONL | `4E99B81B79BE391DF65170FA33A1EF6F48F308B4F63AC455CF1B0A5BF78D16BA` |
| Execution summary CSV | `2161A57E81EB9D46FCD848CF7300597A3B3469FEB2C54AA58F83B3D586C9AF55` |

Artifacts are under `forward_results/frozen_v9_avgw3`.

## Interpretation Rule

This is a negative start, but it is not a tuning signal. Continue the frozen
ledger unchanged. Review at fixed milestones of 40 and 60 realized trading
days, with cumulative return, benchmark-relative return, maximum drawdown,
turnover, and execution failures reported each time.
