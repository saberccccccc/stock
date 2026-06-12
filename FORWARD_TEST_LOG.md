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

## Discarded First Ledger

Status: **invalid for model assessment**. A subsequent PIT audit found that
the local fundamental cache contained 2026 Q1 reports for only 2,753 of 5,322
stocks as of 2026-05-19. The remaining 2,567 stocks were mostly still using
2025 annual reports even though sampled live endpoints already contained
their Q1 reports. The incremental downloader incorrectly used a 90-day
freshness rule and skipped these stocks.

| Account | Cumulative return | Maximum drawdown | Positive days | Average names |
|---|---:|---:|---:|---:|
| CNY 500,000 | -6.24% | 7.93% | 7 / 17 | 28.9 |
| CNY 1,000,000 | -6.43% | 8.04% | 7 / 17 | 29.7 |
| CSI 300 benchmark | -2.69% | n/a | n/a | n/a |

The strategy underperformed the CSI 300 by approximately 3.55 percentage
points for CNY 500,000 and 3.74 percentage points for CNY 1,000,000.
Annualized figures are not decision-useful for this 17-day sample.

## Corrected First Ledger

The fundamental cache was rebuilt to 5,318 Q1 reports out of 5,322 stocks
with any cached financial history, or 99.92% coverage. PIT validation found
zero records where `effective_date < end_date`. The forward inference cache
was then rebuilt and the same frozen strategy was rerun.

| Account | Cumulative return | Maximum drawdown | Positive days | Average names |
|---|---:|---:|---:|---:|
| CNY 500,000 | -6.34% | 8.12% | 5 / 17 | 28.7 |
| CNY 1,000,000 | -4.91% | 7.98% | 7 / 17 | 29.8 |
| CSI 300 benchmark | -2.69% | n/a | n/a | n/a |

The corrected Top 30 list retained an average of 25.1 names from the
uncorrected list, so approximately five names per day changed. Correcting the
financial cache improved the CNY 1,000,000 ledger by 1.52 percentage points,
but did not improve the CNY 500,000 ledger. Missing reports therefore
materially affected rankings but do not fully explain the negative forward
period.

## Evidence

| Artifact | SHA256 |
|---|---|
| Corrected forward Alpha JSONL | `0B62FD2A441C5D33A432E3055D9ED65BF7A7EFB41A09F348C54ACF2E5203E7AF` |
| Corrected execution summary CSV | `BCCC04B06C962FDEAFBF2EFE2B825C9A4F809E8EF2C51CC0690C465443029952` |

Artifacts are under `forward_results/frozen_v9_avgw3`.

## Interpretation Rule

This is a negative start, but it is not a tuning signal. Continue the frozen
ledger unchanged. Review at fixed milestones of 40 and 60 realized trading
days, with cumulative return, benchmark-relative return, maximum drawdown,
turnover, and execution failures reported each time.
