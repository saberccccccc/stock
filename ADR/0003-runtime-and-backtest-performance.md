# ADR 0003: Runtime And Backtest Performance

- Status: accepted
- Date: 2026-07-11

## Context

The 16 GB RAM / 8 GB GPU workstation suffered repeated data preparation and
summary rewrite costs during large parameter sweeps.

## Decision

Use the Torch Conda environment, keep AMP disabled because it produces Loss NaN, prefer shared-OHLC batch sweep runners, cache immutable prepared ledger context and realistic execution masks only, append sweep chunks, and keep Windows memmap DataLoader workers at zero until a tested multiprocess design exists. Execution-mask cache keys include immutable matrix metadata, date/code coverage, listing/ST metadata, and constraint settings. Performance changes require parity coverage.

## Consequences

Speed claims need a reproducible benchmark or equivalent parity/performance
evidence. The official runner must bind selection splits to frozen research
data and forward splits to forward-only data. Numerical shortcuts that change
research protocol require a separate decision record.
