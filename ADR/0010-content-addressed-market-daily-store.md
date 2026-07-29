# ADR 0010: Content-Addressed Market Daily Store

- Status: accepted
- Date: 2026-07-30
- Supersedes: none

## Context

The compatibility market-data layer stores one CSV per stock under both
`data/raw` and `data/forward_raw`. A normal daily update therefore opens more
than 5,000 files, and any changed source-file size or mtime invalidates the
single global OHLC matrix cache. The cache then requires a complete rebuild
even though only one trading date changed.

The project already has logical `DataView` boundaries and an
`OhlcvMatrixProvider`, so storage can change without changing selection,
Forward observation or realistic open-ledger semantics.

## Decision

Adopt one content-addressed, date-partitioned Parquet authority for A-share
equity and index daily bars. Immutable manifests select active partition
revisions, and an atomically replaced `CURRENT` pointer selects one manifest.
Ingestion timestamps and commands belong to manifests/events, not canonical
market rows, so identical data is idempotent.

Expose the store through a storage-neutral `MarketDailyProvider`. Selection and
Forward use the same physical store with different logical `DataView` maximum
dates. Replace the global OHLC cache only after parity with month-sharded,
source-hash-bound execution caches.

CSV remains the read-only parity oracle and immediate rollback backend until
all 24 formal ledger cells have identical proposals, orders, fills, rejection
reasons, costs, positions and equity curves.

Use the already installed PyArrow implementation first. DuckDB is optional and
requires measured benefit plus a later ADR.

## Consequences

- A daily update commits one equity partition rather than modifying thousands
  of stock files.
- Historical revisions create new content-addressed files and manifest
  generations; they never silently overwrite active evidence.
- A changed trading day invalidates only its monthly execution-cache shard.
- The migration temporarily uses more disk because CSV, Parquet and old/new
  caches coexist.
- No default backend changes during MD0-MD5. Any formal ledger difference
  blocks migration and is treated as a behavior change.
- CSV deletion requires a separate non-destructive archive review and explicit
  user approval.
