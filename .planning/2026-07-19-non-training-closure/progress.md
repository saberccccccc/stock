# Progress

## 2026-07-19

- Read the master roadmap, project rules, Registry baseline and decision rules.
- Verified available official CLIs for backtest replay, data/execution audits,
  scorecards, APM completeness, and Shadow lifecycle/replay.
- Created the complete non-training research-closure implementation plan.
- No training, backtest, Registry mutation, or lifecycle transition was run.
- Began NT0 and fetched the remote. `origin/master` can fast-forward to local
  `master`; `origin/main` has one divergent commit and will not be overwritten.
- The first secret-audit command failed because a generic extended-regex pattern
  used unsupported syntax. No push occurred; the retry must use compatible
  patterns and report paths only, never secret values.
- Paused NT0 to audit plan sprawl. Added one planning index, corrected stale
  status labels, and retained old plans as non-authoritative historical evidence.
- Completed NT0 with no force push: fast-forwarded remote `master`, created
  remote `model-experiments`, archived the legacy branch, and published annotated
  tag `accepted-research-20260719` while leaving divergent `origin/main` intact.
- Passed remote recovery acceptance from a fresh shallow clone at `83bba60`;
  the clone was clean, object-valid and removed after checking critical files.
- Completed NT1 by freezing `ledger_path_v3_t0001_nolookahead` into
  `registry/baseline_contract.json` and recording replay lineage in
  `registry/evidence_lineage.json`.
- Added explicit canonical/superseded evidence semantics to Registry. The
  resulting baseline scorecard has 24 unique cells: 16 selectable Val/Test and
  8 observation-only Forward, with zero missing coverage.
- Verified 23 frozen artifacts, retained v2/v3 as audit history, and confirmed
  all eight common v2/v4 ledger summaries have identical SHA-256 values.
- Validation passed: 17 focused tests and the full 550-test suite, with one
  existing pandas FutureWarning.
- Completed NT2 read-only boundary and execution audits for Val 2024, Test 2025
  and Forward 2026. All 5,332 market files and listing dates are readable; the
  only common execution-coverage blocker is historical ST status.
- Made provider-contract audit split-aware and genuinely read-only. It now
  reports selection OHLC cache identity mismatch and the v14 Forward coverage
  shortfall instead of rebuilding cache data or failing without a report.
- Recorded PIT, external-session, adjustment-lineage and real A-share board,
  suspension, limit-open, new-listing and ST samples in the NT2 quality report.
- Validation passed: 16 focused NT2 tests and the full 553-test suite, with one
  existing pandas FutureWarning.
- Began NT3 and completed the fixed baseline dry-run. It compiled Val 2024,
  Test 2025 and Forward 2026 into 24 cells with no missing alpha input and no
  parameter sweep beyond the frozen contract.
- Did not start the formal replay or mutate Registry: available memory was
  2.66 GiB, below the plan's 3 GiB resource stop threshold. NT3 remains active
  and resumable once the resource gate clears.

## 2026-07-30

- Expanded NT6 with
  `NT6_MARKET_DATA_PARQUET_INCREMENTAL_CACHE_PLAN_20260730.md`.
- The specification keeps the existing DataView, Provider and realistic ledger
  boundaries, introduces transactionally written daily Parquet partitions and
  month-sharded incremental OHLC caches, and requires full 24-cell parity before
  any default-backend switch.
- No Parquet migration, cache rebuild, training, Registry mutation or lifecycle
  transition was started.
- The first MD0 profile test incorrectly required a deterministic order for
  tied `Counter.most_common()` entries. The implementation was correct; the
  test now compares the date/count mapping instead of an undefined tie order.
- Completed the MD0 read-only baseline without rebuilding the stale OHLC cache
  or replaying the ledger. The report freezes source inventory, streamed CSV
  timing, cache identity and the existing baseline/artifact parity oracles.
- Completed the MD1 storage contract and ADR 0010. Focused coverage includes
  deterministic idempotency across input order, explicit revisions, immutable
  manifest verification, CURRENT-switch failure recovery, invalid OHLC/key
  rejection and concurrent-writer exclusion.
- Focused MD0/MD1/compatibility tests passed 23/23. The full regression suite
  passed 576 tests with one pre-existing pandas FutureWarning.
- MD2's first pilot exposed O(N^2) root-manifest growth because every daily
  generation copied all historical partition entries. Before full migration,
  the store was upgraded to a small root manifest referencing immutable monthly
  indexes.
- A verified recursive removal of the unregistered 3.3 MiB v1 pilot store was
  blocked by the command safety policy before execution. It remains untouched
  and is superseded by the separate `data/market_daily_candidate_v2` pilot;
  cleanup is deferred to the planned non-destructive archive review.
- Completed MD2 without changing the formal CSV backend. The resumable
  migration scanned the 5,332 stock CSVs once per year and migrated 2010-01-04
  through 2026-07-29 into 4,023 content-addressed daily Parquet partitions.
- All 17 annual audits passed exact key, value and dtype comparison: 13,313,700
  rows in total. Peak observed process RSS was 370,483,200 bytes.
- The active-store audit verified CURRENT, the immutable root manifest, 199
  immutable month indexes, all 4,023 Parquet physical hashes, schemas and row
  counts. Active Parquet is 398,301,688 bytes; the complete candidate store,
  including immutable history and migration progress, is 561,081,706 bytes.
- Added corruption tests for active partitions and month indexes. CSV remains
  the parity oracle and rollback path; MD3 is the next implementation unit.
- Completed MD3 with an isolated real-data pilot for 2026-07-29. Tushare daily
  supplied 5,524 equities and AkShare supplied the four frozen broad indices;
  both active coverages end on the requested date.
- The first real pilot correctly failed before any commit because this Tushare
  account permits only one `index_daily` request per minute and the endpoint
  requires one index code. The production default now composes Tushare equity
  with the project's established AkShare broad-index API; Tushare index mode
  remains explicit for higher-quota accounts.
- AkShare broad-index history does not expose turnover amount. Index `money`
  is therefore stored as zero with the mandatory progress semantic
  `source_unavailable_filled_zero`; it must not be consumed as observed amount.
- Replaying the same real date under a fresh progress manifest returned
  `already_present` for both partitions and left generation 2 and manifest hash
  unchanged. The physical-hash audit passed for both active partitions.
- Daily completion now records a hash-verified root snapshot instead of scanning
  every historical Parquet file. Full partition audits remain explicit CLI
  acceptance/maintenance operations. MD4 Provider parity is next.
- Completed MD4 with storage-neutral CSV and Arrow/Parquet backends behind one
  DataView-bounded `MarketDailyProvider`. Its return contract matches the
  existing field-to-date-by-code matrix API, including money scaling and
  deterministic `pre_close`/`pct_chg` derivation.
- The fixed 48-code Val/Test/Forward audit passed exact index, columns, values,
  dtypes and missing-position comparison for all seven stored numeric fields.
- The first two all-code attempts hit the 10-minute command ceiling. The audit
  initially opened all 5,332 CSVs once per split, then still repeated seven
  pivots and formatted large matrices as CSV solely for hashes. It now scans
  the full date range once, performs one multi-field pivot and uses binary
  index/column/dtype/value hashes. No incomplete report was accepted.
- The optimized 5,332-code full audit completed and passed for 2024 Val, 2025
  Test and Forward through 2026-07-29. CSV took 418.36 seconds and direct
  Parquet 395.83 seconds for the complete 638-calendar-day request.
- Direct Parquet is an authority/query backend, not yet a full-market backtest
  acceleration layer. MD5 month-sharded matrices are required before runtime
  promotion; the formal backend remains CSV.
