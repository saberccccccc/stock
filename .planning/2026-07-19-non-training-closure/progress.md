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
- Completed MD5 with month-sharded raw OHLCV/money matrices bound to the active
  month-index SHA-256. A new day or revision invalidates only its source month;
  unaffected month CURRENT pointers and generations remain unchanged.
- Cache generations are staged, content-addressed and atomically activated.
  Explicit audits verify metadata, shape, file sizes and all field hashes;
  concurrent second writers are rejected.
- Cross-month reads preserve requested code order and derive `pre_close` and
  `pct_chg` only after stitching, matching the existing request-range semantics.
  Three basic market-data masks are also parity-tested: valid OHLC, zero volume
  and basic open tradability. They do not replace ST/listing/limit Providers.
- The real 2026-07 cache covers 21 dates by 5,313 codes. Six stored fields,
  two return fields and three basic masks exactly match direct Parquet. Initial
  build took 0.155 seconds; final warm cache read took 0.027 seconds versus
  13.965 seconds for direct Parquet. Active field data occupy 5,355,504 bytes.
- The legacy global cache and formal CSV backend are unchanged. MD6 ledger
  behavioral parity is the next gate.
- Began MD6 by adding an explicit execution-data backend contract to the direct
  ledger CLI, sweep CLI and Registry-driven wrapper. `legacy` remains default;
  `csv` is the direct audit oracle and `monthly` is the non-authoritative
  candidate.
- Monthly execution now loads all six raw fields once and passes them into the
  realistic-mask builder. Mask cache identity includes the active immutable
  monthly generations, preventing accidental reuse of global-matrix masks.
- Added `run/audit_open_ledger_backend_parity.py` to compare summary economics
  and six path artifacts by sweep key: equity, diagnostics, positions, orders,
  rejections and costs.
- The first focused run used the base Python by mistake and failed three exact
  datetime-dtype assertions (`datetime64[s]` versus `[us]`). Re-running with the
  project Torch interpreter passed 77 focused tests; this was environment drift,
  not a market-value or execution difference.
- Did not start the 5,000-stock 10-day dual replay because available memory fell
  from 2.85 GiB to 0.39 GiB. The largest process was a user application at about
  5.62 GiB; it was not terminated. The fixed 3 GiB resource gate remains active.
- Added the resumable fixed-matrix runner and compiled all six backend/split
  commands in dry-run: CSV and monthly over Val, Test and Forward, with four
  stresses and two capital sizes (24 cells per backend).
- The first real matrix invocation enforced its own resource gate and persisted
  `blocked_low_memory` at 0.429 GiB before launching CSV/Val. No partial result
  was accepted. Architecture, technical-plan and MD6 status documentation now
  describe the implemented harness and the remaining machine gate.
- Resumed MD6 after available memory recovered above 3 GiB. The first attached
  run lost its stdout pipe at the outer tool timeout and returned 120; the
  resumable ledger correctly retained partial rows without declaring completion.
- Completed all six backend/split sweeps. CSV and monthly each produced the
  fixed 24 cells. Val, Test and Forward parity reports passed all summary checks
  and all 144 equity/diagnostic/position/order/rejection/cost comparisons with
  exact values and identical file hashes.
- Clean Test total runtime improved 61.053s to 34.629s; clean Forward improved
  108.114s to 22.937s. MD6 is complete. MD7 call-site consolidation is the next
  NT6 unit; no default switch occurs yet.
- Final regression under the Torch environment passed 609 tests with one
  pre-existing Pandas FutureWarning.
- Completed MD7 call-site consolidation. Added one shared execution-market
  backend contract and propagated it through Registry backtests, Workflow,
  standalone ledger evidence and Daily Shadow/replay while keeping `legacy`
  as the explicit default.
- Added a machine-readable call-site policy and static audit. Formal entrypoints
  are contract-aware, attribution/scorecard remain artifact-only, and all
  legacy internal imports are classified. Focused tests passed 74/74 and the
  full suite passed 614 tests with one pre-existing Pandas FutureWarning.
- MD8 fixed performance acceptance is next. No training, long replay, Registry
  mutation, lifecycle transition or backend default switch occurred in MD7.
- Began MD8 with process I/O and available-memory instrumentation plus a fixed
  acceptance evaluator. Existing MD6 evidence is `provisional_pass`: parity,
  speed/profile and RSS pass, while process I/O evidence is absent.
- The clean MD8 matrix correctly stopped before `csv/val_2024` because free
  memory was 2.96 GiB versus the fixed 3.00 GiB minimum. It remains resumable;
  no resource threshold was weakened.
- MD8 instrumentation and acceptance regression passed the full 618-test suite
  with one pre-existing Pandas FutureWarning.
- Implemented MD9 exact monthly/CSV dual-read, atomic pass/fail evidence and
  Workflow/Shadow/Registry propagation. A real 48-code July 2026 pilot passed.
- Added a resumable Val/Test/Forward full-universe observation runner with the
  fixed 3 GiB memory gate.
- Added a versioned backend policy and fail-closed promotion/rollback manager.
  Promotion checks evidence schemas and hashes and requires actor/reason.
  Current audit correctly remains blocked; default is still legacy.
- Full regression passed 632 tests with one pre-existing Pandas FutureWarning.
  A real negative promotion attempt returned nonzero and left the policy
  SHA-256 unchanged.
- Corrected the resource gate to preserve 3 GiB during execution: launch now
  requires 3.75 GiB free (3 GiB reserve plus 0.75 GiB task headroom). The
  full dual-read runner correctly remained blocked at 2.10 GiB.
- Added and tested an isolated full-scale daily incremental benchmark. It
  replays one source month into a temporary store, excludes setup time, measures
  the final partition commit and cache refresh, verifies warm hits and hashes,
  and removes its workspace afterward.
- The first real attempt failed safely on a 303-character Windows path. Added
  short-temp fallback with a regression test, then reran successfully against
  the real 2026-07 candidate month.
- Extended MD8 acceptance so a final pass now requires both clean ledger I/O
  evidence and passing incremental evidence. The current combined result stays
  `provisional_pass` solely because old MD6 reports have no process I/O.
- Focused regression passed 7 acceptance/benchmark tests and 19 related
  storage/cache tests.
- Full repository regression passed 636 tests with one pre-existing Pandas
  FutureWarning. Available memory remained 2.284 GiB, so the 3.75 GiB clean
  matrix launch gate correctly remains closed.
- Hardened MD9 promotion/rollback into tested pure policy transitions with
  append-only history and strict backend-path validation. A temporary-policy
  drill proved default resolution changes monthly then returns to legacy.
- A real rollback attempt against the active legacy policy exited 1 with
  `rollback requires the monthly backend to be active`; policy SHA-256 remained
  `ABFB10B10CA892729FD700B8846B42E3ED02EE121DDDE2C8105739D88E2AE1D4`.
- Thirteen focused promotion, rollback and market-data-contract tests passed.
- Full regression after MD9 transition hardening passed 639 tests with the same
  single pre-existing Pandas FutureWarning.
- Bound MD8 acceptance to SHA-256 identities for its matrix, four performance
  reports and incremental benchmark. Promotion now rejects missing source
  hashes even when an acceptance file claims `passed`.
- Added `run/close_nt6_market_backend.py` and a read-only closure inspector. The
  controller is resumable, enforces phase order and the 3.75 GiB launch gate,
  writes one status artifact and deliberately has no automatic promotion.
- The first source-rehash test run exposed an obsolete closure fixture that used
  placeholder hashes without source files. The fixture was upgraded to real
  files and SHA-256 identities; the rerun passed all 15 focused tests.
- Fifteen focused closure/acceptance/policy tests passed. The live controller
  stopped at `clean_matrix` without launching a subprocess because only
  0.674 GiB was available; no Python process remained afterward.
- Full regression after closure orchestration and source re-hashing passed 643
  tests with the same single pre-existing Pandas FutureWarning.
- Upgraded the backend policy schema to v2 and froze candidate store/cache
  roots. Contract, benchmark, clean-matrix, dual-read and closure-controller
  defaults now consume the policy instead of repeating path literals.
- Added monthly store/cache identities to exact dual-read reports. Promotion
  binds the current active manifest and rejects store updates, wrong cache
  roots and project-path escapes.
- Updated the NT6 completion definition to use policy/manifest authority and a
  real monthly-to-legacy-to-monthly recovery drill.
- Twenty-eight focused identity, policy, dual-read, runner and closure tests
  passed. The real call-site audit also passed and controller dry-run froze the
  expected candidate paths.
- Full regression after the v2 candidate-identity contract passed 646 tests
  with the same single pre-existing Pandas FutureWarning.
- Removed the final candidate-path fallback literals from Workflow and
  historical Shadow replay. Added fail-closed replay validation plus a policy
  default propagation test.
- Twenty-nine Workflow/Shadow/Contract tests passed and the real call-site
  audit passed again.
- Full regression after formal-entrypoint fallback removal passed 648 tests
  with the same single pre-existing Pandas FutureWarning.
- Added ADR 0011 to govern policy-v2 promotion and supersede ADR 0010's stale
  CSV rollback clause without rewriting ADR history.
- Fixed the MD9 matrix's real-runtime alpha date normalization and added a
  regression covering mixed string and pandas timestamp inputs.
- Eighteen focused tests passed, followed by 649 full repository tests with the
  same single pre-existing Pandas FutureWarning.
- Resumed the closure controller from its persisted MD8 evidence. It skipped
  the accepted clean matrix and completed Val 2024, Test 2025 and Forward 2026
  dual-read reports in 94.4 seconds.
- All dual-read and promotion-audit gates passed. The controller stopped at
  `ready_for_manual_promotion` as designed; the active backend is still legacy.
