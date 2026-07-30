# Findings

- Registry, not historical discussion, defines the formal baseline as
  `ledger_path_v3_t0001_nolookahead`.
- Existing official CLIs cover Registry-driven replay, data-boundary audit,
  execution-coverage audit, scorecards, APM completeness, and Shadow replay.
- The baseline has both historical registrations and a later formal replay,
  so canonical evidence lineage must be resolved before another leaderboard.
- Historical ST execution coverage remains explicitly incomplete.
- The repository migration is complete locally, but remote durability still
  needs a no-force-push audit and recovery check.
- The repository had one authoritative master plan but several old documents
  still looked active, including a Qlib plan marked `in_progress` and a stale
  current-index statement. The content remains useful; the authority labels
  were the defect.
- The accepted hierarchy is now master plan -> one `.active_plan` ledger -> at
  most one current-stage technical specification. Historical plans remain
  immutable evidence rather than being deleted or moved and breaking links.
- NT0 found that `origin/master` could be fast-forwarded, while `origin/main`
  contained one independent commit. The safe solution was to leave `main`
  unchanged and publish accepted, working, archive and tag refs separately.
- A fresh shallow clone of remote `model-experiments` recovered the exact
  accepted commit with a clean status, valid objects and all critical entrypoints.
- The baseline Registry contained 24 legacy rows plus 16 formal Val/Test rows.
  Without an explicit canonical-evidence flag, the scorecard duplicate-weighted
  the same baseline cells.
- Formal workflow v4 is the latest complete Val/Test replay. Its eight shared
  ledger summaries are byte-identical to v2, so canonicalizing v4 changes
  evidence lineage rather than economic results.
- Forward currently has only eight legacy observation rows. They remain
  canonical but selection-ineligible until NT3 replaces them with a formal
  Forward replay.
- The frozen baseline inventory contains 23 artifacts with no missing files.
  Training checkpoint provenance remains historically unresolved, but the
  frozen-alpha ledger replay itself is complete and reproducible.
- Both physical market roots contain 5,332 readable stock files. Research is a
  physical superset through 2026-06-29 but its logical view is safely capped at
  2025-12-31; Forward is physically covered through 2026-06-30.
- OHLC, volume, money and all 5,332 listing dates cover Val, Test and Forward.
  Historical ST is not covered in any split. The current-name fallback cannot
  establish historical status and may apply present-day names retrospectively.
- The v14 cache ends on 2026-05-18 and cannot serve a full 2026-06-30 Forward
  feature replay. Fundamentals have effective dates and quality flags but end
  on 2026-05-15; external markets enforce a strictly-prior-session rule.
- The shared OHLC cache is currently bound to `data/forward_raw`, so a research
  provider sees an identity mismatch. The prior read-only provider audit could
  rebuild this cache; it now reports the mismatch without mutating it.
- Tushare daily OHLC is raw and the stored factor is fixed at 1.0. That is
  appropriate for execution, but v14 metadata lacks corporate-action/adjustment
  lineage for feature construction.
- NT3 dry-run resolves all three baseline alpha files and expands to exactly 24
  fixed cells with correct role-specific dates and data roots. The formal replay
  was not launched because available RAM was 2.66 GiB, below the predeclared
  3 GiB stop threshold.
- Daily market storage remains the main NT6 architecture debt: `data/raw` and
  `data/forward_raw` together hold about 3.0 GiB across more than 10,000 files,
  while the 0.95 GiB global OHLC cache is invalidated by any source-file change.
- The accepted NT6 direction is one partitioned Parquet authority with logical
  selection/Forward DataViews, a storage-neutral MarketDailyProvider and
  month-sharded incremental execution caches. CSV remains the parity oracle and
  rollback path until all 24 ledger cells are behaviorally identical.
- `pyarrow` is already available in the Torch environment. DuckDB is absent and
  is optional rather than a Phase-1 dependency.
- MD0 measured a 5,332-file, 0.945 GiB Forward CSV streamed range scan at
  22.91 seconds / 42.22 MiB/s. The 0.954 GiB global matrix ends at 2026-06-30
  and no longer matches the verified Forward source through 2026-07-29.
- MD1 uses content-addressed Parquet payloads, immutable hash-verified manifests
  and one atomic CURRENT pointer. Ingestion timestamps remain outside canonical
  rows so identical market data is a stable no-op.
- MD2 proves full-history storage parity against `data/forward_raw`: all 17
  annual comparisons were exact for keys, values and dtypes, covering
  13,313,700 rows and 4,023 trading dates from 2010-01-04 through 2026-07-29.
- Active Parquet payloads occupy 398,301,688 bytes versus about 0.945 GiB for
  the Forward CSV stock files. Immutable manifest/index history raises the full
  candidate footprint to 561,081,706 bytes but is not on the normal query path.
- One-time migration produced one immutable generation per trading day. This is
  acceptable evidence history for the candidate; normal daily ingestion creates
  only one new generation, and query cost follows the 199 active month indexes.
- Tushare `index_daily` requires `ts_code`; the current account permits one call
  per minute, so four serial broad-index requests are not an efficient default.
  The accepted MD3 composition is Tushare for the full A-share daily cross-section
  and AkShare for the four existing broad-index symbols.
- The real 2026-07-29 Tushare cross-section has 5,524 rows versus the 5,313 codes
  present in the migrated local CSV date. MD4 must compare requested/intersecting
  codes and expose universe additions separately; it must not assert equal global
  row counts between the legacy compatibility universe and the new authority.
- MD4 exact parity is now proven over every one of the 5,332 legacy codes for
  all seven stored numeric fields across Val 2024, Test 2025 and Forward through
  2026-07-29. The authority migration did not change values or missingness.
- Direct Arrow filtering plus wide pivot is not sufficient for repeated
  full-universe backtests: the full MD4 read took about 396 seconds versus 418
  seconds for CSV. Month-sharded dense matrices remain the correct MD5 runtime
  layer; Parquet remains the updateable authority and arbitrary-query layer.
- MD5 confirms the intended three-layer split is necessary: Parquet authority
  for safe updates and arbitrary queries, month-sharded dense matrices for
  repeated full-universe execution, and DataView for logical date boundaries.
- A 2026-07 monthly shard stores six dense fields in 5,355,504 bytes and reads
  the 21x5,313 universe in about 0.027 seconds after warm-up, versus about 14
  seconds through direct Parquet. This is the first material runtime gain in NT6.
- Cache invalidation is based on the active month-index hash, not the root
  manifest hash. Therefore adding August does not invalidate July, while a July
  revision necessarily creates a new July cache generation.
- MD6 now has three explicit execution-data modes: unchanged `legacy`, audit
  oracle `csv`, and candidate `monthly`. The formal default remains `legacy`;
  no existing command silently changes backend.
- Realistic execution masks for `monthly` are built from the same monthly
  high/low/volume/money frames as open/close/ADV. Their cache key binds every
  requested month's immutable generation and source month-index hash.
- The canonical Val/Test workflow contains summary evidence, while a later
  frozen-baseline acceptance run contains detailed path artifacts. Forward does
  not have an equivalent full-year detailed oracle, so MD6 uses direct CSV as a
  uniform audit oracle for all three splits.
- Project validation must use
  `C:\Users\x\miniconda3\envs\torch\python.exe`. The base interpreter currently
  has Pandas 3.0.3 and produces different datetime units than the formal Torch
  environment's Pandas 2.3.3, which can create false exact-parity failures.
- MD6 completed all 24 CSV oracle cells and all 24 monthly candidate cells.
  Val, Test and Forward each passed summary parity plus 48 detailed artifact
  comparisons; all 144 frames and their serialized file hashes are identical.
- Clean Test runtime fell from 61.053 seconds to 34.629 seconds and clean Forward
  runtime from 108.114 seconds to 22.937 seconds. Val timing is not comparable
  because a stdout interruption was resumed, though its behavioral parity is
  valid. Recorded subprocess RSS remained below 527 MiB.
- MD7 exposes one `ExecutionMarketDataContract` across Registry, Workflow,
  standalone experiment evidence and Daily Shadow. Every active caller can
  declare `legacy`, `csv` or `monthly`; the formal default is still `legacy`.
- Shadow manifests now freeze the backend and deterministic replay inherits it.
  Workflow v2 freezes the same fields in schema-valid configuration.
- Registry attribution and scorecard are artifact-only consumers. A static
  allowlist now rejects any new unregistered import of the legacy global-matrix
  or per-stock CSV internals. The repository audit passes with no unregistered
  imports and no stale allowlist entries.
- MD8 provisional acceptance passes parity, runtime-profile and RSS gates.
  Test OHLC loading is 6.44x faster and Forward is 31.99x faster. Test total
  runtime improves 1.76x because monthly OHLC is only 13.0% of elapsed time;
  realistic constraints and ledger execution are now the dominant work.
  Forward total runtime improves 4.71x.
- Old MD6 reports do not contain process I/O counters. The clean MD8 matrix was
  blocked before its first subprocess at 2.96 GiB free versus the immutable
  3.00 GiB gate. This is an evidence-completeness gap, not a failed performance
  result.
- MD9 supports only monthly/CSV dual-read because both expose the same six
  execution fields. Legacy is intentionally excluded from dual-read; its
  three-field facade cannot prove full execution-input parity.
- A real 48-code, 21-session July 2026 dual-read passed all six fields exactly.
  Full Val/Test/Forward observation remains required before promotion.
- Backend promotion is now fail-closed and versioned. It verifies report
  schemas, backend identities and SHA-256 hashes, and records actor/reason/time.
  Current audit remains blocked on MD8 clean I/O evidence and three full
  dual-read reports; active backend remains legacy.
- The old pre-launch check could start a roughly 0.5 GiB task when exactly
  3 GiB was free, violating the intended in-run reserve. MD8/MD9 launchers now
  require 3 GiB reserve plus 0.75 GiB measured task headroom.
- MD8's original evaluator covered ledger replay performance but not every
  acceptance item in the technical plan. A new isolated benchmark now records
  source first/warm reads, partition coverage, one-day local commit, affected
  month refresh, warm-cache hits, process I/O, RSS/system memory and physical
  integrity without mutating the formal candidate store.
- The real July 2026 replay loaded 111,369 rows over 21 sessions. Its latest
  5,299-row day committed in 0.066 seconds and the six-field 21-by-5,313 cache
  rebuilt in 0.161 seconds. Both are far below the fixed 10-second and
  30-second gates. The commit used 24 process reads, not a 5,000-file scan.
- A report-nested temp workspace produced a 303-character destination and hit
  Win32 MAX_PATH during the first attempt. Cleanup completed and the candidate
  source identity was unchanged. The benchmark now preflights path depth and
  automatically uses the system temp root when necessary.
- The first MD9 manager overwrote `last_transition` on rollback and allowed a
  rollback while legacy was already active. The transition logic is now a pure
  tested state machine: promotion requires legacy, rollback requires monthly,
  each record freezes from/to, actor, reason and time, and history is appended
  rather than discarded.
- Policy loading now rejects malformed transition paths or a last-transition
  pointer that disagrees with history. A real negative rollback drill exited 1
  and preserved the active policy SHA-256 exactly.
- MD8 acceptance previously recorded source paths without source hashes. It now
  freezes the matrix status, four Test/Forward performance reports and
  incremental benchmark SHA-256 values. The promotion audit requires all five
  passing gates and all six source hashes, so replacing an input cannot silently
  reuse an old acceptance decision.
- Added one resumable MD8-MD9 closure controller. It runs incremental evidence,
  clean matrix, clean acceptance and full dual-read in order, then stops at
  `ready_for_manual_promotion`; it contains no promote action. On the live
  machine it correctly stopped before spawning a child at 0.674 GiB available.
- The MD9 prose still described an obsolete CSV default/rollback. The governed
  implementation uses legacy as active/rollback, monthly as candidate and CSV
  only as the six-field shadow oracle. The technical plan now matches that
  actual contract.
- The v1 backend policy froze only the backend name while runner defaults
  repeated candidate paths. Policy v2 now freezes candidate store/cache roots;
  Contract and every MD8/MD9 runner consume those paths, and dual-read reports
  include the actual monthly identity.
- Promotion now rejects three additional failures: the active store manifest
  changed after incremental evidence, a split used another cache root, or a
  candidate path escapes the project. The real current manifest still exactly
  matches the incremental benchmark identity.
- The completion definition had stale requirements for a literal
  `data/market_daily` directory and CSV rollback. It now defines authority via
  policy v2 plus manifest and requires a `monthly -> legacy -> monthly` drill.
- Workflow and historical Shadow replay still carried candidate-path fallback
  literals. Workflow now delegates missing paths to policy v2. Historical
  monthly replay instead requires its original frozen store/cache roots and
  fails closed if an old manifest omitted them; it never substitutes today's
  candidate silently.
- ADR 0010 still called CSV the immediate rollback backend. Project rules make
  ADR history append-only, so ADR 0011 now supersedes only those switch/rollback
  clauses and records policy v2, identity-bound manual promotion, legacy
  rollback and the required monthly recovery drill.
