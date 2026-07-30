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
