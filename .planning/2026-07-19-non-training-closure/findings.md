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
