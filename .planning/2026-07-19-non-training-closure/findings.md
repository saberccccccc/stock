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
