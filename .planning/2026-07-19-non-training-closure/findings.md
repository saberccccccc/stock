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
