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
