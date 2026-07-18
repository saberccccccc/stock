# Progress

## 2026-07-18

- P3 closed with 535 passing tests and a real provenance-enabled Adapter replay.
- Audited the existing Compact full monthly experiment. It has 24 completed
  windows, 485 unique OOS dates, complete model/alpha hashes, and 16 ledger cells
  for the Compact candidate.
- Compact lacks P3 provenance and trade-level attribution because it predates
  those gates. It remains valid historical engineering evidence but is not
  promotion eligible and will not be rerun solely for metadata backfill.
- Added P3 provenance generation to the resumable staged strong controller;
  runtime sampling now includes the child-process tree while exact/selected
  transition and per-stage checkpoint recovery remain unchanged.
- Frozen the 24-window launch contract: 456 target epochs, estimated 16.05 hours
  and 14.96 GiB of artifacts. With 154.15 GiB free and a 30 GiB reserve, the
  disk gate passes.
- The expensive full run has not been launched in this implementation unit.
  The remaining code boundary is routing the staged full controller through the
  accepted Adapter delegate rather than merely retaining the legacy oracle
  subprocess path.
- Added `P4_STRONG_MONTHLY_ROLLING_EXECUTION_PLAN_20260718.md` as the detailed
  Chinese execution plan. It freezes the research contract, staged Adapter
  gate, single-window resume drill, 24-window sequential run, OOS ownership
  audit, 16-cell ledger evidence, Records, decision rules and Forward-only
  observation boundary.
- Completed P4-B: every staged strong trainer call now passes through the
  accepted Torch Adapter and a purged per-window ProjectDataset. The runner
  persists Dataset/Adapter metrics, supports recovery of a valid exact
  checkpoint after a pre-progress-write interruption, and records the actual
  Adapter/Dataset/provider sources and processor states in provenance.
- Real v14 read-only validation for `oos_2024_01` matched the frozen counts
  (965 Train, 117 Valid, 22 Predict), input width 250 and risk width 59. Full
  regression passed: 536 tests with one existing pandas FutureWarning.
- Completed P4-C in immutable v2 output after preserving the first pause-status
  failure. The controlled e6 pause resumed without rewriting its checkpoint;
  e15 ran epochs 7-15 and e19 ran 16-19. All three stages report the Torch
  Adapter, 22 selected OOS alpha dates passed byte-identical Adapter replay,
  19 artifacts and all six provenance manifests validated.
- Resource evidence recorded about 4.65 GiB process-tree peak RSS and one
  transient system-available-memory sample near 0.42 GiB. The sustained
  three-reading 0.75 GiB guard did not trigger; P4-D still requires a fresh
  resource check before launch.
- Added predeclared exact/selected dual inference profiles and isolated rolling
  manifests. The v2 24-window launch plan estimates 16.05 hours and 15.04 GiB;
  its 30 GiB post-run disk reserve gate passes.
- Launched P4-D from the hash-matched v2 launch plan after 540 passing tests.
  The user then requested a pause. `oos_2024_01` is complete with both exact
  and selected Alpha; `oos_2024_02/base_e6` stopped after epoch 5 of 6.
- Explicitly terminated both Windows parent and child Python processes after
  the tool-session termination left the child alive. GPU use returned to idle
  and a `strong_full_rolling_user_paused` event was appended without changing
  the experiment contract.
- Read-only recovery resolution points to `oos_2024_02/.../epoch_005.pt`, with
  optimizer reset and resume-start override both absent as required for an
  intra-stage continuation. Resume must use the same launch plan and `--resume`.
- User requested that training remain stopped while details are hardened.
- Added P4-R before any resume: correct stage semantics, strengthen profile
  provenance, audit checkpoint-selection diagnostics, and separate the current
  monthly audit arm from production-frequency selection.
- Updated the master and detailed P4 plans. No checkpoint, alpha, Registry,
  lifecycle state, model code, or running process was changed.
- Completed P4-R without training: generated the additive v2 profile, added
  canonical stages, ran 57 existing-checkpoint inference audits, and generated
  1/3/6-month fixed-window contracts with identical OOS coverage.
- The checkpoint audit rejected `rawtopstable` as a consistently transferable
  monthly selector and did not invent a replacement score from three windows.
- The full monthly strong run remains paused; the hardened recommendation is
  to compare update frequencies with a low-cost learner before paying the full
  monthly strong-model cost.
- Compatibility regression initially compared the in-memory checkpoint
  `agg_groups` tuples directly with their JSON list representation and failed.
  The assertion was corrected to compare canonical JSON values; no profile
  field or hash had drifted.
- Final focused regression passed: 58 tests. A direct compatibility check
  confirmed that v1 emits no new canonical-stage fields while v2 emits all
  three; no training or audit process remains active.
