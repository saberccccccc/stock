# Findings

## 2026-07-18 Detailed-plan review

- P2C and P3 have passed their engineering gates. The accepted strong-model
  path is `ExistingStrongTrainerDelegate` through `TorchStrongAlphaAdapter`.
- The resumable staged rolling controller still invokes the legacy trainer as
  a subprocess. Routing all three stages through the accepted Adapter delegate
  is therefore the final pre-run architecture boundary.
- The historical Compact monthly experiment is complete enough for engineering
  comparison: 24 windows, 485 uniquely owned OOS dates and 16 ledger cells.
  Its performance was rejected, and it will not be rerun solely to backfill
  newer provenance fields.
- The frozen strong rolling contract contains 24 monthly windows, 456 target
  epochs, exact-checkpoint stage transitions, an estimated 16.05 hours of GPU
  work and 14.96 GiB of artifacts. The disk gate passed with a 30 GiB reserve.
- Formal selection remains fixed to 2024 Val and 2025 Test. 2026 Forward is an
  observation-only phase after the candidate and portfolio contract are frozen.
- The formal execution evidence must use one continuous realistic open-price
  share-ledger, 50w/100w capital and normal/lag1/cost2x/capacity_3pct stresses.
- Contract hardening found that the historical `base_e6` stage already uses
  `lag1_loss_weight=0.25`; `lag1_low_lr_e15` changes learning rate and optimizer
  state but does not introduce lag1 supervision. The legacy stage names are
  therefore semantically misleading.
- Original e6/e15 command files still exist and should be upgraded from
  inferred to confirmed provenance. The e16-e19 command remains reconstructed
  from the exact component equation, checkpoint architecture, and parent
  command, and must remain labelled as such.
- The earlier three-window pilot showed that `rawtopstable_h5_top0p6` did not
  reliably transfer to the next OOS month. This is a checkpoint-selection
  diagnostic gap, not evidence that the Rolling framework itself is wrong.
- The 4y Train / 6m Valid / 1m OOS monthly schedule is a project research arm,
  not a Qlib default or a proven production retraining frequency.
- P4-R generated a hardened v2 profile without changing v1, added canonical
  stage identities, and evaluated all 57 existing pilot checkpoints by
  inference only. `rawtopstable`, `rawtopret`, and validation Alpha do not have
  a directionally consistent relationship with next-month OO/lag1 results
  across the three pilot windows.
- The generalized fixed-window builder produces 24 monthly, 8 quarterly, or 4
  half-year windows with the same 485 uniquely owned 2024/2025 OOS dates.
  These are comparable contracts, not performance-selected frequencies.
