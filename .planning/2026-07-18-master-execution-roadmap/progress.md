# Progress

## 2026-07-18

- Created `MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md` as the sole active
  execution sequence.
- Linked it from project rules, README, and current index.
- Marked the long-term roadmap and Qlib adoption plan as technical/historical
  references for sequencing purposes.
- Corrected the stale `2026-05-18 research boundary` wording in the project
  index and recorded the single-roadmap decision in ADR 0008.
- Verified required-reading, README, index, old-plan notices, active-plan
  pointer, split contract, and Qlib/open-ledger boundary references.
- No model, Registry, baseline, ledger, Forward, or lifecycle state changed.

## P4 contract hardening

- Kept the user-paused strong Rolling process stopped.
- Reconciled current Qlib-style alignment with the actual P2/P3/P4 state.
- Added a P4-R gate before resume and corrected stale master-plan statuses and
  strong-stage descriptions without altering immutable experiment artifacts.
- P4-R completed without training: hardened profile v2, canonical stage IDs,
  57-checkpoint inference-only transfer audit, generalized 1/3/6-month window
  contracts, and 58 passing focused regressions. Monthly strong Rolling remains
  paused pending a low-cost frequency comparison decision.

## P2A execution resumed

- Confirmed no residual LightGBM parity process was active and revalidated the
  completed 2024 byte-parity evidence.
- Added immutable second-window parity configs for `predict_2025`; the legacy
  and project Dataset arms differ only by `data.dataset_runtime`.
- Completed both 2025 arms: sample counts, best iteration, model bytes, raw
  alpha, and stitched Test alpha are identical. Legacy took 322.6 seconds and
  Project Dataset 329.8 seconds.
- Closed P2A with G1 passed. `legacy_iter` remains the default because the new
  path has no stable speed advantage; `project_dataset` remains a validated
  interface. Execution now moves to P2B Model Adapter convergence.
- Added the formal daily seeded quota sampler and missing LightGBM parameters
  to `LightGBMModelAdapter`, plus an explicit `project_adapter` runner path.
  The legacy trainer remains the default parity oracle. Targeted tests pass.
- Closed P2B after 2024 and 2025 both reproduced the exact legacy model and raw
  alpha bytes through `LightGBMModelAdapter`.
- Started P2C by adding a strong-model Project Dataset view for `y_seq`, raw
  returns, independent lag1 labels, risk, industry, and masks. It matches the
  legacy `PrecomputedMemmapDataset` on synthetic data and on six real v14 dates.
- Recorded ADR 0009: internal pilot checkpoint metrics cannot promote a model;
  formal checkpoint/profile choice waits for the complete P4 2024/2025 OOF
  ledger, avoiding a circular dependency and a three-month selection shortcut.
- Persisted remaining P2C work under
  `.planning/2026-07-18-p2c-strong-adapter-mainline/`; the master plan remains
  the active execution authority.
