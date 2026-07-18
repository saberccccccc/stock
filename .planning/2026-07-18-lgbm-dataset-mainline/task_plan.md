# LightGBM Dataset Mainline Migration

## Goal

Integrate the project-native Dataset/DataHandler runtime into the formal
LightGBM rolling path without changing features, labels, windows, sampling,
model parameters, Registry state, or ledger behavior. Prove one-window parity
before considering a default-path switch.

## Phase 1 - Trace And Contract

Status: completed

- Trace the exact legacy sample flow in `run/rolling_lgbm_alpha.py`.
- Map it to `data/dataset_runtime.py`, providers, processors, and adapters.
- Define parity dimensions and a bounded one-window fixture.

## Phase 2 - Runtime Integration

Status: completed

- Add an explicit Dataset runtime path behind a compatible configuration flag.
- Keep the legacy path available for parity and rollback.
- Put reusable loading behavior in `data/`, not the CLI.

## Phase 3 - Parity Evidence

Status: completed

- Compare row ownership, dates, codes, feature/label values, masks, sampling,
  model predictions, and artifact metadata on one rolling window.
- Fail closed on any unexplained difference.

## Phase 4 - Verification And Documentation

Status: completed

- Run focused and full regression tests.
- Update architecture, Qlib plan/alignment matrix, development log, and a
  dated migration report with measured status.
- Do not promote candidates or change the formal baseline.

## Constraints

- Selection uses only 2024 Val and 2025 Test; 2026 remains Forward-only.
- Preserve low-memory behavior for the 16 GB workstation.
- No training hyperparameter, label, feature, checkpoint, or ledger change.
- No deletion of legacy trainers until end-to-end parity and migration pass.

## Errors

| Error | Attempt | Resolution |
|---|---:|---|
| New parity config inherited legacy `research_end=2026-05-18` and was rejected by the current selection boundary | 1 | Set effective research end to 2025-12-31; retain cache superset as physical coverage only |
| Formal scope recorded configured 2010-01-01 before the first physical date 2010-01-04, creating a reversed warm-up range | 1 | Clamp recorded train/warm-up start to cache physical coverage; training indices remain unchanged |
| Initial foreground wait timed out while the legacy process continued normally | 1 | Detected the live PID and waited for the existing process instead of launching a duplicate |
