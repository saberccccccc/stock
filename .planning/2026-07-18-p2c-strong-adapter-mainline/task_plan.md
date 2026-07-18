# P2C Strong Model Adapter Mainline

Parent plan: `MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`, stage P2C.

## Goal

Run the reconstructed `multi_downside_e19` training and prediction lifecycle
through the governed Dataset and `TorchStrongAlphaAdapter` contracts without
changing its proven multi-label inputs or training behavior.

## Checklist

- [x] Re-audit the reconstructed profile, three-window staged pilot, resume,
  purge, cache binding, and PredictionFrame artifacts.
- [x] Add a strong-model Project Dataset view and prove field parity with
  `PrecomputedMemmapDataset` on synthetic and real v14 data.
- [x] Bind one concrete trainer delegate that consumes the governed Dataset
  contract while preserving the existing DataLoader shuffle, collate, staged
  resume, optimizer reset, loss weights, and checkpoint semantics. Independent
  CUDA retraining is not byte deterministic, so inference parity is measured
  with the same checkpoint.
- [x] Run one immutable one-window Adapter smoke under CUDA/memory guards.
- [x] Compare checkpoint, dated score, sample coverage, timing, and memory with
  the existing staged-pilot oracle; same-checkpoint legacy/Adapter alpha is
  byte-identical over all 22 dates.
- [x] Reuse three cross-state staged checkpoints through the accepted Adapter
  path; all 63 dated alpha rows are byte-identical, closing the P2 engineering
  gate.
- [ ] Hand formal checkpoint/profile choice to P4 complete 2024/2025 OOF ledger
  evaluation under ADR 0009.

## Stop Rules

- Do not start the 24-window strong run before the one-window Adapter gate.
- Do not silently replace multi-label fields with scalar labels.
- Do not select from three pilot months or from Forward 2026.
- If Adapter substitution changes samples or checkpoint behavior, retain the
  current staged runner as oracle and resolve the mismatch first.
