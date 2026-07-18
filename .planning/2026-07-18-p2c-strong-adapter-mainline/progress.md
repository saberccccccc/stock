# Progress

## 2026-07-18

- Completed strong Dataset field implementation and synthetic/real parity.
- Recorded ADR 0009 for engineering pilot versus formal checkpoint selection.
- Full regression passed: 528 tests, one pre-existing pandas FutureWarning.
- Next fixed implementation unit is the concrete strong trainer delegate; no
  training or backtest should start until its contract tests pass.
- Added `ExistingStrongTrainerDelegate` and the immutable one-window Adapter
  acceptance runner. The real CUDA run completed in 325 seconds with 963 Train,
  115 Valid, and 22 label-free OOS cross sections.
- Same-checkpoint legacy inference and Adapter inference are byte-identical.
  Independent CUDA reruns retain identical validation metrics but are not
  tensor-byte deterministic; that distinction is now explicit in the report.
- Replayed the three completed cross-state staged checkpoints through the
  Adapter. All 63 OOS alpha rows are byte-identical to the oracle.
- P2C engineering gate is complete. Formal checkpoint/profile choice remains
  deferred to P4 complete 2024/2025 OOF ledger evaluation.
