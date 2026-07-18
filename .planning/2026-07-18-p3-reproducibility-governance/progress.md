# Progress

## 2026-07-18

- P2C closed with same-checkpoint byte-identical Adapter inference over one
  trained window and three reused cross-state windows.
- Started P3 as the next fixed master-plan unit. No model selection, ledger,
  Forward, Registry, or lifecycle state is being changed.
- Implemented and unit-tested the six-manifest `provenance_bundle_v1` plus an
  artifact-index completeness gate.
- Integrated provenance into the real three-window strong Adapter replay. All
  63 alpha rows remain byte-identical and all seven provenance index entries
  validate.
- Runtime evidence: 23.97 seconds, 1.71 GiB process peak RSS, 1.38 GiB minimum
  system available memory, and 81.3 MiB CUDA peak allocated memory.
- Full regression passed: 535 tests, with one pre-existing pandas FutureWarning.
  P3 is closed and the next fixed unit is the P4 Compact OOF evidence pre-audit.
