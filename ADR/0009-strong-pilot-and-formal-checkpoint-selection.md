# ADR 0009: Separate Strong-Model Pilot Checkpoints From Formal Selection

Date: 2026-07-18
Status: accepted

## Context

The reconstructed strong-model pilot uses `rawtopstable_h5_top0p6` internally
to save a recoverable checkpoint. Requiring an executable full-period ledger to
select that checkpoint inside P2C would create a dependency cycle: a continuous
OOF ledger exists only after P4 rolling has completed.

Selecting a rule from the three engineering pilot months would also be an
unstable substitute for complete 2024 Val and 2025 Test evidence.

## Decision

P2C may use the predeclared internal validation metric only to create pilot
checkpoints and verify training, resume, inference, and artifact contracts. Such
checkpoints are engineering artifacts and cannot be promoted.

Formal checkpoint/profile selection occurs in P4 after complete 2024/2025 OOF
signals exist. It must use the realistic open-price ledger, fixed capitals and
stress scenarios. IC and `rawtopstable` remain diagnostics, not promotion
objectives. Forward 2026 remains observation-only.

## Consequences

- P2C can finish its engineering gate without selecting from three isolated
  OOS months.
- P4 must compare the predeclared checkpoint/profile alternatives under one
  continuous OOF ledger before P5 or P7 promotion.
- No existing e19 checkpoint is promoted by this ADR.
