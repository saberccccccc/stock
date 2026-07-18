# Findings

## Initial findings

- `PROJECT_RULES.md` and ADR 0001 still state a hard research end of
  2026-05-18, while the canonical runtime protocol uses 2024 Val, 2025 Test,
  and full-year 2026 Forward observation. This is a governance contradiction.
- Q7B historical deterministic replay is implemented and accepted, while the
  lifecycle intentionally remains `prepared` pending manual activation.

## Evidence log

Further source-grounded findings will be appended during the audit.

## Verified findings

- ADR 0001 and `PROJECT_RULES.md` contradicted the canonical protocol by using
  2026-05-18 as the Forward boundary. ADR 0007 now records 2024 Val, 2025 Test,
  and all-2026 Forward; 2026-05-18 is legacy cache provenance only.
- Dataset/DataHandler and Model adapter implementations are real and tested,
  but formal LightGBM and strong-model runners bypass them. Q2/Q3 are partial
  mainline migrations, so legacy trainers cannot yet be removed.
- Rolling `run_mode=formal` meant executed instead of dry-run and did not prove
  governance-formal status. OOF evidence now labels that distinction and
  validates a governance-formal parent when one is claimed.
- The generic Recorder lacks a complete package/hardware environment freeze
  and cannot fully reproduce dirty source from Git commit plus path-only status.
- Project-native TopK/dropout and realistic open ledger are genuinely
  integrated. Qlib Executor remains an intentional non-adoption.
- Unified alpha-risk-cost portfolio construction and dynamic as-of rolling
  ensemble remain missing. Historical ST completeness remains externally
  blocked.
- Disk use is artifact-heavy: cache 75.456 GiB, reports 14.592 GiB, archive
  5.609 GiB, and mixed legacy root outputs 21.698 GiB. Rebuildable inference
  matrices alone occupy 11.117 GiB.
- The two largest v14 cache families are `fundaq` and `funda` contracts, not
  byte-equivalent duplicates. Current model lineage and formal evidence paths
  must remain protected.
