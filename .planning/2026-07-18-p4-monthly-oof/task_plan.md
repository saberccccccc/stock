# P4 Monthly Rolling And OOF

Parent plan: `MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`, stage P4.

Detailed execution plan: `P4_STRONG_MONTHLY_ROLLING_EXECUTION_PLAN_20260718.md`.

## Goal

Use complete 2024 Val and 2025 Test monthly OOS predictions plus one continuous
realistic ledger to compare frozen and monthly-retrained model families without
using 2026 Forward for selection.

## Checklist

- [x] Audit the existing 24-window Compact LightGBM experiment against coverage,
  unique OOS ownership, artifact hashes, ledger cells, Records, and P3
  provenance.
- [x] Decide whether Compact needs a rerun: retain it as historical engineering
  evidence and a rejected performance candidate; do not rerun only to backfill
  unavailable runtime metrics.
- [x] Upgrade the full strong rolling controller to emit P3 provenance while
  preserving staged resume and exact/selected transition semantics.
- [x] Produce a dry-run contract and compute/storage estimate for all 24 strong
  windows before launching GPU work.
- [x] P4-B: route all three staged trainer calls through the accepted
  `TorchStrongAlphaAdapter` delegate; retain the legacy subprocess as parity
  oracle and emergency fallback only.
- [x] P4-B: add tests for exact e6 -> e15 -> e19 transitions, optimizer reset,
  resume, hash drift, epoch/input-width validation, and Adapter/oracle parity.
- [x] P4-C: run one immutable staged window and perform an interruption/resume
  drill; validate PredictionFrame, artifact hashes, provenance and process-tree
  resource metrics.
- [x] P4-R: harden profile evidence, stage semantics, checkpoint-selection
  diagnostics, and the distinction between the current monthly audit arm and
  a future production retraining frequency. No GPU training is allowed during
  this gate.
- [ ] P4-D: after P4-R, decide whether to resume the immutable
  `multi_downside_e19` 24-window monthly arm. It is currently user-paused after
  `oos_2024_02/base_e6/epoch_005`; no concurrent strong windows are allowed on
  the 16 GiB host.
- [ ] P4-E: validate one owner per OOS date, 242 Val + 243 Test dates, hashes,
  terminal completion, lineage, and stitched exact/selected PredictionFrames.
- [ ] P4-F: run the same realistic open ledger for 50w/100w and
  normal/lag1/cost2x/capacity_3pct on Val 2024 and Test 2025.
- [ ] P4-G: compare frozen e19, monthly exact e19, monthly internally selected
  e19, Compact, and formal baseline using ledger results; IC/rawtopstable remain
  diagnostics only.
- [ ] P4-G: generate standard Records, attribution, Registry scorecard and a
  Chinese decision report with explicit evidence paths.
- [ ] P4-H: freeze the selected research contract first, then observe all
  available 2026 Forward dates without using them for selection.

## Stop Rules

- Do not rerun rejected Compact merely to manufacture complete modern metadata.
- Do not launch 24 strong windows unless resume, disk estimate, provenance, and
  memory guards pass dry-run validation.
- Do not select exact versus internally selected checkpoint from the three pilot
  months.
- Do not register or shadow any candidate before complete 2024/2025 ledger
  evidence and standard Records exist.
