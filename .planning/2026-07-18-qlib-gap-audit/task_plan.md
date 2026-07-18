# Qlib Alignment Gap Audit

## Goal

Audit the current implementation against the local Qlib reference and the
project's A-share requirements. Separate genuine missing alignment, shallow or
misstated alignment, intentional non-adoption, and optional future work. Fix
high-confidence governance or framework defects found during the audit.

## Phases

### Phase 1 - Contract and source inventory

Status: completed

- Reconcile project rules, protocol, ADRs, roadmap, alignment matrix, registry,
  and current implementation.
- Inventory Qlib reference components and matching project modules/tests.

### Phase 2 - Implementation-depth audit

Status: completed

- Verify Workflow, Dataset/Processor, Model, Record, Rolling/OOF, Strategy,
  Executor, experiment tracking, and lifecycle behavior from code and tests.
- Identify stubs, duplicated paths, missing end-to-end integration, and stale
  completion claims.

### Phase 3 - A-share and governance audit

Status: completed

- Check PIT timing, universe/status coverage, adjusted-vs-raw prices, execution
  constraints, selection/Forward separation, capital/stress contracts, and
  immutable experiment lineage.

### Phase 4 - Correct verified defects

Status: completed

- Correct high-confidence contract/document/code defects without changing
  candidate selection, Registry state, or model parameters.
- Add focused regression coverage.

### Phase 5 - Cleanup inventory

Status: completed

- Classify files and directories as retain, archive, delete-safe, or unresolved
  by tracing imports, configs, Registry references, and artifact lineage.
- Estimate reclaimable space and produce a non-destructive archive review.
- Do not delete or move artifacts during the audit.

### Phase 6 - Verification and report

Status: completed

- Run focused and full tests.
- Produce a Chinese severity-ranked audit report and update maintained plans.

## Constraints

- 2024 Val and 2025 Test are selection evidence; 2026 is observation-only.
- Do not use Forward results for tuning or promotion.
- Keep the project realistic open-price share ledger as formal executor.
- Do not change the formal baseline, Registry state, or lifecycle state.
- Preserve unrelated user changes and historical artifacts.
