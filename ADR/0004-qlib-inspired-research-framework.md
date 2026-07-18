# ADR 0004: Qlib-Inspired Research Framework

- Status: accepted
- Date: 2026-07-12

## Context

The project already has a formal A-share execution and governance path:
point-in-time data, dated alpha, realistic open-price share-ledger execution,
registry evidence, attribution, and scorecard decisions. Model experiments
remain difficult to reproduce because their training window, cache contract,
model artifact, alpha, and execution evidence are not yet one immutable task
record.

The local Qlib reference demonstrates useful patterns: experiment recording,
fit-versus-infer processing contracts, purged rolling tasks, chronological OOS
prediction lineage, bounded tuning, and online model activation history. Its
default data provider, close-price strategy, and large-account assumptions do
not match this project's A-share execution protocol.

## Decision

Create a project-native research framework inspired by those Qlib patterns.

1. The framework is research and manual-shadow infrastructure only. It does
   not introduce automatic trading, automatic retraining, or automatic model
   replacement.
2. Each experiment will have an immutable manifest covering source revision,
   cache and transform contract, date boundaries, model/checkpoint, alpha,
   execution settings, artifacts, and status history.
3. Rolling tasks must use label-tail purge and produce dated OOS alpha with a
   unique owner window for each OOS date. A stitched result runs through one
   continuous existing realistic ledger; monthly account resets are prohibited.
4. Existing project-native PIT data, `oo_lag1` execution-aware labels,
   realistic A-share open ledger, and `registry/` remain authoritative. Qlib
   default close execution, provider data, TopkDropout behavior, and CNY 100m
   assumptions are not formal evidence.
5. The framework must enforce the frozen boundary: selection uses 2024
   validation and 2025 test only; 2026 is forward observation only.
6. Before the first monthly walk-forward performance experiment, complete the
   A-share acceptance gate for effective dates, historical universe, adjusted
   feature versus raw execution prices, board/ST/new-listing limits,
   suspension/zero-volume, lots, costs, ADV, and external-input timing.

## Consequences

The first implementation work is governance, artifact provenance, and task
contracts. No current candidate is promoted or demoted by this ADR. Existing
`backtest/portfolio_optimizer.py`, `backtest/risk_model.py`, open-ledger
execution, and registry scorecards are extended through adapters and tests
rather than replaced.

The first model experiment after framework completion is fixed-window monthly
walk-forward retraining. Its initial schedule is trailing four-year Train,
six-month Valid, and one-month OOS with a monthly step. Fixed three-year,
five-year, and expanding-history schedules are comparison arms only after the
initial controller passes integrity and ledger-parity checks.

## Supersedes

None.
