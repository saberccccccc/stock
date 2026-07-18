# ADR 0008: Single Master Execution Roadmap

Date: 2026-07-18
Status: accepted

## Context

The repository accumulated a refactor blueprint, a Qlib adoption plan, a
long-term roadmap, several completed `.planning/` tasks, and historical report
recommendations. Their technical content was often valid, but each could appear
to define the next project phase. Incremental status appendices also left stale
date-boundary and priority wording visible beside newer contracts.

This created a governance failure: a locally reasonable next action could move
the project away from the intended end-to-end sequence.

## Decision

`MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md` is the sole active execution
sequence. It defines P0-P9, dependencies, gates, fallbacks, stop rules,
prohibited shortcuts, and the industrialization completion criteria.

The following remain more authoritative for their own values:

- `registry/` for formal baselines, candidates, and promotion rules;
- `RESEARCH_PROTOCOL.md` for split and research-boundary semantics;
- `PROJECT_RULES.md` for engineering governance.

Specialized plans may expand only the currently active master-plan stage. They
cannot reorder the overall sequence. A sequence change requires an ADR with the
reason, impact, and rollback plan.

## Consequences

- Progress reports must name the current master-plan phase and passed gate.
- Old roadmap and Qlib-plan next-step wording becomes historical context.
- Framework convergence, return research, and Shadow operation are separate
  gated phases.
- The current phase is P2 training-mainline convergence; no new broad tuning
  starts before its acceptance gates are resolved.
- This decision does not change any model, baseline, Registry value, ledger
  result, Forward observation, or lifecycle state.
