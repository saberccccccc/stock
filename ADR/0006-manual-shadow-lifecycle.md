# ADR 0006: Manual Shadow Lifecycle

- Status: Accepted
- Date: 2026-07-17

## Context

The historical Phase 6 manifest freezes one observation-only package, but it
does not provide a reusable model/policy version lifecycle, validated state
transitions, an event hash chain, or dated observation ownership. Qlib's
OnlineManager demonstrates the value of explicit online model history, but its
automatic online orchestration is not appropriate before this project has a
governed profitable candidate and real operational controls.

## Decision

Use a project-native, manual Shadow lifecycle after a complete standard Record
bundle exists.

- Allowed states are `prepared`, `shadow`, `paused`, and `retired`.
- Every transition requires an explicit human actor, reason, and approval.
- `retired` is terminal.
- There is no automatic `active` or trading transition.
- Model, Workflow, and six-Record artifacts are frozen by SHA-256.
- Lifecycle events are append-only and hash chained; the state file is only an
  atomic projection of the latest valid event.
- Dated Shadow observations are unique, hash their artifacts, and are always
  marked nonselecting.
- Automatic trading, retraining, promotion, and Forward selection remain
  disabled.

## Consequences

The project gains reproducible model-version history, pause/retire semantics,
and daily observation provenance without importing Qlib's Executor or enabling
automatic deployment. A candidate cannot enter Shadow from summary-only
evidence. The first real lifecycle remains blocked until a new formal Workflow
produces a complete six-Record bundle.
