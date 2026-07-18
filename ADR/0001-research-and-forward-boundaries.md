# ADR 0001: Research And Forward Boundaries

- Status: accepted
- Date: 2026-07-11

## Context

Research and forward observations require an irreversible separation to avoid
look-ahead model selection.

## Decision

Research ends at 2026-05-18. `data/raw` is frozen research data and `data/forward_raw` is forward data. Only 2024 validation and 2025 test select models/rules; 2026 forward is observation-only. All official results retain date bounds.

## Consequences

No script may silently merge forward observations into research data or use
forward results to choose a checkpoint, parameter, filter, or candidate.
