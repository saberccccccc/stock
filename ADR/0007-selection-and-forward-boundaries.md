# ADR 0007: Selection And Forward Boundaries

- Status: accepted
- Date: 2026-07-18
- Supersedes: ADR 0001

## Context

ADR 0001 described `2026-05-18` as the end of research. That date was the
physical cutoff of an older cache and report family, not the boundary used by
the current selection protocol. Keeping both meanings made manifests and
cleanup decisions ambiguous.

## Decision

Model, checkpoint, rule, and portfolio-policy selection may use only
`val_2024` and `test_2025`, and ends on `2025-12-31`. Data dated in 2026 belongs
to `forward_2026`, is observation-only, and cannot affect tuning, selection, or
promotion.

`2026-05-18` remains valid only when it identifies the physical coverage of a
legacy cache or historical artifact. It must not be presented as the current
research/Forward boundary.

One physical data store may expose date-bounded research and Forward views.
Every formal artifact must record its effective data range, physical coverage
when relevant, and split role.

## Consequences

- Current protocol code, Registry selection rules, and documentation use the
  same 2024 Val / 2025 Test / 2026 Forward contract.
- Historical filenames and immutable manifests containing `end20260518` are
  not renamed; their date remains artifact provenance.
- A 2026 result can be monitored and reported, but cannot justify changing a
  model, threshold, ensemble weight, or portfolio rule.
