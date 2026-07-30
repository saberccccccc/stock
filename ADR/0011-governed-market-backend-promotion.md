# ADR 0011: Governed Market Backend Promotion

- Status: accepted
- Date: 2026-07-31
- Supersedes: ADR 0010 default-switch and rollback clauses only

## Context

ADR 0010 established the content-addressed daily store, storage-neutral
Provider and month-sharded execution cache. It described CSV as both parity
oracle and immediate rollback backend before the final execution contract had
three distinct modes.

The implemented contract now has separate responsibilities:

- `legacy`: current formal backend and rollback target;
- `csv`: direct six-field parity and dual-read oracle;
- `monthly`: candidate backed by the content-addressed store and monthly cache.

A backend name alone is insufficient governance. Evidence can accidentally
refer to another store generation or cache root unless candidate paths and
immutable identities are frozen and re-verified at promotion time.

## Decision

Use `configs/execution_market_backend_policy.json` schema v2 as the sole atomic
default, candidate and rollback pointer. It must freeze:

- active, candidate, shadow and rollback backend roles;
- candidate store and monthly-cache roots;
- required Val 2024, Test 2025 and Forward 2026 dual-read evidence;
- MD6 parity, MD7 call-site and MD8 performance evidence paths.

Promotion from `legacy` to `monthly` is manual and fail-closed. It requires:

1. exact 24-cell ledger parity;
2. governed call-site audit;
3. clean MD8 performance acceptance with source hashes;
4. full Val/Test/Forward monthly-versus-CSV dual-read;
5. current candidate manifest equality with incremental evidence;
6. identical candidate store/cache identities in every dual-read report.

The promotion record freezes actor, reason, UTC time, evidence hashes,
candidate paths and active manifest hash. Transitions are append-only.

Rollback is allowed only from `monthly` to `legacy`. A complete recovery drill
must then restore `monthly` using the same promotion gates. CSV remains the
read-only six-field oracle; it is not the formal rollback backend.

`run/close_nt6_market_backend.py` may generate and audit evidence but must stop
at `ready_for_manual_promotion`. It must never switch the active backend.

## Consequences

- Candidate authority is assigned by policy plus immutable manifest identity,
  not by a directory name.
- Updating the candidate store after acceptance invalidates promotion until
  evidence is regenerated.
- A dual-read report from another store or cache cannot satisfy the gate.
- Historical monthly Workflow and Shadow artifacts must retain explicit frozen
  paths; missing identities fail closed rather than adopting current defaults.
- `legacy` remains active until all MD8-MD9 evidence passes and a human approves
  promotion.
- CSV remains available for parity diagnosis and is not deleted without the
  separate archive review required by ADR 0010.
