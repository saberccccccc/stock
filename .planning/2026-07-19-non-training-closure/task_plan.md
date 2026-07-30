# Non-Training Research Closure

## Goal

Execute `NON_TRAINING_RESEARCH_CLOSURE_PLAN_20260719.md` without starting,
resuming, or tuning any model training job.

## Phases

- [x] Define the complete non-training work package and align it with the
  master P0-P9 roadmap.
- [x] Consolidate plan authority, label superseded documents, and establish one
  machine-readable active-plan index.
- [x] NT0: make the accepted repository state recoverable from the remote.
- [x] NT1: freeze the formal baseline contract and artifact inventory.
- [x] NT2: audit research/forward data views, PIT inputs, and execution coverage.
- [ ] NT3: replay the formal baseline through the fixed 24-cell contract.
- [ ] NT4: qualify and fairly replay existing candidates only.
- [ ] NT5: complete APM, risk, portfolio, and execution attribution.
- [ ] NT6: optimize backtest runtime with strict behavioral parity.
- [ ] NT7: run prepared-only Shadow replay and failure drills.
- [ ] NT8: close Registry, reports, documentation, archive, and tests.
- [ ] NT9: issue one GO/HOLD/STOP decision and a pre-registered next-study draft.

Current implementation unit: NT6 MD8 clean ledger evidence. MD0-MD7 are
complete, including exact 24-cell CSV/monthly ledger equivalence and governed
Workflow/Shadow call-site consolidation. The isolated full-scale incremental
benchmark now passes daily commit, affected-month refresh, warm-cache,
partition-coverage, I/O and integrity gates. The migrated Parquet store remains
a non-authoritative candidate until the clean 24-cell I/O matrix and MD9
Val/Test/Forward dual-read evidence complete.
The v2 backend policy freezes candidate store/cache paths, and promotion
requires the current active manifest plus every full dual-read identity to
match those paths exactly.
NT3 dry-run remains the parity oracle and resumes only after NT6 performance
work passes its behavior-equivalence gate.

## Constraints

- No model training, continuation, tuning, loss search, or checkpoint selection.
- Val 2024 and Test 2025 select; Forward 2026 only observes.
- The formal baseline remains `ledger_path_v3_t0001_nolookahead` unless a later
  fully governed promotion changes Registry.
- Realistic open-price share-ledger, 50w/100w, and four fixed stresses are
  mandatory.
- Lifecycle must remain `prepared` during this plan.
