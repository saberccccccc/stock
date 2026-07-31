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
- [x] NT6: optimize backtest runtime with strict behavioral parity.
- [ ] NT7: run prepared-only Shadow replay and failure drills.
- [ ] NT8: close Registry, reports, documentation, archive, and tests.
- [ ] NT9: issue one GO/HOLD/STOP decision and a pre-registered next-study draft.

NT6 is complete. The v2 policy now activates the identity-bound monthly backend;
MD8 clean performance acceptance, Val/Test/Forward full dual-read, the 24-cell
equivalence matrix and the monthly-to-legacy-to-monthly recovery drill all
passed. CSV remains the read-only oracle and the 5,332 legacy stock files were
not deleted.

Next fixed implementation unit: resume NT3 formal-baseline replay through the
same 24-cell contract. NT6 artifacts and backend policy are frozen inputs, not a
new parameter-search surface.

## Constraints

- No model training, continuation, tuning, loss search, or checkpoint selection.
- Val 2024 and Test 2025 select; Forward 2026 only observes.
- The formal baseline remains `ledger_path_v3_t0001_nolookahead` unless a later
  fully governed promotion changes Registry.
- Realistic open-price share-ledger, 50w/100w, and four fixed stresses are
  mandatory.
- Lifecycle must remain `prepared` during this plan.
