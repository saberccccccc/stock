# Cleanup Completion Audit 2026-06-19

This audit closes the gradual cleanup plan without deleting or relocating
active research evidence.

## Plan Requirements

| Area | Evidence | Result |
|---|---|---|
| Safety inventory | `source_inventory.*`, `archive_plan.*`, category ledgers | Complete |
| Alpha I/O and transforms | `alpha/io.py`, `alpha/transforms.py` | Complete |
| Market overlays and diagnostics | `alpha/market_overlays.py`, `alpha/diagnostics.py` | Complete |
| Open-ledger engine | `backtest/open_ledger.py`, `backtest/execution.py` | Complete |
| Official presets and stress cases | `backtest/presets.py`, `backtest/stress.py` | Complete |
| Candidate registry and leaderboard | `experiments/registry.py`, `experiments/leaderboard.py` | Complete |
| Training presets and checkpoint selection | `core/training_presets.py`, `experiments/checkpoint_selection.py` | Complete |
| Compatibility entrypoints | all 93 `run/*.py` files indexed and tracked | Complete |
| Test ownership | all 33 `tests/test_*.py` files indexed and tracked | Complete |
| Research plans and reports | root plans, decision reports, compact snapshots tracked | Complete |
| Local artifact handling | root-anchored ignores plus generated inventory/audits | Complete |

## Verification

```text
full pytest suite: 160 passed
target modules: 12/12 present
run scripts: 93/93 tracked
test files: 33/33 tracked
PowerShell repro parse errors: 0
all tracked PowerShell scripts free of user-specific absolute paths
active training/backtest Python processes: 0
git diff --check: passed
git status: clean
```

## Preserved Boundaries

- Official baseline remains V9 avgw3 + maxret095 + open-price share-ledger.
- Research data remains frozen through 2026-05-18.
- Forward-only observations begin on 2026-05-19 and require explicit mode.
- No loss, reranker, or overlay was promoted during cleanup.
- Historical CLI import paths remain compatible where wrappers were thinned.

## Follow-up Audit

A second independent pass found and corrected one portability gap that the
original six-script repro audit did not cover: `scripts/run_full_training.ps1`
still contained a user-specific Python path. The portability test now covers
all tracked PowerShell scripts. Active setup documentation now uses portable
interpreter discovery and distinguishes the official open-price share-ledger
baseline from legacy close-based return diagnostics.

## Remaining Local Artifacts

Large checkpoint and experiment directories remain on disk by design. The
checkpoint reference audit records `hold=30` and `keep=3`; no checkpoint has a
safe automatic archive decision. Rejected overlay directories are still
referenced by protected result configurations.

These are retained research state, not unfinished source cleanup. Any future
move must use an exact-name dry run, update references, regenerate all cleanup
indexes, and rerun the focused cleanup tests.
