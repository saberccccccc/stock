# Progress

## 2026-07-18

- Started a read-only branch and worktree audit.
- Confirmed the shared-worktree topology and branch ancestry.
- Counted 64 modified, 52 deleted, and 420 untracked status entries.
- Began ownership, artifact, and commit-batch classification.
- Ran the full Torch test suite: 545 passed and two entrypoint tests failed.
- Fixed official-backtest import precedence so direct script execution resolves
  the project `backtest` package instead of `run/backtest.py`; retest pending.
- Hardened the remaining formal workflow entrypoints with the same precedence
  rule; both focused entrypoint tests pass.
- Re-ran the complete suite: 547 passed with one pandas future warning.
- Staged the reviewed framework baseline only: source, tests, configs, schemas,
  registry contracts, and requirements; generated artifacts remain unstaged.
- Committed the tested framework baseline as `8f918ca` after cleaning staged
  whitespace diagnostics.
- Applied repository artifact policy and staged governance, ADR, roadmap,
  planning, and lightweight registry state as the second commit batch.
- Committed governance and registry state as `a4e6de3`.
- Verified all 52 tracked report deletions against retained archive copies;
  committed the report cleanup and corrected archive link as `406c060`.
- Verified a clean tracked/untracked worktree. Ignored local experiments,
  reports, data, caches and checkpoints remain on disk and were not deleted.
