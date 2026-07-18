# Findings

- `deepseek_model_exp` is the `model-experiments` linked worktree of the Git
  repository whose common metadata is under `deepseek_optimized/.git`.
- The branch is 89 commits ahead of `master` with no master-only commits.
- Initial status contains 116 tracked changes and 420 untracked entries.
- Expanded untracked inventory contains about 10,800 files, dominated by
  generated CSV/JSON/JSONL/Parquet evidence; these are not source candidates.
- The initial full test run passed 545 tests and found two failures caused by
  `run/backtest.py` shadowing the `backtest` package during direct execution of
  `run/official_backtest_from_registry.py`.
- Import-priority hardening across the formal workflow entrypoints closes the
  shadowing class; the full Torch suite now passes 547 tests.
- The reviewed framework batch contains 288 source/config/schema/test files,
  about 3 MiB total, with no staged object above 1 MiB.
- Sensitive assignment scanning found only the locally approved Tushare token
  in `CLAUDE.md` and a placeholder in `README.md`; neither is in the framework
  source batch.
- Of 52 tracked report deletions, 47 have byte-identical archive copies. The
  remaining five have same-name archive versions and require final review in
  the separate cleanup commit.
- The active master execution plan remains
  `.planning/2026-07-18-master-execution-roadmap`; this cleanup is subordinate.
