# Progress

## 2026-07-19

- Confirmed both linked worktrees and branch ancestry.
- Confirmed 145.72 GiB free on C: and no active project training process.
- Started legacy preservation and standalone-repository migration.
- Copied the unique stability report and V9/GAT diagnostic into
  `archive/legacy_optimized_20260719/` and verified matching SHA-256 hashes.
- Preserved all 17 legacy paths in isolated branch
  `legacy/optimized-pre-model-exp-20260719` at commit `c91eb2a`.
- Fast-forwarded `master` to accepted commit `382d928`; `master` and
  `model-experiments` are identical before the migration-record commit.
- Recorded the retirement plan and archive ignore policy at `f19b632` and
  `c081ba8`, then fast-forwarded both accepted branches to that state.
- Installed a no-hardlink standalone `.git` directory in
  `deepseek_model_exp`; its common Git directory now lives inside the project.
- Restored the original remote and local Git identity, retained the legacy
  snapshot at `c91eb2a`, and verified repository objects with `git fsck`.
- Passed the full independent-repository regression suite: 547 passed with one
  existing pandas FutureWarning.
- Removed the temporary clone, obsolete linked-worktree pointer, and the
  2.98 GiB `deepseek_optimized` directory after all safety gates passed.
