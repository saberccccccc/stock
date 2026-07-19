# Legacy Worktree Retirement

## Goal

Preserve the remaining legacy worktree evidence, fast-forward `master` to the
accepted `model-experiments` history, make `deepseek_model_exp` an independent
Git repository, and retire `deepseek_optimized` without losing history or local
research artifacts.

## Phases

- [x] Archive unique legacy files and commit the dirty legacy state on an
  isolated non-merge branch.
- [x] Fast-forward `master` to `model-experiments`.
- [ ] Create a no-hardlink standalone Git clone and install its metadata in
  `deepseek_model_exp`.
- [ ] Verify branches, objects, remote, status, and tests from the standalone
  repository.
- [ ] Remove the retired legacy worktree only after all gates pass.

## Safety Rules

- Never merge the legacy snapshot into `master` or `model-experiments`.
- Do not delete ignored data before the standalone repository passes `fsck`.
- Preserve the original GitHub remote URL.
- Verify every recursive-delete target resolves under
  `C:\Users\x\code\stock_prediction`.
