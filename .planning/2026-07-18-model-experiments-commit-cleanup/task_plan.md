# Model Experiments Branch Commit Cleanup

## Goal

Audit the accumulated `model-experiments` working tree, exclude generated or
unsafe artifacts, verify coherent change groups, and commit the retained work
in reviewable batches without changing the active master execution sequence.

## Phases

- [x] Inventory every tracked and untracked change by ownership and purpose.
- [x] Audit secrets, large files, generated outputs, deletions, and ignored
  artifact policy.
- [x] Define commit batches and run focused/proportional verification for each.
- [x] Stage and commit only reviewed paths with imperative messages.
- [x] Verify branch history, residual working-tree state, and document any
  intentionally uncommitted artifacts.

## Constraints

- Do not use `git add -A` or broad wildcard staging.
- Do not revert or overwrite accumulated user work.
- Do not commit caches, checkpoints, raw data, temporary reports, or credentials.
- Preserve the active master-plan pointer and all research/Forward boundaries.
- A residual dirty tree is acceptable only when every remaining path is
  explicitly classified and documented.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Text scan traversed binary experiment artifacts | 1 | Stopped it and restricted scanning to source-like files below 2 MiB. |
| Full suite found script/package name shadowing | 1 | Reordered project-root imports in formal workflow entrypoints; 547 tests pass. |
