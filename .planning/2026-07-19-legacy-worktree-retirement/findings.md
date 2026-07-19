# Findings

- `model-experiments` is clean and 93 commits ahead of `master`.
- The legacy worktree has 13 tracked modifications plus a historical report,
  V9/GAT diagnostic, empty test marker, and local watchlist.
- Key loss, no-lookahead breadth, update, data-dir, GAT, batch, and ensemble
  changes are already represented by newer tested implementations.
- The legacy files are retained only as historical evidence, not merge input.
- The standalone clone retained `master`, `model-experiments`, the isolated
  legacy branch, and the original GitHub remote without hard-linked objects.
- Replacing the linked-worktree metadata initially produced stat-cache-only
  modifications. Blob hashes matched the index, and refreshing the index
  removed every false modification without changing or staging content.
- The independent repository passed `git fsck` and all 547 tests before the
  old directory was removed.
