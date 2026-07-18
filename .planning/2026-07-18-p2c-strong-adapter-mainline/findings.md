# Findings

- Existing strong rolling artifacts prove the legacy staged trainer can execute
  and resume three windows, but the Workflow adapter name currently compiles to
  a subprocess command; it does not call `TorchStrongAlphaAdapter.fit()`.
- The generic scalar-label Dataset was insufficient for the strong model.
  Required fields are `y_seq`, `raw_y_seq`, `lag1_y_seq`, `lag1_mask`, `risk`,
  and `industry_ids` in addition to `X` and scalar `y`.
- The new strong Dataset view matches all eight relevant fields on six real v14
  dates. The apparent raw-return mismatch was identical NaN placement, with
  zero finite-value difference.
- The existing map-style `PrecomputedMemmapDataset`, shuffled DataLoader, and
  staged optimizer behavior must be preserved when binding the trainer; a
  naive iterable replacement would change training order and is unacceptable.
