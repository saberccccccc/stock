# Progress

- Read governance, architecture, research protocol, Registry contracts, and
  planning/graph skills.
- Started Phase 1 source and data-contract tracing.
- Traced the exact legacy sampling rule and identified why direct use of the
  existing LightGBM adapter would change results.
- Defined the first migration boundary as sample-source parity only.
- Added `build_v14_rolling_dataset` and integrated an explicit
  `data.dataset_runtime` selector into the formal LightGBM runner.
- Added exact synthetic parity coverage for daily rows, bounded sampling,
  predictions, index validation, and fail-closed runtime selection.
- Focused regression: 15 passed.
- First real legacy run stopped before artifact creation because its new config
  copied the obsolete cache ceiling into `research_end`. Corrected both parity
  configs to the canonical 2025-12-31 effective boundary.
- The next formal preflight exposed and fixed a reversed feature-warmup range
  caused by configured calendar start preceding physical trading coverage.
- Completed both real one-window formal parity runs. Counts, sampled rows,
  best iteration, model hash, raw alpha hash, and stitched alpha hash match.
- Added a zero-processor array-copy fast path to address the measured 23%
  Dataset runtime overhead; optimized real rerun remains to run.
- Optimized real rerun completed with all three economic artifact hashes still
  identical; runtime improved to 307 seconds.
- Wrote the Chinese parity report and machine summary; updated architecture,
  Qlib alignment matrix/plan, and development log.
- Updated README with the explicit runtime selector, accepted parity, and
  deferred default switch.
- Validated all three formal manifests and economic artifact hashes.
- Full regression: 524 passed, 1 existing pandas FutureWarning.
