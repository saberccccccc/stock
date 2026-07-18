# Findings

- Q2 Dataset/DataHandler interfaces exist but the formal LightGBM rolling CLI
  still owns its legacy sample path.
- No `graphify-out/graph.json` exists, so this migration uses direct targeted
  source tracing rather than a stale or newly built whole-repo graph.
- `run/rolling_lgbm_alpha.py` already creates `V14MemmapProvider`, but uses it
  only for date-bound validation and the provider manifest. Training and
  prediction still call `iter_rolling_samples` directly.
- `resolve_window_indices` owns label-tail purge. A runtime Dataset built from
  unadjusted config date ranges would reintroduce purged tail dates; the bridge
  must derive and verify each named segment from the resolved index lists.
- The existing `LightGBMModelAdapter._collect_bounded` takes the first N rows.
  The formal legacy trainer samples a fixed quota per date with RNG seed
  `seed + time_index`, then truncates. Switching to the adapter now would alter
  the training population and cannot be called parity.
- Phase 2 should migrate only the sample source: Provider -> ProjectDataset ->
  the unchanged legacy sampler/trainer. Model-adapter migration remains a
  later independently tested step.
- The bridge now requires each resolved segment to equal one contiguous slice
  of the provider calendar. This preserves the exact purged end index instead
  of reconstructing the original unpurged config interval.
- Empty processor chains intentionally preserve the already normalized v14
  values; Dataset `learn` and `infer` keys are still exercised without adding
  a second transform.
- Formal rolling scope had a latent boundary bug when a configured start date
  preceded the first physical trading date by a weekend/holiday. Scope now
  records the actual first available date; sample ownership is unchanged.
- Real one-window parity produced byte-identical LightGBM model, raw alpha, and
  stitched Val alpha. Dataset runtime was 338 seconds versus 275 seconds for
  legacy because an empty processor chain still copied every decoded array.
- The zero-processor fast path can retain mapping isolation without copying
  provider-owned arrays; non-empty chains retain defensive copies.
- Optimized Dataset rerun remained byte-identical and improved from 338 to 307
  seconds. It is still 11.6% slower than the 275-second legacy run, so a default
  switch is not justified by one window.
