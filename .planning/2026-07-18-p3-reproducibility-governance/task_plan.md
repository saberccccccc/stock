# P3 Reproducibility And Runtime Governance

Parent plan: `MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`, stage P3.

## Goal

Make every governed training or replay run carry enough immutable evidence to
reconstruct its environment, participating source files, data view, fitted
transform state, resolved command/config, and runtime resource behavior.

## Checklist

- [x] Define and validate one `provenance_bundle_v1` contract containing all six
  mandatory manifests.
- [x] Capture Python/package/CUDA/hardware and dependency-file hashes without
  invoking a heavyweight environment export.
- [x] Hash the actual participating source files, including dirty-worktree
  content, instead of relying only on the Git commit.
- [x] Record physical cache coverage, logical data view, universe dimensions,
  label families, PIT/transform contract, and processor state.
- [x] Freeze the complete argv and resolved experiment config.
- [x] Sample wall time, process RSS, system available memory, CPU time, and CUDA
  peak allocation while the governed runner executes.
- [x] Integrate the bundle into one real Adapter replay and require all files in
  the artifact index before closing P3.
- [x] Run the full regression suite and update the master plan/development log.

## Stop Rules

- Do not claim byte reproducibility for independent CUDA training unless
  deterministic algorithms are explicitly enabled and proven.
- Do not fingerprint the whole repository or duplicate large data caches.
- Do not allow a completed formal experiment into a leaderboard when its
  provenance bundle is missing or stale.
- Do not start P4 full monthly strong Rolling until this gate is complete.
