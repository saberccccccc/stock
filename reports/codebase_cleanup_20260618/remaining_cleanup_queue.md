# Remaining Cleanup Queue 2026-06-19

This queue starts after the legacy `backtest_results_*` cleanup was completed.
It summarizes what remains in `archive_plan.csv` and sets the next safe order.

For navigation across the cleaned workspace, start with:

```text
reports/codebase_cleanup_20260618/project_entrypoints_index.md
```

## Current Candidate Counts

| Class | Count |
|---|---:|
| `experiment_output` | 36 |
| `checkpoint_or_model` | 30 |
| `archive_or_cache` | 0 |

After the no-reference checkpoint/model archive batch, checkpoint/model
candidates are down to 30. The four remaining old archive/cache candidates were
moved to `archive/cache_202606/`; regenerated `.pytest_cache` and `__pycache__`
are now skipped by source inventory because they are volatile test/runtime
caches.

## Remaining Experiment Outputs

| Group | Count | Handling |
|---|---:|---|
| Training/candidate/loss validation | 13 | Build or update a training-validation ledger first. These directories still explain M0/A0/A4/loss/lag1 decisions. |
| Reranker artifacts | 8 | Use the reranker artifact ledger before moving; keep anything needed for OOF reconstruction. |
| Open-reranker attack candidates | 7 | Use `open_reranker_market_overlay_ledger.md`; keep until forward/attack-candidate evidence is frozen. |
| Market overlay experiments | 4 | Use `open_reranker_market_overlay_ledger.md` and `rejected_candidate_summary.md`; rejected overlays need path-reference checks before archive. |
| V9 strategy evidence | 3 | Keep unless superseded by protected official baseline and report summaries. |
| Other | 1 | Inspect manually before moving. |

The legacy `backtest_results_*` group is complete:

```text
remaining backtest_results_* candidates=0
top-level backtest_results_* entries=0
```

## Remaining Checkpoint/Model Outputs

| Group | Count | Handling |
|---|---:|---|
| Loss-ablation checkpoints | 15 | Do not move until the loss-ablation decision ledger maps winners, rejected runs, and protected baselines. |
| Alpha checkpoints | 8 | Highest risk; preserve current or historically referenced checkpoints until checkpoint references are audited. |
| Reranker checkpoints | 6 | Pair with reranker artifact ledger; archive only after OOF/model reproducibility is documented. |
| Other model outputs | 5 | Inspect individually: benchmarks, smoke rawmetrics, LightGBM/switch-value model directories. |

Reference audit:

```text
reports/codebase_cleanup_20260618/checkpoint_reference_audit.md
```

Current audit decision counts:

```text
keep=3
hold=30
archive_after_matching_artifact_ledger=0
```

The four no-reference benchmark/smoke/switch-value model directories have been
archived to `archive/checkpoints_202606/`. Remaining checkpoint/model candidates
should stay in place until a category-specific decision ledger says otherwise.

## Current Hold Policy

The experiment, training, reranker, and checkpoint ledgers now exist and the
checkpoint reference audit has been regenerated. Its current decisions are:

```text
hold=30
keep=3
```

Therefore no checkpoint/model batch is eligible for automatic movement. Leave
official V9/open-ledger/cutoff evidence protected. Revisit exact rejected
artifact directories only when their protected references are retired or a
new baseline supersedes them.

## Safety Rules

1. Use exact `--name-prefix` or `--name-glob`; avoid broad `--class` moves.
2. Dry-run before every move.
3. Regenerate source inventory and archive plan after every move.
4. Run focused cleanup tests after every move:

```text
C:\Users\x\miniconda3\envs\torch\python.exe -m pytest `
  tests\test_archive_plan.py `
  tests\test_source_inventory.py `
  tests\test_review_docs.py -q
```

5. Commit only code/report/index changes; do not commit archive payloads.

## Working-Tree Policy

Registered root-level checkpoint and experiment-output families are ignored by
Git using anchored artifact patterns. They remain visible to
`generate_source_inventory.py` and the archive/reference audits, but no longer
obscure source and report changes in daily `git status` output.

`backtest_result_snapshots/` is the exception: it is compact replacement
evidence for already archived raw backtest directories and is versioned with
the reports.
