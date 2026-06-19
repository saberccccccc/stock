# Test Suite Index 2026-06-19

This index classifies every Python test directly under `tests/`. It records
which subsystem each test protects and prevents experimental tests from being
committed separately from the implementation they exercise.

## Alpha And Signal Tests

| Test | Purpose | Decision |
|---|---|---|
| `tests/test_alpha_io.py` | Alpha JSONL loading, validation, and writing. | Maintain. |
| `tests/test_alpha_transforms.py` | Shared alpha transform behavior. | Maintain. |
| `tests/test_signal_blend.py` | Saved-alpha blend behavior. | Maintain. |
| `tests/test_transform_alpha_for_execution.py` | Backward-compatible execution wrapper and combined transforms. | Add to maintained suite. |

## Backtest And Execution Tests

| Test | Purpose | Decision |
|---|---|---|
| `tests/test_backtest_presets.py` | Named backtest and stress presets. | Maintain. |
| `tests/test_execution_constraints.py` | Legacy constrained-execution rules. | Maintain. |
| `tests/test_open_ledger_execution.py` | Cash/share/lot/cost/open-price ledger behavior. | Maintain. |
| `tests/test_open_ledger_preset_cli.py` | Thin open-ledger CLI compatibility. | Maintain. |
| `tests/test_open_reranker.py` | Open-reranker OOF label cutoff, tail safety, and forward boundary. | Maintain. |

## Data, Training, And Metrics Tests

| Test | Purpose | Decision |
|---|---|---|
| `tests/test_config.py` | Core configuration behavior. | Maintain. |
| `tests/test_pipeline.py` | Dataset pipeline behavior. | Maintain. |
| `tests/test_precomputed_memmap_dataset.py` | Memmap dataset and optional raw/lag1 label exposure. | Maintain. |
| `tests/test_train_batch_config.py` | Training CLI batch/config/time-split behavior. | Maintain. |
| `tests/test_metrics.py` | Validation metrics. | Maintain. |
| `tests/test_fundamental_update.py` | Point-in-time fundamental update behavior. | Maintain. |
| `tests/test_training_presets.py` | Registered training presets. | Maintain. |
| `tests/test_downside_loss.py` | Downside loss and independent lag1-loss activation semantics. | Maintain; experimental losses remain disabled by default. |

## Reranker Tests

| Test | Purpose | Decision |
|---|---|---|
| `tests/test_reranker_dataset.py` | Candidate percentile rank and graded relevance labels. | Maintain. |
| `tests/test_reranker_imports.py` | Import compatibility across the V1-V4 historical reranker script family. | Maintain. |

## Experiment And Cleanup Infrastructure Tests

| Test | Purpose | Decision |
|---|---|---|
| `tests/test_archive_plan.py` | Archive classification and move-plan safety. | Maintain. |
| `tests/test_checkpoint_reference_audit.py` | Checkpoint reference decisions. | Maintain. |
| `tests/test_checkpoint_selection.py` | Portfolio-aware checkpoint selection. | Maintain. |
| `tests/test_experiment_leaderboard.py` | Candidate registry and leaderboard grouping. | Maintain. |
| `tests/test_review_docs.py` | Review-document indexing. | Maintain. |
| `tests/test_run_script_index.py` | Complete classification of `run/*.py`. | Maintain. |
| `tests/test_source_inventory.py` | Top-level source inventory behavior. | Maintain. |
| `tests/test_test_suite_index.py` | Complete classification of `tests/test_*.py`. | Maintain. |

## Rules

1. Every new `tests/test_*.py` file must be added to this index.
2. A test for uncommitted experimental implementation stays with that
   implementation bundle; do not commit it alone.
3. Compatibility-wrapper tests remain even when shared modules have deeper
   unit tests because they protect old CLI/import contracts.
4. Moving a legacy implementation requires updating or retiring its test in
   the same commit.
