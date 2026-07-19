# Planning Index

This directory stores execution ledgers, not competing project roadmaps.
`MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md` is the sole project sequence,
and `.active_plan` is the sole pointer to current work.

## Current

- Active: `2026-07-19-non-training-closure`
- Detailed specification: `NON_TRAINING_RESEARCH_CLOSURE_PLAN_20260719.md`
- Training status: paused; this active plan must not start or resume training.

## Plan Status

| Plan directory | Status | Authority |
|---|---|---|
| `2026-07-19-non-training-closure` | active | current execution ledger |
| `2026-07-18-p4-monthly-oof` | paused | future P4 technical evidence |
| `2026-07-18-p2c-strong-adapter-mainline` | transferred | remaining decision belongs to P4 |
| `2026-07-18-p3-reproducibility-governance` | complete | historical evidence |
| `2026-07-18-lgbm-dataset-mainline` | closed | historical implementation evidence |
| `2026-07-18-qlib-gap-audit` | complete | historical audit evidence |
| `2026-07-12-qlib-adoption` | superseded | historical Qlib implementation ledger |
| `2026-07-18-master-execution-roadmap` | complete | master-plan creation evidence |
| `2026-07-18-model-experiments-commit-cleanup` | complete | Git cleanup evidence |
| `2026-07-19-legacy-worktree-retirement` | complete | repository migration evidence |

## Rules

1. Add progress only to the active plan.
2. Do not create a new plan for a subtask already covered by the active plan.
3. A new plan requires a distinct stage boundary, protocol, owner, deliverable,
   and acceptance gate.
4. Completed and superseded directories remain immutable audit evidence.
5. Change `.active_plan` and this index together.
