# Project Rules

Status: active from 2026-07-11. This is the development-governance source of truth. Formal research values remain authoritative in `registry/`.

## Project Goal

This project researches executable A-share cross-sectional Alpha signals using point-in-time data, PyTorch models, portfolio policies, and realistic open-price share-ledger execution. It targets CNY 500k and CNY 1m accounts on a 16 GB RAM / 8 GB GPU workstation.

## Required Reading And Precedence

Before non-trivial code, protocol, or formal-result changes, read:

1. `PROJECT_RULES.md`
2. `ARCHITECTURE.md`
3. `RESEARCH_PROTOCOL.md`
4. `MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`
5. `PROJECT_CURRENT_INDEX_20260710.md`
6. `registry/baselines.yaml`, `registry/candidates.csv`, `registry/decision_rules.json`
7. `DEVELOPMENT_LOG.md` and relevant `ADR/*.md`

Precedence: explicit user instruction, `registry/`, `RESEARCH_PROTOCOL.md`, this file,
`MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`, ADRs, then README/historical
plans/`CLAUDE.md`. The master plan is the sole active execution sequence;
specialized plans may expand its current stage but cannot reorder it. Historical
documents cannot override the registry, protocol, rules, or master plan.

## Quant Research Invariants

- Model and rule selection ends at `2025-12-31`: `val_2024` and
  `test_2025` are selection evidence, while all 2026 data is Forward
  observation only. `2026-05-18` is a legacy cache snapshot date, not a
  research/Forward boundary.
- Only `val_2024` and `test_2025` select models and rules.
- `forward_2026` is observation-only; never tune from it.
- Official execution is realistic open-price share-ledger.
- Required stresses: `normal`, `lag1`, `cost2x`, `capacity_3pct`.
- Required capitals: CNY 500,000 and CNY 1,000,000.
- Official reports carry signal and backtest start/end dates.
- Candidate promotion requires complete selection evidence and attribution.

## Required Runtime

Use the Torch environment for every Torch/CUDA command:

```powershell
$env:PYTHON = "$env:USERPROFILE\miniconda3\envs\torch\python.exe"
& $env:PYTHON -m pytest -q
```

Verified stack: PyTorch `2.11.0+cu128`, CUDA, RTX 5070 Laptop GPU. Base Python has no Torch. AMP stays disabled because it has produced Loss NaN.

## Development Workflow

1. Inspect required-reading files and affected modules/tests.
2. State a short architecture review: goal, ownership, dependencies, leakage/execution risks, tests, and output artifacts.
3. Implement one coherent module-level change. Do not mix unrelated cleanup, model work, and protocol changes.
4. Run focused then proportional regression tests.
5. Regenerate affected evidence through the official flow.
6. Update architecture, log, ADR, registry, and README where applicable.
7. Inspect the diff and prepare an imperative commit message.

Architecture-review output must state:

```text
Project understanding:
Affected modules and ownership:
Dependencies and data flow:
Point-in-time / execution / forward-data risks:
Test and evidence plan:
Recommended implementation order:
Architecture issue found: yes/no, with reason
```

Research pipeline:

```text
data snapshot -> alpha signal -> candidate policy -> realistic ledger
-> registry evidence -> attribution -> scorecard -> governance decision
```

## Architecture And Code Rules

- `core/` owns reusable model, dataset, loss, and configuration code.
- `data/` owns PIT inputs, labels, and cache metadata; use effective availability dates.
- `alpha/` owns signal schema/transforms, not execution accounting.
- `backtest/` owns execution, constraints, ledger accounting, and metrics.
- `run/` is thin CLI orchestration; reusable behavior belongs in its owning module.
- `registry/` is the formal source for candidates and evidence; filenames alone are not governance.
- Preserve public CLI compatibility unless an ADR authorizes a break.
- Cache immutable input-derived work only. Never cache candidate-dependent ranking, portfolio state, stress overrides, or future data.
- Put production code in the owner package, CLI orchestration in `run/`, formal metadata in `registry/`, and generated evidence in `reports/`. Do not create a new root-level script or folder when an existing owner exists.
- Name new reports as `<purpose>_<YYYYMMDD>` and include protocol/date bounds in their metadata rather than relying only on the folder name.
- Tushare token may remain in the locally approved project documentation, but do not print it in logs, test output, generated reports, or commit messages.

## Testing, Docs, Git, And Refactoring

- Bug fixes need regression tests where practical; behavior-preserving refactors need parity tests.
- Backtest changes test timing, costs, lots, ADV, limits, and no-lookahead as relevant.
- Run Torch tests with `$env:PYTHON`.
- Append material work to `DEVELOPMENT_LOG.md`; do not rewrite history.
- A module is complete only when: implementation is complete; focused tests pass; relevant README/architecture docs are updated; `DEVELOPMENT_LOG.md` is appended; registry/ADR implications are handled; and an imperative commit message is prepared.
- Add `ADR/NNNN-short-title.md` before durable protocol/execution/runtime decisions. Each ADR has Status, Date, Context, Decision, Consequences, and Supersedes (when applicable). ADRs are append-only; supersede rather than rewrite.
- Keep commits small and imperative. Never revert unrelated user changes.
- Generate a non-destructive archive review before moving or deleting artifacts.

Minimum verification commands:

```powershell
# Torch-dependent code/tests
& $env:PYTHON -m pytest -q <focused-tests>

# Non-Torch backtest utilities when appropriate
$env:PYTHONPATH = (Get-Location).Path
pytest -q <focused-tests>
```

Do not call a candidate complete merely because unit tests pass. Official
candidate work also requires registered realistic evidence, complete selection
coverage, attribution, and scorecard evaluation.

## Change History

- 2026-07-11: initial governance rules. Future changes append here or are recorded in a new ADR.
- 2026-07-18: corrected the selection/Forward boundary contract under ADR 0007;
  retained `2026-05-18` only as legacy cache metadata.
