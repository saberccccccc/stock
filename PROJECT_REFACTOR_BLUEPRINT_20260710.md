# Project Refactor Blueprint - 2026-07-10

Purpose: define a standard, auditable project flow before adding more models,
market-state features, rerankers, or portfolio policies.

This is a light refactor plan. The goal is not to rewrite the whole project.
The goal is to extract a clean research pipeline from the current experiment
sprawl, then route all future work through that pipeline.

## Required Runtime

Torch-dependent training, inference, and tests must use the project Conda
environment, not the default Python interpreter:

```powershell
$env:PYTHON = "$env:USERPROFILE\miniconda3\envs\torch\python.exe"
& $env:PYTHON -m pytest -q
```

Verified stack: PyTorch `2.11.0+cu128` with CUDA on the RTX 5070 Laptop GPU.
AMP is intentionally disabled because this project has observed Loss NaN with
mixed precision.

## 1. Why Refactor Now

The project has enough useful research output, but the evidence layer is too
hard to trust because:

- formal baseline evidence is split across multiple directories;
- proxy and realistic execution results coexist;
- some old documents still describe outdated baselines;
- many report paths encode candidate names but there is no central registry;
- manifest-based backtests rerun slow repeated OHLC loading;
- forward evidence is easy to accidentally mix into selection;
- every new idea currently creates new scripts and report folders.

Without a standard flow, adding a stronger reranker or global market-state
features will likely create more noise instead of better decisions.

## 2. Refactor Principles

1. Do not change the formal research protocol during refactor.
2. Do not delete important evidence until it is registered.
3. Keep old scripts runnable, but stop using them as official entry points.
4. Prefer small adapters around existing code over large rewrites.
5. Every official result must be reproducible from a manifest.
6. Every candidate must declare whether it is selection-eligible.
7. Forward results are observation-only.

## 3. Target Standard Pipeline

The official pipeline should look like this:

```text
data snapshot
  -> alpha signal generation
  -> candidate signal transformation / portfolio policy
  -> realistic open-ledger backtest
  -> APM scorecard
  -> attribution
  -> candidate decision
```

Each stage should write a machine-readable manifest.

## 4. Proposed Directory Layout

New official structure:

```text
project/
  configs/
    candidates/
    backtests/
    protocols/
  pipeline/
    data/
    signals/
    portfolio/
    backtest/
    audit/
    attribution/
  registry/
    candidates.csv
    reports.csv
    baselines.yaml
  reports/
    official/
    research/
    archive/
    tmp/
  scripts/
    official/
    maintenance/
```

This can be introduced gradually. Existing `run/`, `backtest/`, `alpha/`,
and `data/` modules do not need to be moved immediately.

## 5. Candidate Registry

Create a central registry file:

- `registry/candidates.csv`

Required columns:

```text
candidate_id
display_name
family
status
selection_eligible
forward_observation_only
signal_path
backtest_path
execution_mode
base_alpha
created_at
notes
```

Allowed `status` values:

```text
formal_baseline
formal_view
challenger
shadow
research
legacy
rejected
```

Current initial registry should include:

- `ledger_path_v3_t0001_nolookahead`
- `nolookahead_stateobs`
- `alpha_sa_p05`
- `cond_pairrisk_volg001_realistic`
- `cond_pairrisk_v2_svol0_betag001_gate0`
- `ledger_path_v3_capital_aware_hybrid_50_75no_100volg001`

## 6. Report Registry

Create a central report registry:

- `registry/reports.csv`

Required columns:

```text
report_id
candidate_id
path
split
stress
capital
execution_mode
signal_start
signal_end
backtest_start
backtest_end
selection_eligible
is_forward
source_script
notes
```

The audit process should read this registry first, then fall back to path
discovery only for unregistered research folders.

## 7. Baseline Registry

Create:

- `registry/baselines.yaml`

Initial content:

```yaml
formal_baseline: ledger_path_v3_t0001_nolookahead
formal_baseline_view: nolookahead_stateobs
legacy_baseline: alpha_sa_p05

paths:
  ledger_path_v3_t0001_nolookahead:
    normal: reports/state_aware_policy_training_20260704/ledger_path_v3_t0001_nolookahead_open_ledger
    stress: reports/state_aware_policy_training_20260704/stress_ledger_path_v3_t0001_nolookahead
```

## 8. Official Backtest Runner

Create a wrapper:

- `pipeline/backtest/run_official_backtest.py`

Responsibilities:

1. Read candidate registry.
2. Read requested split/stress/capital grid.
3. Prefer `run/sweep_open_price_ledger_params.py` for batch execution.
4. Use `run/backtest_retention_open_ledger.py` only for one-off debugging.
5. Write report registry rows after successful backtests.
6. Refuse to mix `proxy` and `realistic` in official scorecards.

Default official grid:

```text
splits: val_2024, test_2025, forward_2026
selection_splits: val_2024, test_2025
observation_splits: forward_2026
stresses: normal, lag1, cost2x, capacity_3pct
capitals: 500000, 1000000
execution_mode: realistic
```

## 9. Official Scorecard Runner

Create a wrapper:

- `pipeline/audit/run_apm_scorecard.py`

Responsibilities:

1. Load registered open-ledger summaries.
2. Enforce complete stress coverage for promotion candidates.
3. Enforce selection-only rule.
4. Produce:
   - long CSV;
   - candidate summary CSV;
   - coverage audit CSV;
   - markdown decision report.

Promotion rules should be read from a config file, not rewritten inside each
experiment script.

## 10. Attribution Runner

Create:

- `pipeline/attribution/run_candidate_attribution.py`

Responsibilities:

1. Compare a challenger to formal baseline.
2. Split by val/test/forward.
3. Split by normal/lag1 and capital.
4. Summarize:
   - replacement return delta;
   - pair risk delta;
   - beta delta;
   - specific vol delta;
   - ret20 delta;
   - industry transition;
   - turnover and cost effects.

## 11. Documentation Cleanup

Update README to separate:

```text
historical baseline: V9 avgw3 + maxret095
current formal portfolio-layer baseline: ledger_path_v3_t0001_nolookahead
```

Add links to:

- `RESEARCH_PROTOCOL.md`
- `PROJECT_CURRENT_INDEX_20260710.md`
- this blueprint

## 12. Cleanup Rules

Do not delete yet:

- current formal baseline evidence;
- current realistic candidate evidence;
- v14 / multi_downside / e22-e25 signals and checkpoints;
- OHLC matrix cache;
- forward raw data.

Can archive later after registry:

- old proxy-only backtests;
- old close-based reports;
- failed experimental reranker reports;
- duplicate intermediate manifests;
- long-path failed output directories.

## 13. Implementation Order

### Phase 1: Governance and Registry

1. Create `registry/`.
2. Add `baselines.yaml`.
3. Add initial `candidates.csv`.
4. Add initial `reports.csv` by scanning known official paths.
5. Update README baseline section.

### Phase 2: Batch Backtest Wrapper

1. Build wrapper around `sweep_open_price_ledger_params.py`.
2. Support `--candidate-id`, `--split`, `--stress`, `--dry-run`.
3. Write outputs under `reports/official/<candidate_id>/`.
4. Append report registry rows.

### Phase 3: Scorecard Wrapper

1. Build registry-driven APM scorecard.
2. Enforce no proxy/realistic mixing.
3. Enforce no forward selection.
4. Produce decision report.

### Phase 4: Attribution Wrapper

1. Standardize challenger-vs-baseline attribution.
2. Produce replacement and risk deltas.
3. Attach attribution path to candidate registry.

### Phase 5: Archive

1. Mark old folders as `legacy` or `archive_candidate`.
2. Move only after registry has captured key evidence.
3. Never delete raw data, active caches, current signals, or formal evidence.

## 14. Immediate Next Step

Start with Phase 1:

```text
registry/baselines.yaml
registry/candidates.csv
registry/reports.csv
```

Then regenerate one fair scorecard from the registry before any new model or
reranker work.

## 15. Implementation Status - 2026-07-10

- Phase 1 complete: baseline, candidate, and report registries are in `registry/`.
- Phase 2 complete: `run/official_backtest_from_registry.py` calls the shared-OHLC batch runner.
- Phase 3 complete: `run/scorecard_from_registry.py` reads versioned decision rules and separates forward observation.
- Phase 4 complete: `run/attribution_from_registry.py` audits registered attribution coverage before a candidate can be promoted.
- Phase 5 complete: registry-aware archive review was generated, then 162 reviewed unregistered/unreferenced report folders were moved to `archive/experiments_202606/`. No formal evidence, raw data, active cache, or current signal was deleted.

## 16. Current Next Step

`cond_pairrisk_volg001_realistic` has completed its missing 2024/2025
`cost2x` and `capacity_3pct` attribution cells. Its next action is a
governance review, not automatic promotion: it passes the selection evidence
but has weak forward attribution. New candidates should now use the same
registry, attribution, and scorecard flow.

Regenerate the current official audit with:

```text
python run/attribution_from_registry.py
python run/scorecard_from_registry.py
```
