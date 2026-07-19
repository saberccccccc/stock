# Registry Usage

This directory is the source of truth for the refactored research flow.

## Files

- `baselines.yaml`: formal baseline, split, stress, capital, and execution-mode rules.
- `baseline_contract.json`: immutable formal-baseline identity and replay contract.
- `evidence_lineage.json`: canonical and superseded baseline replay lineage.
- `candidates.csv`: candidate metadata and signal/backtest paths.
- `reports.csv`: registered backtest evidence rows.
- `decision_rules.json`: versioned baseline-comparison and promotion rules.
- `attributions.csv`: registered candidate-vs-baseline attribution evidence.

`reports.csv` uses `canonical_evidence` to control scorecard consumption. A
blank value remains backward-compatible for ordinary candidates, `true` is
canonical evidence, and `false` is retained audit history. `superseded_by`
points from an obsolete baseline row to the canonical experiment manifest.

## Freeze The Formal Baseline

```powershell
python run/freeze_formal_baseline.py --apply-registry-lineage
```

The command validates the canonical experiment, verifies every referenced
artifact, freezes the baseline contract and evidence lineage, and marks
duplicate historical baseline rows as superseded. It does not train a model or
run a backtest. Run it only when intentionally changing baseline evidence.

## Build A Registry Scorecard

```powershell
python run/scorecard_from_registry.py
```

Output:

- `reports/official_registry_scorecard_20260710/registry_apm_scorecard.md`
- `reports/official_registry_scorecard_20260710/registry_scorecard_long.csv`
- `reports/official_registry_scorecard_20260710/registry_selection_summary.csv`
- `reports/official_registry_scorecard_20260710/registry_forward_summary.csv`
- `reports/official_registry_scorecard_20260710/registry_coverage.csv`
- `reports/official_registry_scorecard_20260710/registry_decisions.csv`

## Dry-Run An Official Batch Backtest

```powershell
python run/official_backtest_from_registry.py `
  --candidate-id cond_pairrisk_volg001_realistic `
  --candidate-id cond_pairrisk_v2_svol0_betag001_gate0 `
  --split val_2024 `
  --dry-run
```

## Run A Small Smoke Backtest

```powershell
python run/official_backtest_from_registry.py `
  --candidate-id cond_pairrisk_volg001_realistic `
  --candidate-id cond_pairrisk_v2_svol0_betag001_gate0 `
  --split val_2024 `
  --stresses normal `
  --run-id smoke_phase2_20260710
```

The wrapper calls `run/sweep_open_price_ledger_params.py`, then materializes
the sweep output into standard `open_ledger_summary.csv` files.

## Register Newly Generated Results

Use `--append-registry` only for intentional official evidence:

```powershell
python run/official_backtest_from_registry.py `
  --candidate-id some_candidate `
  --split val_2024 `
  --stresses normal,lag1,cost2x,capacity_3pct `
  --run-id official_some_candidate_YYYYMMDD `
  --append-registry
```

Do not append exploratory or forward-selected experiments as official evidence.
New official rows are registered with `canonical_evidence=true`. Superseding
evidence is an explicit governance operation; appending a newer row alone does
not silently invalidate older evidence.

## Build Attribution Coverage

```powershell
python run/attribution_from_registry.py
```
