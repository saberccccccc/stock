# Research And Forward Evaluation Protocol

## Canonical Split Contract

The single source of truth is `core/research_protocol.py`.

| Split | Interval | Role | Selection eligible |
|---|---|---|---|
| `val_2024` | 2024-01-01 through 2024-12-31 | validation evidence | yes |
| `test_2025` | 2025-01-01 through 2025-12-31 | test evidence under the predeclared joint gate | yes |
| `forward_2026` | 2026-01-01 through the latest complete 2026 date, currently 2026-06-30 | forward observation | no |

`2026-05-18` is legacy cache/report metadata, not a Forward boundary. It may
still appear in historical artifact names and regime-attribution reports.

## Selection And Lineage Rules

- Model, checkpoint, blend, policy, and parameter selection use only 2024 Val
  and 2025 Test under the predeclared decision rule.
- 2026 Forward is observation-only. It cannot rank candidates, change a
  threshold, choose a checkpoint, or justify a parameter rerun.
- A model claiming full-year 2026 Forward evidence must finish model fitting,
  train-fitted preprocessing, checkpoint selection, and policy selection no
  later than 2025-12-31, after label-tail purge.
- A registry row with `is_forward=true` must have
  `selection_eligible=false`. Contradictory rows are invalid evidence.
- Every formal result records requested split plus actual `signal_start`,
  `signal_end`, `backtest_start`, and `backtest_end`.

## Data Views

The target architecture is one versioned physical PIT market-data store with
experiment-defined logical date views. The current `data/raw` and
`data/forward_raw` directories remain compatibility roots during migration;
directory names do not determine selection eligibility.

- A provider may read pre-split history for feature warm-up.
- `max_data_date` is a read ceiling, not a split definition.
- Formal manifests must distinguish physical coverage, feature warm-up,
  train/valid ranges, transform fit range, signal range, and backtest range.
- Financial statements become available on `ann_date`/`effective_date`, with
  estimated notice and freshness/age flags retained when applicable.
- Later physical rows must never leak into a train-fitted processor or label.

## Execution Contract

- Official execution is the project-native realistic open-price share-ledger.
- Signal information is available after T close; normal execution is T+1 open.
- The ledger owns cash, shares, A-share lots, minimum commission, taxes/fees,
  ADV participation, suspension/zero-volume, listing age, and board/ST price
  limits.
- Qlib strategy ideas may propose holdings, but Qlib's executor must not
  replace `open_ledger` in official evidence.
- Official comparisons use CNY 500,000 and CNY 1,000,000 under `normal`,
  `lag1`, `cost2x`, and `capacity_3pct`.

## Historical ST Limitation

Historical ST adapters and download logic exist, but a complete audited event
file is not currently available. This must remain visible in execution
coverage. It blocks a claim of complete historical-ST realism, but does not
block framework construction or a rolling audit explicitly carrying this
limitation. Repeated low-rate `namechange` polling is not on the critical path.

## Required Audit Example

```powershell
$python = "$env:USERPROFILE\miniconda3\envs\torch\python.exe"
& $python run/audit_data_boundary.py `
  --dataset-role research `
  --effective-end-date 2025-12-31 `
  --output reports/qlib_research_framework_20260712/data_boundary_audit_research_YYYYMMDD.json
```

Forward audits use the same physical source contract, an explicit
`forward_2026` logical view, and the current complete `max_data_date`.
