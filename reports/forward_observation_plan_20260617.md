# Forward Observation Plan 2026-06-17

## Current Decision

Official baseline:

```text
main_candidate
V9 avgw3 + maxret095 + open-price share-ledger
target_frac=0.006
hold_frac=0.10
max_replace=5
rebalance_band=0.20
market=legacy
```

First attack candidate:

```text
negfilter_r030_100_drop3
```

First stability candidate:

```text
edge_r030_100
```

The attack candidate is not an official replacement yet because its historical test advantage is strongly concentrated in 2026-04/05.

## Observation Rules

Use the same open-price share-ledger execution for every candidate:

```text
capital=500000,1000000
target_frac=0.006
hold_frac=0.10
max_new_names=5
rebalance_band=0.20
market=legacy
max_data_date=<latest available date>
```

Do not select candidates by Alpha IC.

Ranking priority:

1. Forward/live executable portfolio Sharpe.
2. Annualized return.
3. Lag1 robustness.
4. Cost2x robustness.
5. MDD.
6. Turnover and blocked buys.

## Promotion Gate

`negfilter_r030_100_drop3` can replace the official baseline only if forward/live data shows:

- 50w and 100w Sharpe both not lower than baseline;
- annualized return improves or stays flat;
- MDD does not materially worsen;
- lag1 and cost2x are not weaker;
- improvement is not concentrated in one short burst.

## Low-Memory Discipline

Allowed:

- JSONL transforms;
- single-candidate open-ledger runs;
- CSV summary scripts.

Avoid:

- loading full feature caches for parameter sweeps;
- retraining reranker models during forward observation;
- parallel backtests on a 16GB RAM machine.

## Candidate Artifacts

Baseline alpha:

```text
v9_avgw3_filter095_validation_20260616/v9_avgw3_val_maxret095.jsonl
v9_avgw3_extend_to_20260518_20260616/v9_avgw3_test_to_20260515_maxret095.jsonl
```

Attack candidate alpha:

```text
open_reranker_current_v9_negfilter_20260617/r030_100_drop3/val_negfilter_r030_100_drop3.jsonl
open_reranker_current_v9_negfilter_20260617/r030_100_drop3/test_negfilter_r030_100_drop3.jsonl
```

Stability candidate alpha:

```text
open_reranker_current_v9_edge_20260617/r030_100/val_edge_r030_100_w095.jsonl
open_reranker_current_v9_edge_20260617/r030_100/test_edge_r030_100_w095.jsonl
```

## Current Summary Files

```text
reports/open_ledger_candidate_summary_20260617/candidate_summary.md
reports/open_ledger_candidate_summary_20260617/candidate_summary_long.csv
reports/open_ledger_candidate_summary_20260617/candidate_summary_normal_wide.csv
reports/open_price_share_ledger_optimization_log_20260616.md
```

## Next Practical Step

After new post-2026-05-18 market data is updated:

1. Generate/update baseline alpha for the new dates.
2. Generate/update `negfilter_r030_100_drop3` alpha by applying the same low-memory negative filter transform.
3. Run open-price share-ledger for baseline and attack candidate.
4. Append forward results to this plan and to the optimization log.

## Reproducible Forward Script

Added helper script:

```text
run_forward_observation_candidates_20260617.ps1
```

Purpose:

- take a baseline V9 avgw3/maxret095 alpha file and a full-rerank w095 alpha file;
- generate `negfilter_r030_100_drop3` with the low-memory negative-filter transform;
- generate `edge_r030_100` with the low-memory partial-rerank transform;
- run the same open-price share-ledger execution for all three candidates.

Example after data and alpha files have been updated:

```powershell
.\run_forward_observation_candidates_20260617.ps1 `
  -BaseAlpha <updated_base_alpha.jsonl> `
  -FullRerankAlpha <updated_full_rerank_alpha.jsonl> `
  -OutputRoot <forward_output_dir> `
  -MaxDataDate <latest_tradable_date>
```

Use `-SkipBacktest` if only candidate alpha files should be generated.

Memory rule:

- run this script by itself;
- do not run other training jobs or parallel backtests at the same time on the 16GB machine;
- if the full-rerank alpha has not been produced yet, avoid retraining/reranking with full feature-cache loading unless memory is clearly available.

## 2026-06-17 Forward Update

Data was updated after the machine restart:

- `data/raw` stock files mostly reached 2026-06-17; broad/industry indices reached 2026-06-16.
- `data/forward_raw` stock files were incrementally updated; broad/industry indices reached 2026-06-16.
- Forward evaluation uses `data/forward_raw` and `max_data_date=2026-06-16`.

Generated:

```text
forward_results/frozen_v9_avgw3/alpha_20260519_20260616.jsonl
forward_results/frozen_v9_avgw3/alpha_20260519_20260616_maxret095.jsonl
```

Main observation:

- Official params `target=0.006/hold=0.10/max_new_names=5/legacy` were poor in 2026-05-19 to 2026-06-16:
  - 50w: ann -61.03%, Sharpe -2.997, MDD 11.23%
  - 100w: ann -52.25%, Sharpe -2.017, MDD 11.78%
- Lower target was clearly better:
  - 50w target 0.002: ann -23.16%, Sharpe -0.402
  - 100w target 0.003: ann -1.24%, Sharpe 0.170
- Dynamic timing reduced MDD to about 7%, but did not improve Sharpe.

New follow-up:

Validate low-target overlays `target_frac=0.002/0.003/0.004` on historical validation/test with the same open-price share-ledger. Do not promote based only on this short forward window.

Detailed report:

```text
reports/forward_observation_20260617/forward_observation_20260519_20260616.md
reports/forward_observation_20260617/forward_open_ledger_summary_20260519_20260616.csv
reports/forward_observation_20260617/forward_open_ledger_top_by_sharpe_20260519_20260616.csv
```

## 2026-06-17 Target Fraction Follow-Up

Historical validation/test was rerun to verify the forward low-target clue.

Result:

- Fixed low target helps validation and forward.
- Fixed low target hurts historical test.
- Mid target `0.0055` is not a clean compromise.
- Risk-target switching helps validation, but still hurts test.

Best risk-target observation candidate:

```text
risk_target_r004
normal target=0.006
risk target=0.004 when market_mult < 1.0
```

Validation:

- 50w: ann 90.82%, Sharpe 1.920, MDD 18.85%
- 100w: ann 95.67%, Sharpe 1.957, MDD 18.80%

Test:

- 50w: ann 81.60%, Sharpe 2.770, MDD 15.38%
- 100w: ann 84.84%, Sharpe 2.727, MDD 16.52%

Decision:

Do not replace the official fixed `target=0.006` baseline. `risk_target_r004` is observation-only because it improves validation and slightly improves forward, but it still weakens test versus official `0.006`.

Code update:

```text
run/backtest_retention_open_ledger.py
```

New optional arguments:

```text
--risk-target-frac
--risk-target-market-mult-below
```

Default behavior is unchanged when these arguments are omitted.

Detailed report:

```text
reports/target_fraction_validation_20260617/target_fraction_validation_report.md
reports/target_fraction_validation_20260617/target_fraction_all_results.csv
reports/target_fraction_validation_20260617/target_fraction_compact_results.csv
reports/target_fraction_validation_20260617/target_fraction_top_by_sharpe.csv
```

Next direction:

Do not simply lower target. Build a better risk trigger based on recent executable strategy performance, breadth, or alpha-spread deterioration, then test whether it lowers target only during genuinely harmful regimes.
