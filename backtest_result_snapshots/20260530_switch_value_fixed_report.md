# Switch Value Model Fixed Run Report - 2026-05-30

## Purpose

Build and validate a no-hand-threshold switch value dataset/model:

- Current holding `A`
- Candidate replacement `B`
- Learn whether switching from `A` to `B` has enough forward excess return after explicit cost, impact cost, and execution-risk proxy.

This is not yet a final backtest policy. It is the dataset and bucket validation stage.

## Code Audit Fixes

### 1. Removed lookahead in switch dataset universe construction

File: `run/build_switch_value_dataset.py`

Original issue:

```python
valid = np.isfinite(y_seq[:, :max(HORIZONS)]).all(axis=1)
```

This used future label availability to decide today's ranked/tradable universe. That is a future-information leak and can also create survivorship-style filtering.

Fixed behavior:

- Today's alpha universe is based on current prediction validity only.
- Future return labels are checked only after an `A -> B` pair is generated.
- Pairs with missing future labels are skipped, but they no longer change today's rank list.

### 2. Added current alpha finite filtering

`np.argsort` on non-finite alpha values can pollute rankings. The dataset now filters non-finite current alpha before ranking.

### 3. Added per-date feature caching

Repeated per-pair calls to price/return/liquidity helpers were a major slowdown. The builder now caches features for today's holdings and candidates before pair generation.

### 4. Added safer output behavior

The builder now supports:

- `--skip-csv`
- split-level parquet output
- progress logging with `--progress-every`

This avoids waiting for the whole run before seeing any output and avoids large CSV overhead.

### 5. Fixed training report horizon hardcoding

File: `run/train_switch_value_model.py`

Original issue:

- Bucket reports always used `switch_success_h5` and `switch_edge_raw_h5`.
- If training h1/h3, reports would silently use the wrong horizon.

Fixed behavior:

- Horizon is parsed from `--target-col`.
- Bucket/yearly reports use matching success/raw edge columns.

## Full Dataset

Command:

```bash
F:/miniconda3/envs/pytorch/python run/build_switch_value_dataset.py --output-dir switch_value_data_20260530_fixed --max-pairs-per-holding 3 --skip-csv --progress-every 200
```

Output:

- `switch_value_data_20260530_fixed/switch_value_dataset.parquet`
- `switch_value_data_20260530_fixed/switch_value_dataset_train.parquet`
- `switch_value_data_20260530_fixed/switch_value_dataset_val.parquet`
- `switch_value_data_20260530_fixed/switch_value_dataset_summary.csv`

Rows:

| split | rows | h1 net edge mean | h3 net edge mean | h5 net edge mean | h5 success | cost mean |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,242,012 | 0.1932 | 0.1166 | 0.1083 | 0.5218 | 0.0021 |
| val | 571,269 | 0.2561 | 0.1258 | 0.0766 | 0.5075 | 0.0021 |

Interpretation:

- Candidate pool itself has positive average switch edge versus current holdings.
- But validation h5 success is only slightly above 50%, so the task is noisy.
- Average explicit+risk+impact cost is 0.21%, matching the configured commission/slippage/stamp assumptions when impact/risk is mostly zero.

## H5 Value Model

Command:

```bash
F:/miniconda3/envs/pytorch/python run/train_switch_value_model.py --dataset switch_value_data_20260530_fixed/switch_value_dataset.parquet --output-dir switch_value_models_20260530_fixed/switch_edge_lgb_h5 --target-col switch_edge_net_h5 --n-estimators 500 --min-child-samples 500
```

Output:

- `switch_value_models_20260530_fixed/switch_edge_lgb_h5/switch_value_model.pkl`
- `switch_value_models_20260530_fixed/switch_edge_lgb_h5/metrics.json`
- `switch_value_models_20260530_fixed/switch_edge_lgb_h5/bucket_report_val.csv`
- `switch_value_models_20260530_fixed/switch_edge_lgb_h5/yearly_bucket_report_val.csv`
- `switch_value_models_20260530_fixed/switch_edge_lgb_h5/feature_importance.csv`

Metrics:

| split | rows | target mean | Spearman IC | success | pred positive true edge | pred positive success |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,242,012 | 0.1083 | 0.3676 | 0.5218 | 0.4773 | 0.6184 |
| val | 571,269 | 0.0766 | 0.0347 | 0.5075 | 0.1125 | 0.5175 |

Interpretation:

- Train fit is much stronger than validation, so overfitting/noise is real.
- Validation IC is positive but weak.
- The model has some filtering value, but not enough evidence yet to use it alone as a full no-threshold trading engine.

## Validation Buckets

Validation decile summary:

| bucket | rows | pred mean | true h5 net edge | success |
|---|---:|---:|---:|---:|
| lowest | 57,127 | -0.3436 | 0.1489 | 0.5066 |
| 2 | 57,127 | -0.0976 | -0.0379 | 0.4780 |
| 3 | 57,127 | -0.0451 | 0.0014 | 0.4959 |
| 4 | 57,127 | -0.0242 | -0.0030 | 0.4929 |
| 5 | 57,127 | 0.0140 | 0.0197 | 0.4994 |
| 6 | 57,126 | 0.0631 | 0.0451 | 0.5064 |
| 7 | 57,127 | 0.1271 | 0.0649 | 0.5120 |
| 8 | 57,127 | 0.2230 | 0.0739 | 0.5164 |
| 9 | 57,127 | 0.3840 | 0.1584 | 0.5277 |
| highest | 57,127 | 0.8420 | 0.2945 | 0.5401 |

The top predicted bucket is clearly better than the middle buckets, but the bottom bucket is anomalously positive. This means the model is useful mainly for identifying very strong switch candidates, not for cleanly ranking the entire candidate set.

Yearly top bucket:

| year | true h5 net edge | success |
|---|---:|---:|
| 2023 | 0.1321 | 0.5368 |
| 2024 | 0.2075 | 0.5182 |
| 2025 | 0.2155 | 0.5308 |
| 2026 | 0.4567 | 0.5715 |

The top bucket is positive across validation years, which is the strongest encouraging sign in this run.

## Feature Importance

Top features are dominated by candidate `B` alpha/rank dynamics:

1. `B_alpha`
2. `B_alpha_change_3d`
3. `B_alpha_change_1d`
4. `B_rank_pct`
5. `B_rank_change_3d`
6. `B_alpha_ma3`
7. `B_rank_change_1d`
8. `B_alpha_vs_ma3`

This means the model is mostly learning "which replacement looks strong" rather than a deep comparison between current holding `A` and candidate `B`.

## Conclusion

The fixed dataset/model stage is usable but not enough yet for final deployment:

- Good: no obvious future-label universe leak after fix.
- Good: top predicted switch bucket is positive in every validation year.
- Weak: validation IC is only 0.0347.
- Weak: ranking is not monotonic across all buckets.
- Weak: feature importance is too dominated by `B` alpha, so the value model currently behaves partly like a candidate-alpha filter.

Recommended next step:

1. Backtest a conservative retention-first policy where the value model is used only to choose replacements among vacancies or obvious weak holdings.
2. Do not allow the value model to force large daily turnover yet.
3. Compare against daily alpha top5 baseline and previous v2 retention-first policy under explicit costs.
4. Add diagnostics for realized turnover, average holding days, failed-switch proxy, and yearly performance.
