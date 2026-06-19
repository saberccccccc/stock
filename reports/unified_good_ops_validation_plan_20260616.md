# Unified Good-Operations Validation Plan

Date: 2026-06-16

## Objective

Stop the current lag1-loss training and re-test the already discovered good or
possibly good operations under two fixed validation frameworks:

1. Frozen Top30 share-ledger framework for apples-to-apples comparison with the
   current frozen small-account baseline.
2. Realistic next-open open-to-open framework for live tradability.

No model, loss, threshold or strategy parameter may be selected from
2026-05-19 onward. Forward data remains untouched evidence.

## Training Status

The current lag1-loss ablation was interrupted intentionally. The partial
checkpoints may be kept for diagnostics, but they are not promoted and should
not drive the next decision unless they are later scored through this unified
validation protocol.

## Operations To Include

### A. Confirmed Useful Operations

1. V9 `average_w3` signal.
2. 9.5% signal-day chase filter.
3. 20% rebalance band for the frozen Top30 share-ledger strategy.
4. Legacy market timing for return-first comparison.
5. Dynamic market timing as a risk-control overlay.
6. Complete PIT fundamental factors; do not neutralize them by default.
7. Share ledger with whole-share lots, minimum commission, cash accounting and
   sell-before-buy execution.
8. Cost and delay stress tests before promotion.

### B. Possibly Useful Operations

1. Open-to-open `target3/hold50` and `target3.5/hold60`.
2. Minimum ADV 20M as an open-execution universe-quality filter.
3. Dynamic min0.2 and min0.3 market exposure overlays.
4. Frozen fallback Top20-style setting: `target=0.004`, `hold=0.06`.
5. V3 state-aware marginal-fill reranker.
6. V4 confidence-gated marginal-fill reranker.
7. Raw V9 signal as diagnostic only; do not rank against production results
   unless re-tested under the same execution engine.
8. V10 temporal wide-retention branch as research/ensemble only.
9. ST/listing-age filters as an unresolved data-quality improvement, pending
   exact fields.

### C. Explicitly Excluded From Promotion

1. `trade_policy_v1` high-return result.
2. Pure price stop-loss.
3. Old switch-value model.
4. V9+V10 rank ensemble as previously tested.
5. M0 + Top-focus 0.005.
6. Spread loss.
7. Checkpoint selection by IC, NDCG or Top30 stability alone.

## Candidate Set

Primary candidates with existing 2024 Alpha files:

| Candidate | Alpha file | Reason |
|---|---|---|
| Frozen V9 share-ledger | `lag1_checkpoint_sweep_m0_20260616/frozen_v9/alpha_maxret095.jsonl` | Current strongest frozen Top30/share-ledger reference; do not use as the old V9 open-to-open baseline |
| V9 avgw3/topfocus open-to-open | `backtest_results_v9_retention_20260531/topfocus_avgw3_{val,test}_legacy_finegrid/v9_daily_alpha_top_order.jsonl` | Historical open-to-open baseline; these JSONL files are missing in the copied workspace, but old summaries still reference them |
| M0 | `lag1_checkpoint_sweep_m0_20260616/m0_e006/alpha_maxret095.jsonl` | Current new-model baseline |
| A4 | `candidate_model_validation_20260614/a4_e6/alpha_maxret095.jsonl` | Best earlier auxiliary-loss candidate |
| V3 | `reranker_validation_20260615/regression_v3/alpha_maxret095.jsonl` | State-aware marginal-fill candidate |
| V4 | `reranker_validation_20260615/gated_v4/alpha_maxret095.jsonl` | Confidence-gated marginal-fill candidate |

Secondary candidates only after primary pass:

- Frozen fallback Top20 setting using the Frozen V9 alpha.
- Raw V9 signal, if a matched 2024 raw average-w3 alpha is available or
  regenerated. The copied workspace currently lacks the exact historical
  `topfocus_avgw3_*_legacy_finegrid` JSONL files, so old open-to-open summaries
  must not be overwritten by results from `alpha_maxret095.jsonl`.
- V10 temporal wide-retention branch, if converted to the same Alpha format.

## Framework 1: Frozen Top30 Share-Ledger

Purpose: compare against the current frozen small-account strategy.

Fixed settings:

```text
engine = run/backtest_retention_execution_constraints.py
target_frac = 0.006
hold_frac = 0.10
portfolio_value = 500000,1000000
min_adv_cny = 3000000
adv_participation_cap = 0.05
rebalance_band = 0.20
market_timing_mode = legacy
limit_threshold = 0.095
lot_size = 100
min_commission_cny = 5
execution_lag = 0
costs = 1x
```

Stress settings:

```text
cost2x = commission/stamp/slippage x2
cap3 = adv_participation_cap 0.03
lag1 = execution_lag 1
fallback = target_frac 0.004, hold_frac 0.06 for Frozen V9 only
```

Decision focus:

- 2024 base annualized return and Sharpe.
- 2x-cost robustness.
- One-day-lag robustness.
- Drawdown, turnover, blocked buys, unfilled turnover.
- Frozen V9 remains the reference, not a clean independent model test.

## Framework 2: Next-Open Open-To-Open

Purpose: test realistic execution for 500k-1M capital.

Fixed settings:

```text
engine = run/backtest_retention_open_execution.py
return_mode = open_to_open
target_fracs = 0.03,0.035
hold_fracs = 0.50,0.60
portfolio_value = 500000,1000000
min_adv_cny = 20000000
adv_participation_cap = 0.05
limit_threshold = 0.095
execution_lag = 0
costs = 1x
```

Market timing grid:

```text
legacy
dynamic min0.2
dynamic min0.3
```

Stress settings:

```text
cost2x for the best rows
cap3 for the best rows
optional open_to_close reference only after open_to_open is complete
```

Decision focus:

- Do not mix Frozen V9 share-ledger `alpha_maxret095.jsonl` with old V9
  avgw3/topfocus open-to-open results. They are different signal files.
- Compare models only within this open-execution framework.
- Prefer settings that keep validation 2024 acceptable and do not rely only on
  2025-2026 confirmation.

## Execution Order

1. Confirm no training Python process remains.
2. Run Framework 1 for primary candidates.
3. Run Framework 2 for primary candidates.
4. Summarize each framework separately.
5. Identify whether candidate failure is caused by model signal, execution
   timing, liquidity, market regime, or turnover.
6. Only after this comparison decide whether to resume any training.

## Promotion Rules

No candidate is promoted unless:

- It beats or closely matches Frozen V9 in the relevant framework.
- It does not collapse under 2x costs.
- It does not collapse under one-day delay or open-to-open execution.
- Drawdown and turnover are not materially worse.
- The result is not driven by a single month.

If no candidate passes, stop tuning loss terms and treat Frozen V9 as the live
baseline while researching data-quality and execution-layer improvements.
