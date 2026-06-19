# Strategy Archaeology Review

Date: 2026-06-16

## Purpose

Re-check prior summaries and reports to identify strategy choices that were
already shown to work, but were not fully carried into the recent M0/loss/lag1
experiments.

Additional data audit artifacts generated during this review:

- `reports/result_csv_bestrow_audit_20260616.csv`: best Sharpe row from each
  detected result CSV.
- `reports/result_csv_strategy_summary_audit_20260616.csv`: filtered strategy
  summary rows, excluding yearly/monthly/diagnostic rows.

## Key Finding

Recent candidate validation drifted away from several historical frameworks
that had already produced useful evidence.

The recent evaluator `run/validate_candidate_models.py` uses the same broad
shape as the frozen Top30 small-account strategy: `target_frac=0.006`,
`hold_frac=0.10`, `min_adv=3M`, 20% rebalance band, legacy market timing and a
close-based constrained engine. That is useful for comparing new checkpoints
against the frozen small-account baseline, but it is not the same as the
stricter next-open open-to-open research framework tested on 2026-05-31.

Earlier reports had also tested a more realistic 1M account framework using
next-open open-to-open execution, `target=3.0%-3.5%`, `hold=50%-60%`,
`min_adv=20M`, and market exposure overlays.

Therefore several recent comparisons were useful for loss diagnosis and for
the frozen Top30 baseline, but they were not full open-execution
production-candidate tests.

## Two Historical Baselines Must Be Kept Separate

### Frozen forward baseline from 2026-06-12

Source: `FROZEN_FORWARD_STRATEGY.md`

This is the currently frozen forward ledger strategy:

- checkpoint: `checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt`
- signal: V9 `average_w3`
- target: `0.006` or about Top30
- hold: `0.10`
- market timing: legacy
- account sizes: CNY 500k and CNY 1M
- min ADV: CNY 3M
- board lot: 100 shares
- minimum commission: CNY 5
- rebalance band: 20% from the later optimization report
- execution: next tradable close, no extra delay

This baseline is the live/frozen comparison point. It is not open-to-open.

The frozen manifest also defines a fallback:

- target: `0.004` or about Top20
- hold: `0.06`

This fallback is an operational preference for fewer names. It is not a
parameter-search option after forward results are observed.

### Open-execution research framework from 2026-05-31

Source: `backtest_result_snapshots/20260531_model_strategy_comparison_report.md`

This is the stricter execution research framework:

- signal: V9 topfocus `average_w3`
- execution: next trading day's open
- return mode: open_to_open
- account size: CNY 1M
- min ADV: CNY 20M
- target region: 3.0%-3.5%
- hold region: 50%-60%
- market timing: legacy / dynamic min0.3 / dynamic min0.2

This should be revived for realistic execution validation, but it should not be
confused with the frozen Top30 ledger.

## Historically Useful Operations

### 1. Use next-open open-to-open as the primary execution check

Source: `backtest_result_snapshots/20260531_model_strategy_comparison_report.md`

The project already added `run/backtest_retention_open_execution.py` and used:

- capital: CNY 1M
- execution: next trading day's open
- return mode: open_to_open
- ADV cap: 5%
- min ADV: CNY 20M
- limit blocking: open gap beyond +/-9.5%
- full explicit costs

This should be the main validation framework for user-size capital.

### 2. Keep V9 average_w3 as the main production signal

Open-to-open tests rejected raw/average blends:

- pure `average_w3` remained strongest under realistic open execution
- raw weight worsened validation
- raw was mostly an unconstrained/backtest artifact

CSV audit nuance:

- Raw V9 has extremely strong old same-engine test rows, e.g. `target3/hold30`
  and `target3/hold40`.
- These are not the same as the frozen share-ledger/Top30 strategy and should
  not be compared directly against M0/loss candidates without matching
  execution assumptions.
- Under the stricter 1M next-open open-to-open framework, pure `average_w3`
  remained the safer production signal.

### 3. Use target around 3.0%-3.5%, not Top30/0.6%, for the old V9 production strategy

The stronger 1M open-to-open region was:

- return-first: `target3/hold50`
- validation-first: `target3.5/hold60`

The recent validation uses `target_frac=0.006`, equivalent to a much narrower
Top30-style book. That is appropriate for some new-model diagnostics, but it is
not the old best production strategy.

### 4. Use higher retention: hold 50%-60%

Earlier open-to-open fine grids found:

- `target3/hold50`: best test return and Sharpe
- `target3/hold60`: cleaner validation and lower turnover
- `target3.5/hold60`: best validation-first candidate

This differs from recent `hold_frac=0.10`.

### 5. Keep min ADV at 20M even for 1M capital

The old min-ADV sweep found 20M was not just a capacity control; it acted as a
universe-quality filter. 5M/10M had weaker validation, while 50M was too
restrictive.

### 6. Market timing matters, but as a return/risk choice

Open-to-open decision matrix:

| Profile | Setup | Validation | Test | Interpretation |
|---|---|---:|---:|---|
| Return-first | V9 avg_w3 target3/hold50 legacy | 35.46% / 0.989 / 20.44% MDD | 73.19% / 2.724 / 15.25% MDD | Max growth |
| Balanced live | V9 avg_w3 target3/hold50 dynamic min0.3 | 32.10% / 1.001 / 16.53% MDD | 60.04% / 2.947 / 11.11% MDD | Practical live candidate |
| Risk-first | V9 avg_w3 target3/hold50 dynamic min0.2 | 31.88% / 1.015 / 16.18% MDD | 57.35% / 3.044 / 9.91% MDD | Best drawdown/Sharpe |
| Validation-first | V9 avg_w3 target3.5/hold60 legacy | 36.65% / 1.024 / 20.21% MDD | 69.63% / 2.637 / 15.70% MDD | Strict validation winner |

Dynamic exposure did not raise annual return, but it reduced drawdown and
improved Sharpe. It should not be discarded; it is the risk-control overlay.

### 7. 9.5% signal-day chase filter is a proven improvement

Source: `reports/alpha_execution_filter_20260613.md`

For the frozen V9 strategy:

- 500k: 56.59% / 1.575 -> 59.56% / 1.625
- 1M: 58.83% / 1.597 -> 60.81% / 1.622
- lag stress Sharpe improved materially
- blocked buys fell

This filter should stay in every production validation.

### 8. Do not use pure price stop-loss

Earlier stop-loss tests showed -5% and -8% stops hurt both return and drawdown.

### 9. Do not rely on the old trade_policy_v1 or switch-value model

The audit marked trade_policy_v1 invalid/not comparable because it mixed
holding scores and alpha ranks. Later switch-value models reduced turnover but
gave up too much alpha. These are research branches, not production logic.

### 10. V9+V10 rank ensemble was already tested and rejected

Under strict execution, adding V10 lowered turnover slightly but reduced
validation return and Sharpe too much. Current V10 signal should remain a
research branch unless retrained/validated under the same open-to-open
framework.

### 11. Keep the 20% rebalance band

Source: `SHARPE_OPTIMIZATION_REPORT.md`

The rebalance band skips small resizing trades for retained positions while
still allowing new entries, exits, limit rules, board lots, liquidity limits
and market exposure changes.

It improved both 500k and 1M validation and independent-test results, and also
survived 2x, 3x and one-day-delay stress better than account-specific
alternatives. The common 20% band remains preferred.

Rejected nearby ideas:

- minimum CNY resize threshold for retained positions
- rank-tilted weights
- separate 10% band for 1M

### 12. Keep complete point-in-time fundamental factors

Source: `SHARPE_OPTIMIZATION_REPORT.md`

Neutralizing ROE, revenue growth and quarter-over-quarter changes at inference
reduced validation Sharpe materially. The PIT fundamental factors should stay
enabled. Earlier forward work also showed a "fundamental fixed" rerun, so
fundamental-data handling should be checked carefully before any new
comparison.

### 13. Reranker V3/V4 results are not failures of the whole reranking idea

Source: `RERANKER_IMPLEMENTATION_PLAN_20260614.md` and
`RERANKER_V4_PLAN_20260615.md`

Important retained conclusions:

- LambdaRank V1 improved NDCG but damaged the executable portfolio; reject it.
- V2 continuous executable target aligned better, but portfolio gains were too
  mixed; keep as evidence, not production.
- V3 state-aware marginal fill was a real improvement on 2024: higher Sharpe,
  lower drawdown, lower turnover and fewer blocked buys.
- V3 post-2024 confirmation improved Sharpe/drawdown but mixed raw return, so
  it became a frozen shadow candidate, not a live replacement.
- V4 confidence-gated marginal fill is safer than V3 because it abstains and
  reproduces M0 when inactive. V4 is the preferred shadow candidate.
- V4.1 meta-gate looked strong historically but failed the short true-forward
  window; reject it and do not retune from forward data.

This means the viable reranker direction is conservative state-aware marginal
fills, not broad daily reranking.

### 14. ST and listing-age filters remain an unresolved execution-quality gap

Older notes repeatedly mention adding ST/new-listing filters, but the raw CSV
coverage did not yet include the needed exact fields. This remains a missing
production filter rather than a rejected idea.

### 15. Earlier long-short/MVO results are strong but not directly applicable

Early reports showed very high Sharpe for long-short and MVO/optimizer variants.
Those are useful evidence that the alpha has cross-sectional structure, but
they do not directly solve the user's long-only 500k-1M account problem. MVO
can be kept as a low-drawdown research branch, not the current live path.

### 16. V10 temporal is a research candidate, not the current mainline

Sources:

- `backtest_result_snapshots/20260531_temporal_longonly_metric_report.md`
- `backtest_result_snapshots/20260604_exp_cleanup_report.md`

Important retained conclusions:

- V10's long-only checkpoint metric was changed away from long-short
  `topbot_h5` toward top-bucket return metrics.
- The best early V10 checkpoint could identify a broad useful pool, but daily
  full replacement had excessive turnover and poor validation returns.
- Retention-first state logic greatly improved V10 by lowering turnover, but
  validation 2024 remained weak without market defense.
- Market defense was essential for V10: legacy timing turned validation from
  roughly 5% annualized with 33%-35% drawdown into about 32% annualized with
  16%-17% drawdown in the best tested setup.
- The best practical V10 candidate from that report was around
  `target30/hold80 + legacy`, but it still did not beat the V9 mainline.
- Seed 5591 and two-seed rank ensemble improved validation robustness, but did
  not beat the original seed on test return. Ensemble is a model-risk
  smoothing idea, not a proven replacement.
- The later V10 warm-start run was also recorded as usable but not competitive
  with V9.

Therefore V10 should remain a research/ensemble branch. It should not distract
from first restoring the V9/M0 evaluation stack.

### 17. Some high historical results are audited-invalid or not comparable

Source: `backtest_result_snapshots/20260530_backtest_code_audit.md`

Do not use these as decision evidence:

- `trade_policy_v1` high annualized result: invalid/not comparable because it
  mixed holding probabilities with alpha ranks and effectively rebuilt the
  whole portfolio daily.
- Older engines where initial or gap-period transaction costs could be trimmed
  out of evaluated returns.
- Fixed-window engines for daily rank/hold/sell strategies.

For daily state or reranking ideas, use a dedicated portfolio state engine with
costs retained consistently.

### 18. Forward protocol is strict and should be preserved

Source: `RESEARCH_PROTOCOL.md`

- `data/raw` is the research snapshot through 2026-05-18.
- `data/forward_raw` is the chronological forward dataset from 2026-05-19.
- Forward observations must not be used to choose checkpoints, tune filters,
  alter thresholds or revise strategy parameters.
- Any later model/parameter change must become a new named forward experiment,
  not a rewrite of the active forward ledger.

This matters because several tempting V3/V4/V4.1 observations are forward or
post-selection evidence and cannot be used for retuning.

### 19. Data audit confirms which rows are admissible for decisions

The CSV-level audit found three classes of high-performing rows:

1. Very high Sharpe old same-engine rows, especially raw V9 on 2025-2026 test.
   These show signal strength but are not production-comparable unless the same
   execution engine is used.
2. Frozen/share-ledger Top30 rows, which are the proper comparison for recent
   M0/loss/reranker close-based validation.
3. Next-open open-to-open rows, which are the proper comparison for realistic
   execution.

For decisions, do not mix these three classes in the same ranking table.
Compare models only inside the same class, then summarize the trade-off across
classes.

## Recent Drift / Omitted Items

1. Recent candidate validation did not use open-to-open as the main metric.
2. Recent validation stayed in the frozen Top30 framework and did not also test
   the wider open-execution `target3/hold50-60` framework.
3. Recent validation used `min_adv=3M`, matching the frozen Top30 ledger but
   not the stricter open-execution framework's 20M quality filter.
4. Dynamic market timing was not consistently included in the new model
   comparison.
5. The 9.5% filter was kept in some tests, but the broader old production
   execution stack was not fully reproduced.
6. Recent lag1 labels/losses are close-based proxies, not open-to-open training
   targets.
7. Recent reports did not consistently distinguish live frozen baseline,
   shadow reranker candidates, and open-execution research candidates.
8. V10 temporal evidence was not summarized in the recent M0/loss discussion,
   which made it easy to forget that V10's main weakness was trading-layer
   turnover and regime sensitivity, not only prediction IC.
9. Some older impressive results are not admissible because the code audit
   found accounting or strategy-semantics issues.
10. Some old raw-signal same-engine test rows are valid as signal diagnostics
   but not admissible as live-strategy comparisons against share-ledger or
   open-execution results.

## Corrected Next Plan

### Stage R0: Freeze the reference framework

Build a new unified validator that can run any alpha/checkpoint through:

- frozen Top30 close-based ledger for apples-to-apples continuity
- next-open `open_to_open` for realistic execution
- optional `open_to_close`
- CNY 500k and CNY 1M
- frozen Top30 grid: target 0.006, hold 0.10
- open-execution grid: target 3.0%, 3.5%; hold 50%, 60%
- market timing: legacy, dynamic min0.3, dynamic min0.2
- min ADV: 3M for frozen Top30 continuity; 20M for open-execution research
- ADV cap: 5%, plus 3% stress
- costs: 1x and 2x
- 9.5% chase filter

### Stage R1: Re-score existing candidates

Run the corrected framework on:

- frozen V9 average_w3
- M0 epoch 6
- A4-E6
- downside/top-focus candidates if still needed
- lag1-loss candidates after training completes

### Stage R2: Decide whether M0 is actually worse under the right strategy

Do not compare M0 only under Top30/close-based validation. Compare it under the
same old production framework. If M0 still loses, stop tuning its loss and
focus on restoring V9-like signal quality or using V9 as the live baseline.

### Stage R3: Only then revisit loss design

If open-to-open validation shows M0 has usable alpha but poor execution:

- add open-execution metrics to checkpoint selection first
- only add loss terms after metrics prove which failure mode matters
- avoid adding Top-focus/downside losses that already failed close-based
  executable validation unless open-execution diagnostics contradict that

## Immediate Recommendation

The next action should not be more training. It should be a corrected
open-to-open validation pass over existing models using the old best
production-style settings.
