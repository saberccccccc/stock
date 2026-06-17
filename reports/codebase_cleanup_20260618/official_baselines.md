# Official Baselines And Candidate Status 2026-06-18

## Purpose

This file freezes the current interpretation of project results before code cleanup.

Cleanup work may reorganize utilities and CLI wrappers, but it must not silently change which result is considered official or how backtests are compared.

## Official Baseline

Current official baseline:

```text
V9 avgw3 + maxret095 + open-price share-ledger
```

Core signal:

```text
checkpoint: checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt
signal: V9 avgw3
alpha family: full daily alpha ranking JSONL
filter: maxret095
```

Execution family:

```text
open-price share-ledger
```

Execution meaning:

- T day signal is generated after close.
- T day stocks with close-to-close return >= 9.5% are demoted before execution.
- Trades execute at T+1 open price.
- Position accounting uses cash, shares, board lots, minimum commission, stamp tax, slippage, ADV participation, and limit constraints.
- Backtests ending at 2026-05-18 must not use market data after 2026-05-18.

Official parameters:

```text
target_frac=0.006
hold_frac=0.10
max_new_names=5
rebalance_band=0.20
market_timing_mode=legacy
min_adv_cny=3_000_000
adv_participation_cap=0.05
limit_threshold=0.095
portfolio_values=500000,1000000
commission_rate=0.0001
stamp_tax_rate=0.0005
slippage_rate=0.0005
```

Required stress checks:

```text
normal
lag1
cost2x
capacity_3pct
```

## Current Result Interpretation

### Official

| Name | Role | Status |
|---|---|---|
| `official` | V9 avgw3 + maxret095 + open-price share-ledger | official baseline |

### Shadow Candidates

These can be observed and compared, but cannot replace the official baseline without a full validation packet.

| Name | Role | Current read |
|---|---|---|
| `breadth_m085` | Weak-market exposure cap | Best practical risk-control overlay candidate |
| `negfilter_drop3` | Remove worst reranker names from middle ranks | Strong historical test, weak forward robustness |
| `conditional_breadth_negfilter` | Apply mild negfilter only when breadth is weak | Research candidate, not promoted |
| `V3/V4 conservative reranker` | Conservative marginal fill reranker | Keep as shadow, do not retune from forward |

### Research Only

| Name | Role | Current read |
|---|---|---|
| `risk_target_r004` | Weak-state target shrink | Validation/forward loss reduction, but test damage suggests overfit risk |
| `edge_r030_100` | Middle-rank edge rerank | Low incremental value so far |
| `V10 temporal` | Temporal model research branch | Useful research, not current mainline |
| `open-to-open wide book` | Old V9 target3/3.5 hold50/60 framework | Keep as separate research framework |

### Rejected Or Not Comparable

| Name | Reason |
|---|---|
| `trade_policy_v1` | Not comparable; mixed policy semantics and alpha ranks |
| early fixed-window daily engines | Some accounting/strategy-semantics issues in old audit |
| pure stop-loss overlays | Prior tests damaged return and/or drawdown |
| V9 + V10 rank ensemble old attempt | Did not beat V9 mainline under strict execution |

## Comparison Rules

Do not mix these families in one direct ranking:

```text
official_open_price_share_ledger
legacy_close_based_or_constrained
research_open_to_open_wide_book
```

Allowed:

- rank candidates inside the same execution family;
- report cross-family differences as interpretation;
- use old high-return rows as signal diagnostics.

Not allowed:

- claim one model beats another when execution engine, target size, hold size, ADV filter, or date window differ;
- promote a candidate based only on Alpha IC;
- use 2026-05-19+ forward data to tune thresholds or choose checkpoints.

## Promotion Packet

A candidate can only replace the official baseline if it passes:

```text
validation normal
validation lag1
validation cost2x
validation capacity_3pct
test normal
test lag1
test cost2x
test capacity_3pct
50w and 100w
monthly stability
turnover/cost review
drawdown review
forward observation as non-tuning evidence
```

Alpha IC is only a minimum sanity gate, not the model selection objective.
