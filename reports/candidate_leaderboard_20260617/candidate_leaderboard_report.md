# Candidate Leaderboard 2026-06-17

## Scope

This report compares the current official open-price share-ledger baseline with the main observation candidates:

```text
official:
V9 avgw3 + maxret095 + open-price share-ledger

breadth_m085:
official + breadth weak-day market multiplier cap
trigger: up_ma3 <= 0.35
action: cap market_mult at 0.85

risk_target_r004:
official + target shrink in weak market states

edge_r030_100:
rerank only base ranks [30, 100)

negfilter_drop3:
remove 3 worst full-reranker names from base ranks [30, 100)
```

Common execution settings:

```text
target_frac=0.006
hold_frac=0.10
max_new_names=5
rebalance_band=0.20
market_timing_mode=legacy
minADV=3M
ADV cap=5%
limit/maxret signal filter=9.5%
```

Detailed CSV:

```text
reports/candidate_leaderboard_20260617/candidate_leaderboard.csv
```

## Normal Leaderboard

### 50w

| Split | Rank | Candidate | Ann | Sharpe | MDD |
|---|---:|---|---:|---:|---:|
| Validation | 1 | risk_target_r004 | 90.82% | 1.920 | 18.85% |
| Validation | 2 | breadth_m085 | 83.33% | 1.837 | 20.39% |
| Validation | 3 | official | 84.08% | 1.834 | 21.07% |
| Validation | 4 | edge_r030_100 | 80.89% | 1.789 | 20.67% |
| Validation | 5 | negfilter_drop3 | 80.51% | 1.777 | 21.05% |
| Test | 1 | negfilter_drop3 | 87.03% | 3.009 | 13.94% |
| Test | 2 | breadth_m085 | 84.63% | 2.991 | 14.11% |
| Test | 3 | official | 84.33% | 2.954 | 14.28% |
| Test | 4 | edge_r030_100 | 84.30% | 2.953 | 14.28% |
| Test | 5 | risk_target_r004 | 81.60% | 2.770 | 15.38% |
| Forward | 1 | risk_target_r004 | -54.11% | -2.384 | 10.91% |
| Forward | 2 | breadth_m085 | -55.25% | -2.711 | 10.31% |
| Forward | 3 | negfilter_drop3 | -60.64% | -2.976 | 11.16% |
| Forward | 4 | official | -61.03% | -2.997 | 11.23% |
| Forward | 5 | edge_r030_100 | -61.03% | -2.997 | 11.23% |

### 100w

| Split | Rank | Candidate | Ann | Sharpe | MDD |
|---|---:|---|---:|---:|---:|
| Validation | 1 | risk_target_r004 | 95.67% | 1.957 | 18.80% |
| Validation | 2 | official | 89.31% | 1.879 | 20.79% |
| Validation | 3 | breadth_m085 | 88.24% | 1.878 | 20.43% |
| Validation | 4 | edge_r030_100 | 84.78% | 1.818 | 20.36% |
| Validation | 5 | negfilter_drop3 | 84.87% | 1.812 | 21.09% |
| Test | 1 | negfilter_drop3 | 91.08% | 2.927 | 15.54% |
| Test | 2 | breadth_m085 | 87.07% | 2.890 | 15.21% |
| Test | 3 | official | 88.23% | 2.888 | 15.24% |
| Test | 4 | edge_r030_100 | 88.21% | 2.887 | 15.24% |
| Test | 5 | risk_target_r004 | 84.84% | 2.727 | 16.52% |
| Forward | 1 | risk_target_r004 | -39.05% | -1.271 | 11.41% |
| Forward | 2 | breadth_m085 | -45.03% | -1.675 | 10.71% |
| Forward | 3 | edge_r030_100 | -51.22% | -1.918 | 11.78% |
| Forward | 4 | official | -52.25% | -2.017 | 11.78% |
| Forward | 5 | negfilter_drop3 | -53.06% | -2.086 | 11.78% |

## Stress Notes

Validation/test stress:

- `negfilter_drop3` is the strongest historical test candidate.
  - 50w test lag1: 80.28% ann, Sharpe 2.868.
  - 50w test cost2x: 75.63% ann, Sharpe 2.720.
- `breadth_m085` is the strongest validation stress candidate.
  - 50w validation lag1: 54.86% ann, Sharpe 1.375.
  - 50w validation cost2x: 73.16% ann, Sharpe 1.688.
- `edge_r030_100` is nearly identical to official in most forward results and does not currently justify promotion.

Forward stress:

| Stress | Best Practical Candidate | Reason |
|---|---|---|
| normal | risk_target_r004 by Sharpe, breadth_m085 by MDD | risk target cuts loss more, breadth has lower drawdown |
| lag1 | breadth_m085 | reduces loss and drawdown versus official, edge, and negfilter |
| cost2x | breadth_m085 | reduces loss and drawdown versus official, edge, and negfilter |

## Interpretation

No candidate should replace official outright yet.

Current roles:

```text
official:
formal baseline

breadth_m085:
best risk-control overlay candidate;
worth continued observation and possible conservative blend

negfilter_drop3:
best historical test enhancement;
not robust enough in forward yet

risk_target_r004:
strong validation/forward loss reduction;
too much evidence of test damage, likely state overfit

edge_r030_100:
low value; keep only as reference unless future evidence changes
```

## Next Action

The next useful experiment is not another full reranker tweak. The better direction is a conservative combined overlay:

```text
official ranking
+ maxret095
+ breadth_m085 market multiplier cap
+ very mild negfilter only when breadth trigger is active
```

Rationale:

- breadth handles weak-market exposure;
- negfilter helps historical test selection quality;
- applying negfilter only during weak breadth may reduce its forward damage.

Promotion rule:

```text
Do not promote unless validation/test normal, lag1, cost2x, monthly stability,
and forward observation are all acceptable for both 50w and 100w.
```
