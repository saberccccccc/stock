# Strategy Optimization Summary 2026-05-28

## Current Ranking

| name                                               |   ann |   sharpe |   mdd |   ann_neu |   sharpe_neu |   mdd_neu | note                                                       |
|:---------------------------------------------------|------:|---------:|------:|----------:|-------------:|----------:|:-----------------------------------------------------------|
| alpha/vol p0.5 tubi top10 optimizer_projected_long | 28.75 |     1.38 | 18.08 |     23.15 |         1.58 |     11.02 | better Sharpe, lower neutral drawdown, lower annual return |
| baseline tubi/union top10 optimizer_projected_long | 30.16 |     1.34 | 18.49 |     24.04 |         1.51 |     12.91 | current main long-only baseline                            |
| union long_top5pct avg_rank                        | 30.24 |     1.33 | 20.21 |     24.28 |         1.49 |     13.28 | union top1-5 grid                                          |
| union long_top4pct avg_rank                        | 30.07 |     1.32 | 21.08 |     24.08 |         1.47 |     12.65 | union top1-5 grid                                          |
| risk filter vol q90 tubi top10                     | 28.88 |     1.31 | 18.57 |     22.90 |         1.47 |     12.24 | not better than baseline                                   |
| union_max_score long_top4pct max_rank              | 30.17 |     1.31 | 20.81 |     23.80 |         1.46 |     13.85 | union top1-5 grid                                          |
| union_max_score long_top5pct max_rank              | 29.85 |     1.29 | 20.46 |     23.57 |         1.45 |     14.12 | union top1-5 grid                                          |
| union long_top3pct avg_rank                        | 29.20 |     1.29 | 21.72 |     23.26 |         1.43 |     12.41 | union top1-5 grid                                          |
| dynamic exposure min0.30 tubi top10                | 24.41 |     1.28 | 14.75 |     19.62 |         1.44 |     10.81 | lower drawdown but annual return drops too much            |
| dynamic exposure min0.20 max0.90 tubi top10        | 21.14 |     1.26 | 13.11 |     17.06 |         1.41 |     10.34 | defensive version only                                     |
| union long_top2pct avg_rank                        | 27.79 |     1.25 | 21.93 |     21.80 |         1.36 |     12.50 | union top1-5 grid                                          |
| union_max_score long_top3pct max_rank              | 27.09 |     1.19 | 22.29 |     20.90 |         1.29 |     13.76 | union top1-5 grid                                          |
| union_max_score long_top2pct max_rank              | 23.56 |     1.07 | 20.96 |     17.24 |         1.10 |     13.76 | union top1-5 grid                                          |
| union long_top1pct avg_rank                        | 21.61 |     1.04 | 21.69 |     15.58 |         1.03 |     14.62 | union top1-5 grid                                          |
| union_max_score long_top1pct max_rank              | 20.95 |     0.96 | 21.71 |     14.97 |         0.95 |     14.68 | union top1-5 grid                                          |

## Conclusions

- Main long-only baseline remains `tubi/union top10 + optimizer_projected_long`: annual ~30.16%, Sharpe ~1.34, MDD ~18.49%.
- `alpha/vol p0.5` improves Sharpe and neutral drawdown, but gives up ~1.4 pct annual return. Keep as a conservative variant.
- Dynamic exposure lowers drawdown, but annual return loss is too large for the main strategy.
- Volatility risk filter and max-rank union did not improve the strategy.
- Top1%-5% concentration is worse than top10%; keep top10 as the main long-only concentration.

## Next Tests

1. Hysteresis/holding continuation: buy top10%, keep existing names until they fall below top15% or top20%.
2. Layered weights: top3/top10 or top5/top10 weighting.
3. Weak-market-only exposure cut, not full-period dynamic exposure.
