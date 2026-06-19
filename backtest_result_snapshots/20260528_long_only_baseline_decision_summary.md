# Backtest Snapshot 2026-05-28 Long-only Baseline

Saved before dynamic-exposure/risk-filter strategy experiments.

## Best Known Overall

- Long-short highest return: concentrated top_union_bottom_intersection simple_ls top1%, ann 87.91%, Sharpe 4.33, MDD 11.02%.
- Long-short balanced: concentrated avg_score simple_ls top3%, ann 62.55%, Sharpe 4.29, MDD 6.72%.
- Low drawdown: ensemble tubi optimizer_mvo_ra0p1, ann 29.28%, Sharpe 4.24, MDD 2.92%.
- Long-only baseline preference: top_union_bottom_intersection top10% + optimizer_projected_long.

## TUBI Simple Long

|   top_pct |   ann_raw |   sharpe_raw |   mdd_raw_pct |   ann_neu |   sharpe_neu |   mdd_neu_pct |
|----------:|----------:|-------------:|--------------:|----------:|-------------:|--------------:|
|     10.00 |     30.12 |         1.30 |         18.72 |     23.60 |         1.46 |         13.11 |
|      5.00 |     29.64 |         1.27 |         20.19 |     23.37 |         1.41 |         13.60 |
|      4.00 |     29.22 |         1.25 |         21.17 |     22.91 |         1.38 |         13.15 |
|      3.00 |     28.44 |         1.22 |         21.92 |     22.13 |         1.33 |         12.86 |
|      2.00 |     26.82 |         1.17 |         21.89 |     20.59 |         1.25 |         12.68 |
|      1.00 |     22.09 |         1.03 |         21.55 |     15.68 |         1.00 |         14.33 |

## TUBI Optimizer Projected Long

|   top_pct |   ann_raw |   sharpe_raw |   mdd_raw_pct |   ann_neu |   sharpe_neu |   mdd_neu_pct |
|----------:|----------:|-------------:|--------------:|----------:|-------------:|--------------:|
|     10.00 |     30.16 |         1.34 |         18.49 |     24.04 |         1.51 |         12.91 |
|      5.00 |     30.29 |         1.33 |         20.21 |     24.32 |         1.50 |         13.28 |
|      4.00 |     30.09 |         1.32 |         21.08 |     24.09 |         1.47 |         12.65 |
|      3.00 |     29.13 |         1.28 |         21.67 |     23.20 |         1.43 |         12.41 |
|      2.00 |     27.65 |         1.25 |         21.95 |     21.67 |         1.35 |         12.50 |
|      1.00 |     21.73 |         1.05 |         21.69 |     15.72 |         1.04 |         14.23 |
