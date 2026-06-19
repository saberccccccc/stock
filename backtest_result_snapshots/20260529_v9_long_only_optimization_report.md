# V9 Long-only Optimization Report 2026-05-29

## Goal

Improve the long-only backtest that was stuck around 30% annual return.

All tests used the experiment branch V9 checkpoint:

- `checkpoints_exp/ultimate_v7_best.pt`
- checkpoint epoch: 8
- validation alpha IC: 0.1104
- backtest return metric: `next_close_to_next_close`
- validation window: 775 trading days, average universe about 4935 stocks

## Scripts Added

- `run/v9_long_only_optimization.py`
- `run/v9_long_only_topfrac_sweep.py`
- `run/v9_top5_refine.py`

## Output Files

- `backtest_results_exp_v9_long_only_opt/v9_long_only_optimization_all_returns_summary.csv`
- `backtest_results_exp_v9_long_only_topfrac/v9_long_only_topfrac_summary.csv`
- `backtest_results_exp_v9_top5_refine/v9_top5_refine_summary.csv`

## Best Results

| Test | Ann | Sharpe | MDD | Neutral Ann | Neutral Sharpe | Neutral MDD | Comment |
|---|---:|---:|---:|---:|---:|---:|---|
| V9 top4% simple_long | 30.80% | 1.24 | 20.20% | 23.28% | 1.33 | 13.39% | highest raw annual return, drawdown too high |
| V9 top5% simple_long | 30.74% | 1.25 | 19.66% | 23.34% | 1.35 | 13.56% | best simple candidate |
| V9 top5% projected beta=0.15 | 30.62% | 1.31 | 19.43% | 24.07% | 1.44 | 13.78% | best balanced candidate |
| V9 top5% projected beta=0.30 | 30.30% | 1.28 | 19.62% | 23.49% | 1.39 | 13.72% | baseline projected top5 |
| V9 top5% weak timing + alpha/vol | 25.84% | 1.38 | 15.68% | 19.91% | 1.49 | 12.24% | defensive only |
| Previous V9+GAT top10 projected baseline | 30.16% | 1.34 | 18.49% | 24.04% | 1.51 | 12.91% | still better risk-adjusted |

## Findings

1. V9-only can exceed the old 30% long-only return, but only by concentrating to top4%-5%.
2. Top10% is not optimal for V9-only. It gives about 28.4%-28.8% annual return.
3. Wider holdings reduce drawdown but steadily reduce return. Top20%-30% falls to about 26.7%-27.6% annual.
4. `alpha_vol_power=0.5` reliably improves Sharpe and drawdown, but gives up too much annual return for the main strategy.
5. Dynamic weak-market timing also gives up too much annual return. It should be kept as a defensive variant, not mainline.
6. Hysteresis did not improve returns or drawdown materially.
7. V9+GAT top10 remains slightly better risk-adjusted than the new V9-only top5 candidates, but V9-only top5 is simpler and has higher raw annual return.

## Recommendation

Use two long-only variants:

1. Main return-seeking candidate: `V9 top5% simple_long`
   - annual return 30.74%
   - Sharpe 1.25
   - MDD 19.66%

2. Balanced candidate: `V9 top5% optimizer_projected, beta_limit=0.15`
   - annual return 30.62%
   - Sharpe 1.31
   - MDD 19.43%
   - neutral annual return 24.07%

Do not replace the old V9+GAT top10 projected baseline yet. The new V9-only top5 improves raw return but has higher drawdown and lower Sharpe. It is a promising candidate, not a clean win.

## Next Best Test

The remaining bottleneck is not portfolio mechanics; it is that the model is still trained for cross-sectional IC / long-short quality. The next model experiment should add a small top-only auxiliary loss:

- reward top decile or top 5% forward return versus universe average
- keep weight small, around 0.01 to 0.05
- evaluate long-only top5% and top10% directly after training

This targets the actual production use case instead of rewarding bottom-leg accuracy that only helps long-short.
