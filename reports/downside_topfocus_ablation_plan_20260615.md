# Downside and Top-focus Ablation Plan

Date: 2026-06-15

## Objective

Test whether a long-only downside penalty improves executable Top30 results
over M0, and whether a small Top-focus term is complementary.

The frozen references are:

- M0: weighted multi-horizon global IC only.
- M1: M0 plus Top-focus 0.005, already rejected in 2024 execution tests.

## Downside Definition

For each stock, take the worst raw cumulative return among h1, h3, h5 and h7.
Only negative worst returns count as downside. The loss penalizes downside
concentrated in the model's soft long-only top book.

Raw returns are used instead of normalized cross-sectional labels. The loss is
enabled from epoch 3 so the main IC objective establishes the ranking first.

## Experiment Matrix

| ID | Downside | Top-focus | Purpose |
|---|---:|---:|---|
| M0 | 0 | 0 | Existing baseline |
| D001 | 0.001 | 0 | Low downside weight |
| D003 | 0.003 | 0 | Middle downside weight |
| D005 | 0.005 | 0 | High downside weight |
| T001 | 0 | 0.001 | Isolate low Top-focus weight |
| D003_T001 | 0.003 | 0.001 | Test interaction |
| M1 | 0 | 0.005 | Existing rejected upper Top-focus reference |

All new runs use seed 42, six epochs, training labels through 2023-12-31,
validation labels through 2024-12-31, batch size 4 and gradient accumulation 4.

## Evaluation Order

1. Compare every epoch, not only the checkpoint selected by validation IC.
2. Run the fixed 2024 executable Top30 backtest with the 9.5% chase filter.
3. Test CNY 500k and CNY 1m under base, doubled costs, one-day delay and 3% ADV.
4. Reject settings that improve IC but reduce executable Sharpe or delay robustness.
5. Prefer a simpler single-loss setting unless the combined setting improves both
   account sizes and does not materially worsen drawdown, turnover or blocked buys.

No data after 2024-12-31 is used for training, validation or parameter selection.
