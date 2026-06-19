# V9 Top-Metric Selection Test - 2026-05-29

## Change

- Added validation metrics for top buckets:
  - `topic_h{h}_top{pct}`: IC inside predicted top bucket.
  - `topret_h{h}_top{pct}`: average realized label in predicted top bucket.
  - Existing `topbot_h*` is preserved.
- Added configurable checkpoint selection:
  - `DataConfig.best_val_metric`
  - CLI `--best-val-metric`
  - CLI `--eval-top-fracs`

## Test Run

Command:

```powershell
F:/miniconda3/envs/pytorch/python -X utf8 -u run/train.py --model v9 --epochs 18 --lr 1e-4 --output-dir checkpoints_exp_topfocus_w005_topic --top-focus-loss-weight 0.005 --top-focus-temperature 0.75 --top-focus-delay-epochs 5 --best-val-metric topic_h5_top10 --eval-top-fracs 0.05,0.10
```

Best checkpoint:

- path: `checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt`
- epoch: 12
- selected metric: `topic_h5_top10 = 0.02314`
- `alpha_IC = 0.11243`
- `topbot_h5 = 0.33488`
- `topret_h5_top5 = 0.12467`

## Backtest Result

Backtest command:

```powershell
F:/miniconda3/envs/pytorch/python -X utf8 -u run/v9_top5_refine.py --checkpoint checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt --output-dir backtest_results_exp_v9_top5_refine_topfocus_w005_topic
```

Top raw-return configs:

| mode | ann_raw | sharpe_raw | mdd_raw | ann_neu | sharpe_neu | mdd_neu |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simple_long_top050 | 33.68% | 1.39 | 18.71% | 24.46% | 1.42 | 14.61% |
| simple_long_top040 | 33.41% | 1.36 | 19.13% | 24.03% | 1.38 | 15.48% |
| simple_long_top060 | 33.19% | 1.38 | 18.99% | 24.08% | 1.41 | 14.44% |
| top050_hold075 | 32.22% | 1.38 | 18.88% | 23.46% | 1.40 | 14.92% |

Comparison:

- Previous V9-only best: `simple_long_top040`, ann `30.80%`, Sharpe `1.24`, MDD `20.20%`.
- Failed high top-focus run (`weight=0.02`): best ann `25.52%`.
- This run improves raw long-only annualized return by about `+2.88 pp` vs previous V9-only best, and reduces max drawdown by about `1.49 pp`.

## Conclusion

The checkpoint selection condition should not rely only on full cross-section `alpha_IC` for long-only usage. The first successful setting is:

- Train with mild top-focus loss: `top_focus_loss_weight=0.005`.
- Select checkpoint with a top-bucket metric: `topic_h5_top10`.
- For production selection, monitor both `topic_h5_top10` and `topret_h5_top5`; top IC alone can be noisy, while top return is closer to the deployed long-only portfolio.
