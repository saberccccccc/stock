# Model Experiments Branch

Isolated from main project (`C:/Users/x/code/stock_prediction/deepseek_optimized`, branch `master`).

Research data is frozen through `2026-05-18`. Observations beginning
`2026-05-19` belong only to the forward-test set; see `RESEARCH_PROTOCOL.md`.

## Isolation

| Resource | Main | Experiments |
|----------|------|-------------|
| Checkpoints | `checkpoints/` | `checkpoints_exp/` |
| Backtest results | `backtest_results_*/` | `backtest_results_exp_*/` |
| Logs | `train_*.log` | `exp_*.log` |
| Code | `master` branch | `model-experiments` branch |

## Quick start

```bash
cd C:/Users/x/code/stock_prediction/deepseek_model_exp

# Run experiment (same commands as main, outputs go to exp_ directories)
python run/train.py --model v9 --epochs 25
python run/backtest.py --experiment ensemble

# Compare with main branch baseline (41.09% ann / 3.76 Sharpe)
```

## Experiment log

Use `experiments.log` to track each experiment:

```
[2026-05-19] exp-001: baseline - train.py --model v9 --epochs 25 → alpha_IC=0.099
[2026-05-19] exp-002: head LN+Drop - added LayerNorm+Dropout(0.1) to heads → alpha_IC=TBD
```

## Priority experiments

1. **Head LayerNorm + Dropout**: add `nn.LayerNorm(hidden_dim)` + `nn.Dropout(0.1)` to alpha_heads and horizon_heads
2. **Transformer depth 2 vs 4**: reduce `n_layers` 4→2 in `UltimateV7Model`
3. **FiLM gate**: modulate Transformer output with regime vector
4. **Style exposure penalty**: add `λ * |beta|` to loss
