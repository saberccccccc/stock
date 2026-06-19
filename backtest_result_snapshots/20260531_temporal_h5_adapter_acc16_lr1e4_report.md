# V10 h5-only adapter large-section probe

## Setup

- checkpoint: `checkpoints_exp\temporal_cross_alpha_h5_adapter_h192_t96_n4500_acc16_lr1e4_20260531.pt`
- horizon mode: h5 only
- hidden_dim: 192
- temporal_dim: 96
- temporal residual adapters: enabled
- adapter positions:
  - after temporal fusion, before industry/rank features
  - after `trans1`
  - after `trans2`, before alpha heads
- adapter gate init: -3.0
- adapter dropout: 0.10
- train sample mode: target top/bottom/random
- sample mix: 10% true top, 10% true bottom, 80% random
- max train stocks: 4500
- batch size: 1
- accum steps: 16
- learning rate: 1e-4

## Resource Usage

- GPU memory: about 5.1GB / 6.0GB
- GPU temperature: about 61-66 C
- process RSS: about 7.9-8.3GB after epoch 1
- stderr: empty

The run was resource-stable but close to the GPU memory ceiling.

## Training Result

| epoch | train loss | capped val alpha |
|---:|---:|---:|
| 1 | -0.0953 | 0.0874 |
| 2 | -0.1269 | 0.0862 |

The run was stopped after epoch 2 because validation did not improve and was clearly below the better V10 probes around 0.099-0.103 capped alpha.

Saved checkpoint:

| field | value |
|---|---:|
| epoch | 1 |
| alpha | 0.087368 |
| h5 | 0.078854 |
| topbot_h5 | 0.250781 |
| topret_h5_top5 | 0.028813 |
| topic_h5_top5 | -0.007319 |
| topret_h5_top10 | 0.026395 |
| topic_h5_top10 | -0.000890 |

## Interpretation

The configuration is stable but too conservative/slow:

- `lr=1e-4` plus `accum_steps=16` gives only about 91 optimizer updates per epoch.
- Validation alpha stayed low even though training loss improved.
- Top return improved, but top internal IC remained weak.

This setup should not be continued as-is.

## Recommended Next Run

Use a middle ground:

- hidden_dim: 192
- temporal_dim: 96
- max_train_stocks: 4500 or 4000
- accum_steps: 4 or 8
- lr: 1e-4 to 2e-4
- h5-only + temporal adapters
- 10/10/80 sampler

The most likely issue is not model capacity but too few optimizer updates from `accum_steps=16`.
