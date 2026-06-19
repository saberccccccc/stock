# Temporal full-stock validation

- checkpoint: `checkpoints_exp\temporal_cross_alpha_probe_capped_20260531.pt`
- checkpoint epoch: `5`
- split: `val`
- temporal_chunk: `512`
- meta: `cache\temporal_cross_section_temporal_v1_v13_config_key_all_techn_marke_s40_lb60_h10_t5_min30_hist365_tr20180101_va20240101_te20250101_endnone_seqauto_ec17f1_meta.pkl`

## Summary

| metric | value | count |
|---|---:|---:|
| alpha | 0.096135 | 242 |
| h1 | 0.055115 | 242 |
| h3 | 0.079694 | 242 |
| h5 | 0.089697 | 242 |
| h7 | 0.091821 | 242 |
| topret_h5_top5 | 0.009351 | 242 |
| topret_h5_top10 | 0.011483 | 242 |
| topic_h5_top5 | -0.009183 | 242 |
| topic_h5_top10 | -0.007784 | 242 |
| topbot_h5 | 0.229237 | 242 |
| topret_h7_top5 | 0.000002 | 242 |
| topret_h7_top10 | 0.005003 | 242 |
| topic_h7_top5 | -0.009336 | 242 |
| topic_h7_top10 | -0.010189 | 242 |
| topbot_h7 | 0.232735 | 242 |
| topbot_h1 | 0.148967 | 242 |
| topbot_h3 | 0.206243 | 242 |
| topic_h1_top10 | 0.000138 | 242 |
| topic_h1_top5 | 0.004379 | 242 |
| topic_h3_top10 | -0.003449 | 242 |
| topic_h3_top5 | -0.006263 | 242 |
| topret_h1_top10 | 0.019188 | 242 |
| topret_h1_top5 | 0.019821 | 242 |
| topret_h3_top10 | 0.014576 | 242 |
| topret_h3_top5 | 0.015047 | 242 |

## Checkpoint Sampled Validation

These are the metrics saved during capped validation in training, for comparison.

```json
{
  "alpha": 0.09905777169336381,
  "h1": 0.05764073273669734,
  "topbot_h1": 0.165882676678194,
  "topret_h1_top5": 0.027070729832160374,
  "topic_h1_top5": 0.0045387582280016605,
  "topret_h1_top10": 0.01862906141703071,
  "topic_h1_top10": 0.01027650717805001,
  "h3": 0.08155676122383199,
  "topbot_h3": 0.22107792985894956,
  "topret_h3_top5": 0.025563153575085155,
  "topic_h3_top5": -0.008054778373445428,
  "topret_h3_top10": 0.01671907757861109,
  "topic_h3_top10": 0.00633210870562302,
  "h5": 0.09352234599990397,
  "topbot_h5": 0.24778071555024214,
  "topret_h5_top5": 0.015028441713522534,
  "topic_h5_top5": -0.008563488397832891,
  "topret_h5_top10": 0.017649586891422573,
  "topic_h5_top10": -0.006748040536340818,
  "h7": 0.093572662665454,
  "topbot_h7": 0.23750185940322305,
  "topret_h7_top5": 0.0022326443850184015,
  "topic_h7_top5": -0.011589622371461012,
  "topret_h7_top10": 0.0047306820953722795,
  "topic_h7_top10": -0.006697312279699466
}
```
