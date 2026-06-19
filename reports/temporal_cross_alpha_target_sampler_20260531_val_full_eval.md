# Temporal full-stock validation

- checkpoint: `checkpoints_exp\temporal_cross_alpha_target_sampler_20260531.pt`
- checkpoint epoch: `2`
- split: `val`
- temporal_chunk: `512`
- meta: `cache\temporal_cross_section_temporal_v1_v13_config_key_all_techn_marke_s40_lb60_h10_t5_min30_hist365_tr20180101_va20240101_te20250101_endnone_seqauto_ec17f1_meta.pkl`

## Summary

| metric | value | count |
|---|---:|---:|
| alpha | 0.097444 | 242 |
| h1 | 0.054086 | 242 |
| h3 | 0.080193 | 242 |
| h5 | 0.089937 | 242 |
| h7 | 0.094658 | 242 |
| topret_h5_top5 | 0.019488 | 242 |
| topret_h5_top10 | 0.020359 | 242 |
| topic_h5_top5 | -0.001079 | 242 |
| topic_h5_top10 | -0.001246 | 242 |
| topbot_h5 | 0.244006 | 242 |
| topret_h7_top5 | 0.014549 | 242 |
| topret_h7_top10 | 0.015466 | 242 |
| topic_h7_top5 | 0.000180 | 242 |
| topic_h7_top10 | -0.001108 | 242 |
| topbot_h7 | 0.247965 | 242 |
| topbot_h1 | 0.141098 | 242 |
| topbot_h3 | 0.218409 | 242 |
| topic_h1_top10 | 0.006657 | 242 |
| topic_h1_top5 | 0.009714 | 242 |
| topic_h3_top10 | 0.000851 | 242 |
| topic_h3_top5 | 0.003373 | 242 |
| topret_h1_top10 | 0.015386 | 242 |
| topret_h1_top5 | 0.017043 | 242 |
| topret_h3_top10 | 0.021715 | 242 |
| topret_h3_top5 | 0.019962 | 242 |

## Checkpoint Sampled Validation

These are the metrics saved during capped validation in training, for comparison.

```json
{
  "alpha": 0.09961833682018414,
  "h1": 0.05817980379060867,
  "topbot_h1": 0.1628049587261332,
  "topret_h1_top5": 0.017725067903251148,
  "topic_h1_top5": -0.004586030602338421,
  "topret_h1_top10": 0.014469868828704165,
  "topic_h1_top10": 0.0019269234557780577,
  "h3": 0.0821604486597183,
  "topbot_h3": 0.23554658291722871,
  "topret_h3_top5": 0.02168037413527096,
  "topic_h3_top5": -0.009303844688935385,
  "topret_h3_top10": 0.02402927079800832,
  "topic_h3_top10": -0.006855243603452403,
  "h5": 0.0924152551333253,
  "topbot_h5": 0.2621034763691839,
  "topret_h5_top5": 0.01744545501568316,
  "topic_h5_top5": -0.009451477651814701,
  "topret_h5_top10": 0.023035055541802362,
  "topic_h5_top10": -0.009372044700888007,
  "h7": 0.0948545121788526,
  "topbot_h7": 0.25451628855344927,
  "topret_h7_top5": 0.00837940756630341,
  "topic_h7_top5": -0.0015952497493526058,
  "topret_h7_top10": 0.013693285316999999,
  "topic_h7_top10": -0.0069260601058650385
}
```
