# Alpha Execution Filter Result

Date: 2026-06-13

## Decision

Keep the existing V9 `average_w3`, Top30, 10% hold threshold and 20%
rebalance-band strategy unchanged as the operational baseline. Add the 9.5%
signal-day return filter as the leading research execution candidate.

Do not promote the strictly isolated epoch-9 checkpoint. Its 2024 strict
validation result remained far below the existing strategy even after the
same execution transform.

## Diagnosis

Top30 signal behavior on 2024 validation:

| Alpha | Mean signal-day return | Share at least 7% | Share at least 9.5% | Prior-day Top30 overlap |
|---|---:|---:|---:|---:|
| Frozen V9 average-w3 | -0.05% | 4.20% | 2.88% | 50.71% |
| Old topstable epoch 9 | 0.02% | 5.07% | 3.72% | 52.23% |
| Strictly isolated epoch 9 | 0.40% | 6.46% | 5.00% | 49.16% |

The isolated model concentrated more heavily in stocks that had already
surged on the signal date and had less stable membership. This is consistent
with its high blocked-buy count and weak next-day executable return.

## Candidate Screen

| Candidate | CNY 500k ann / Sharpe | CNY 1m ann / Sharpe | Decision |
|---|---:|---:|---|
| Baseline | 56.59% / 1.575 | 58.83% / 1.597 | Reference |
| Demote return at least 9.5% | 59.56% / 1.625 | 60.81% / 1.622 | Pass |
| Demote return at least 7.0% | 58.09% / 1.619 | 60.20% / 1.633 | Reject: higher turnover and drawdown |
| 75% current + 25% prior-two rank | 36.68% / 1.162 | 36.50% / 1.142 | Reject |
| Rank stability plus 9.5% filter | 40.05% / 1.240 | 41.94% / 1.263 | Reject |
| Isolated model plus combined filter | 17.87% / 0.715 | 18.20% / 0.714 | Reject |

For the passing candidate, blocked buys fell from 78 to 70 at CNY 500k and
from 80 to 68 at CNY 1m. Unfilled turnover also fell. Maximum drawdown changed
from 15.04% to 15.52% and from 15.48% to 16.13%, both inside the two-point
gate.

## Stress Results

| Scenario | CNY 500k baseline / candidate Sharpe | CNY 1m baseline / candidate Sharpe |
|---|---:|---:|
| Base | 1.575 / 1.625 | 1.597 / 1.622 |
| 2x costs | 1.406 / 1.452 | 1.415 / 1.439 |
| Extra trading-day lag | 1.083 / 1.224 | 1.047 / 1.221 |
| 3% ADV cap | 1.575 / 1.625 | 1.597 / 1.622 |

## Historical Confirmation

The fixed candidate was run once on 2025-01-01 through 2026-05-18:

| Capital | Annualized | Sharpe | Maximum drawdown |
|---:|---:|---:|---:|
| CNY 500k | 58.15% | 2.378 | 11.90% |
| CNY 1m | 66.69% | 2.489 | 12.55% |

This is a strategy-level historical confirmation, not a strictly independent
model result. The old V9 checkpoint was selected under the contaminated
default split that included later dates in checkpoint validation.

## Artifacts

- Validation Alpha:
  `C:\Users\x\Documents\股票预测\alpha_execution_screen_20260613\frozen_maxret095.jsonl`
- Historical confirmation Alpha:
  `C:\Users\x\Documents\股票预测\alpha_execution_screen_20260613\frozen_maxret095_test.jsonl`
- Validation Alpha SHA256:
  `CFB65BC54C507A2EF6F1715E0C2D00772EC502FDCC997978B26DF911799D0297`
- Confirmation Alpha SHA256:
  `99C040312AFF3D2BA1A212E71C8A4A8C8E4D72CB859B755FCE6F38CFD5D665AB`
- Strictly isolated checkpoint SHA256:
  `5F9CB20F9CC0648B614EEE47FA636C88C04F907AD98919783F9E8481F81AAA8C`

## Next Model Round

The next training experiment must retain the purged date split and address
execution behavior directly:

1. Save all epoch checkpoints instead of selecting only by normalized-label
   IC.
2. Evaluate each epoch on raw 2024 returns with the strict next-day execution
   simulator.
3. Reduce emphasis on one-day momentum and increase medium-horizon ranking
   consistency.
4. Use 2022, 2023 and 2024 rolling validation slices before freezing a
   checkpoint.
5. Keep batch size 4, validation batch 1, accumulation 4 and memmap trim 64,
   which completed safely on 16 GB RAM and 8 GB VRAM.
