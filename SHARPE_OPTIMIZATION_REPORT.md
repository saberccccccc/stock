# Small-Account Sharpe Optimization

Updated: 2026-06-12

All model and parameter selection in this report uses observations no later
than 2026-05-18. Existing forward results were not used to choose the
candidate.

## Experiments

### Fundamental ablation

ROE, revenue growth, and their quarter-over-quarter changes were neutralized
at inference while preserving the checkpoint dimensions.

The best validation Sharpe fell to approximately 1.27 for CNY 500,000 and
1.25 for CNY 1,000,000. The complete PIT fundamental factors are therefore
retained.

### Market exposure

- No timing materially reduced validation Sharpe.
- Dynamic timing did not exceed the existing legacy rule.
- A 40% bear / 20% crash exposure improved validation Sharpe to 1.620 and
  1.669, but independent test Sharpe fell to 1.620 and 1.827. This parameter
  change was rejected.

### Raw-heavy signal blend

The first research candidate blended cross-sectional percentile ranks:

- 75% current-day V9 raw Alpha.
- 25% V9 three-day average Alpha.
- Target fraction `0.004`, approximately Top 20.
- Hold fraction `0.10`.
- Existing legacy market timing and execution assumptions.

## Results

| Period | CNY 500k annualized / Sharpe / MDD | CNY 1m annualized / Sharpe / MDD |
|---|---:|---:|
| Validation | 57.15% / 1.568 / 23.00% | 59.61% / 1.600 / 23.17% |
| Independent test | 65.94% / 2.312 / 12.80% | 68.42% / 2.304 / 13.33% |
| Validation, 2x costs | 41.63% / 1.249 | 44.83% / 1.302 |
| Validation, 3x costs | 26.37% / 0.893 | 30.04% / 0.971 |
| Validation, one extra day delay | 20.10% / 0.730 | 20.04% / 0.721 |

Compared with the current primary strategy, base validation and independent
test Sharpe improved for both account sizes. Two-times-cost Sharpe also
improved slightly. The candidate remains sensitive to delayed execution and
has higher turnover, so it starts as a separately named shadow-forward
experiment and does not replace the existing frozen ledger.

## Shadow Result

The candidate was also evaluated on the corrected 17-day forward holdout
without changing its parameters:

| Strategy | CNY 500k cumulative | CNY 1m cumulative |
|---|---:|---:|
| Existing average-w3 primary | -6.34% | -4.91% |
| Raw75 / average25 shadow | -8.89% | -9.38% |

The raw-heavy blend failed this shadow check and is not promoted. Its higher
turnover and faster response amplified losses in this market regime.

### Consensus signal

A second candidate used the geometric mean of the raw and three-day-average
cross-sectional percentile scores. Validation selected target fraction
`0.010` and hold fraction `0.06`; the independent test was then run once at
that fixed setting.

| Period | CNY 500k annualized / Sharpe / MDD | CNY 1m annualized / Sharpe / MDD |
|---|---:|---:|
| Validation | 53.80% / 1.584 / 17.42% | 56.66% / 1.598 / 18.22% |
| Independent test | 40.79% / 1.893 / 11.96% | 56.65% / 2.263 / 12.79% |
| Frozen primary test | - / 1.912 / - | - / 2.113 / - |

The CNY 1m result improved, but the CNY 500k result did not exceed the frozen
primary baseline. Because the intended capital range includes both account
sizes, the consensus candidate is rejected under the predeclared
two-account gate. It was not promoted to stress or forward testing.

## Decision

Neither blend replaces the frozen average-w3 primary strategy. The next
research round therefore focused on reducing small-account turnover and
execution drag rather than adding a faster signal.

### Rebalance band

The average-w3 signal and stock-selection rules were left unchanged. For
positions that remained selected, resizing trades were skipped while current
shares stayed within 20% of target shares. New entries, full exits, limit
rules, board lots, liquidity limits, and market exposure changes outside the
band remained executable.

The band was selected from `0%, 10%, 20%, 30%, 40%, 50%` on validation data.
Only the fixed 20% candidate was then run on the independent test.

| Period | CNY 500k annualized / Sharpe | CNY 1m annualized / Sharpe |
|---|---:|---:|
| Validation baseline | 53.27% / 1.507 | 56.39% / 1.543 |
| Validation, 20% band | 56.59% / 1.575 | 58.83% / 1.597 |
| Independent test baseline | 44.55% / 1.912 | 54.25% / 2.113 |
| Independent test, 20% band | 45.19% / 1.939 | 55.42% / 2.159 |
| Validation, 2x costs | 44.46% / 1.322 | 48.88% / 1.395 |
| Validation, 3x costs | 32.73% / 1.055 | 39.01% / 1.183 |
| Validation, one extra day delay | 34.12% / 1.083 | 33.14% / 1.047 |

The candidate exceeded the matching baseline for both account sizes in base,
2x-cost, 3x-cost, and delayed-execution tests. Validation turnover fell from
`0.300` to `0.292` for CNY 500k and from `0.306` to `0.296` for CNY 1m.

A separately named 17-day forward shadow was run without changing the chosen
20% value. CNY 500k improved from `-6.34%` to `-6.07%`, while CNY 1m declined
from `-4.91%` to `-6.18%`. This short mixed result is recorded but is not used
to retune the band or rewrite the existing frozen ledger.

The 20% rebalance band is the first candidate in this report to improve both
capital levels on validation and independent test. It is accepted as the
next execution-policy candidate, while the original forward campaign remains
unchanged until its fixed review milestone.

### Follow-up execution searches

Three follow-up ideas were tested on validation data and rejected:

- A minimum CNY resize amount for retained positions had no effect at CNY
  1,000 and reduced CNY 500k return and Sharpe from CNY 2,000 upward.
- Tilting weights toward the highest-ranked names reduced return and Sharpe
  for both account sizes. Mild reverse tilting improved only CNY 500k and
  weakened CNY 1m; larger reverse tilts weakened both.
- An account-specific 10% band for CNY 1m slightly improved the independent
  base test to 55.58% annualized and 2.161 Sharpe, versus 55.42% and 2.159
  for the common 20% band. However, its 2x-cost, 3x-cost, and delayed Sharpe
  were 1.385, 1.162, and 1.027, all below the 20% band's 1.395, 1.183, and
  1.047.

The common 20% band remains preferred. The small base-test difference does
not justify weaker stress performance or separate live rules by account
size.

## Evidence

| Artifact | SHA256 |
|---|---|
| Validation blended Alpha | `2FA75E80C073096150CCA49368FE30EBBC79366B7E0223504D6AA183A8EF61A9` |
| Test blended Alpha | `F20A9839378BA67F2483CC927495F60EC1C859947AE78B8135F5924274AE225C` |
| Consensus validation Alpha | `5A4D349B6A393438B7951CA3EF97D6DDFDC27C9F1F697F80E78B8459188D31DF` |
| Consensus test Alpha | `23FF2A7D6B7D6539D28150469C645E9D7B6FECFF11EDEBDF49EC947BD29C2DF9` |
| Rebalance-band validation summary | `113E1258D49282DB490F108AE8E61357B3E6A09E412E046DB82B5F7E86C9AD18` |
| Rebalance-band test summary | `BF7140AA6C91F5CFA09BA8485457A424CFF140662B537D3708C99471E3B5A755` |
| Rebalance-band forward summary | `E6962CA0E380B6111763B21F60F1FC153E5CADF4E2D37ECD66CE041BFCD4E745` |
