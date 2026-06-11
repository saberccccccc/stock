# Frozen Forward Strategy

Frozen: 2026-06-12

This manifest was selected using research data ending on 2026-05-18. Data
from 2026-05-19 onward must not be used to change these parameters, retrain
the checkpoint, or choose between the primary and fallback strategies.

## Primary

| Parameter | Value |
|---|---|
| Checkpoint | `checkpoints_exp_topfocus_w005_topic/ultimate_v7_best.pt` |
| Predictor | V9 `average_w3`, window 3 |
| Target fraction | `0.006` (approximately Top 30) |
| Hold fraction | `0.10` |
| Market timing | `legacy` |
| Portfolio values | CNY 500,000 and CNY 1,000,000 |
| Maximum stock weight | `0.05` |
| ADV window | 20 trading days |
| ADV participation cap | `0.05` |
| Minimum ADV | CNY 3,000,000 |
| Board lot | 100 shares |
| Minimum commission | CNY 5 per trade |
| Commission rate | `0.0001` |
| Stamp tax rate | `0.0005`, sells only |
| Slippage rate | `0.0005` |
| Limit threshold | `0.095` |
| Execution | Next tradable close, no extra delay |

## Fallback

Use the same signal and execution settings with:

| Parameter | Value |
|---|---|
| Target fraction | `0.004` (approximately Top 20) |
| Hold fraction | `0.06` |

The fallback is not a parameter-search option after forward results are seen.
It is reserved for an operational preference for fewer names.

## Frozen Evidence

| Artifact | SHA256 |
|---|---|
| Checkpoint | `6850798A6E1522F592E1096B924F7A717B1C127AD887945E311DB14438CD4E0C` |
| Validation alpha | `A899976F3EFF1A9951103AA3ECFECC05B0C759695659670668911447BB952569` |
| Test alpha | `7075EBE1BAA456EAFA7763622FF716033569FAD0CA467653673AB42208B9AF3B` |

Primary validation results are in
`backtest_results_test_plan_share_ledger_candidate_sweep_val`.
Primary independent test results are in
`backtest_results_test_plan_share_ledger_primary_test`.
Stress results are in the three
`backtest_results_test_plan_share_ledger_primary_stress_*` directories.

## Forward Rules

1. Read forward observations only from `data/forward_raw`.
2. Start the forward ledger on 2026-05-19.
3. Do not merge forward rows into `data/raw`.
4. Do not rebuild training, validation, or test caches with forward rows.
5. Record every scheduled order, blocked order, fill, fee, and end-of-day
   position.
6. Report primary results on a fixed cadence even when performance is poor.
7. Any later model or parameter change starts a new named forward experiment;
   it cannot replace this ledger.
