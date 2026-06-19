# 2026-05-30 backtest code audit

## High severity

1. `run/backtest_trade_policy.py` mixes incompatible scores.

   Existing holdings are scored with `hold_prob`, while non-held names keep `alpha_score = 1 - rank_pct`. These two scores are not calibrated to the same scale, then the code selects the top K from the mixed vector.

   Relevant lines:

   - `run/backtest_trade_policy.py:165`
   - `run/backtest_trade_policy.py:176`
   - `run/backtest_trade_policy.py:180`

   Impact:

   The policy backtest becomes a daily full portfolio reconstruction, not a hold/sell policy. This explains the very high turnover and makes the 93% annualized result unreliable.

2. Initial and gap-period transaction costs can be trimmed away.

   Main engines charge execution cost on `entry_day`, but weights start from `entry_day + 1`. Later, returns are trimmed from the first non-zero-weight day.

   Relevant lines:

   - `backtest/engine.py:1384`
   - `backtest/engine.py:1393`
   - `backtest/engine.py:1501`
   - `backtest/layered_engine.py:304`
   - `backtest/layered_engine.py:356`
   - `backtest/layered_engine.py:184`

   Impact:

   The first build cost, and any cost paid after a flat gap, can be excluded from saved/evaluated returns. This biases results upward.

## Medium severity

3. The fixed-window production engine still has fixed holding semantics.

   `run_backtest_production` writes the newly executed weights over `hold_start..hold_end`, where `hold_start = entry_day + 1` and `exit_day = entry_day + future_len`.

   Relevant lines:

   - `backtest/engine.py:1172`
   - `backtest/engine.py:1393`

   Impact:

   It is fine for fixed-window testing, but not correct for "rank every day, sell only deteriorated names" strategies. For daily sell/hold logic, use a dedicated daily portfolio state engine.

4. Trade policy v1 rebalances the whole book daily.

   `target_w` is rebuilt from the selected top K every day, and `target_tradable` moves the whole current book toward that target.

   Relevant lines:

   - `run/backtest_trade_policy.py:180`
   - `run/backtest_trade_policy.py:187`
   - `run/backtest_trade_policy.py:193`

   Impact:

   It does not answer "whether current holdings should be kept". It answers "which K names rank highest under a mixed score today".

5. The hold/sell label is usable but narrow.

   Label:

   `label_hold = hold_ret_fwd >= replace_median - cost_buffer`

   Relevant lines:

   - `run/build_trade_policy_dataset.py:144`
   - `run/build_trade_policy_dataset.py:162`

   Impact:

   This is a reasonable v1 label for "keep vs replace", but it is not a buy model and does not learn the full portfolio action. It should be used with a retention-first backtest, not mixed-score top-K selection.

## Recommended fixes

1. Mark `trade_policy_v1` result as invalid/not comparable in reports.
2. Build `trade_policy_v2` as retention-first:
   - score current holdings with the policy,
   - sell only names below a learned threshold from train data,
   - keep the rest,
   - fill vacancies from the alpha top pool.
3. Fix cost alignment:
   - either write weights from `entry_day` and compute returns consistently,
   - or keep returns from the cost-only entry day when trimming active periods.
4. Use the fixed-window engine only for fixed holding tests.
5. Use a dedicated daily state engine for daily rank/hold/sell strategies.
