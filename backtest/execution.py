"""Open-price order execution with cash, lot, cost, ADV, and limit rules."""

import numpy as np


def open_limit_trade_mask(open_df, close_df, day_pos, threshold):
    open_today = open_df.iloc[day_pos].to_numpy(dtype=np.float64)
    close_prev = (
        close_df.iloc[day_pos - 1].to_numpy(dtype=np.float64)
        if day_pos > 0
        else open_today
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = open_today / close_prev - 1.0
    gap[~np.isfinite(gap)] = np.nan
    return gap >= float(threshold), gap <= -float(threshold)


def apply_open_ledger_constraints(
    desired_weights,
    current_shares,
    cash,
    equity,
    open_df,
    close_df,
    adv_df,
    day_pos,
    args,
):
    prices = open_df.iloc[day_pos].to_numpy(dtype=np.float64)
    safe_prices = np.where(np.isfinite(prices) & (prices > 0), prices, 0.0)
    adv_row = adv_df.iloc[day_pos].to_numpy(dtype=np.float64)
    buy_block, sell_block = open_limit_trade_mask(
        open_df, close_df, day_pos, args.limit_threshold
    )
    lot_size = max(int(args.lot_size), 1)

    desired_shares = np.zeros_like(current_shares, dtype=np.float64)
    valid_price = np.isfinite(prices) & (prices > 0)
    desired_shares[valid_price] = (
        np.floor(
            desired_weights[valid_price]
            * float(equity)
            / prices[valid_price]
            / lot_size
        )
        * lot_size
    )
    share_delta = desired_shares - current_shares
    retained = (current_shares > 0) & (desired_shares > 0)
    has_resize = np.abs(share_delta) >= 1
    within_band = retained & has_resize & (
        np.abs(share_delta)
        <= max(float(args.rebalance_band), 0.0)
        * np.maximum(desired_shares, lot_size)
    )
    share_delta[within_band] = 0.0

    executed_shares = np.zeros_like(share_delta)
    blocked_buy = blocked_sell = adv_blocked = capped = 0
    lot_blocked = missing_adv = no_open = 0
    order = np.concatenate(
        (np.where(share_delta < 0)[0], np.where(share_delta > 0)[0])
    )

    for i in order:
        raw_shares = share_delta[i]
        if abs(raw_shares) < 1:
            continue
        price = prices[i]
        if not np.isfinite(price) or price <= 0:
            no_open += 1
            continue
        adv_cny = adv_row[i]
        if not np.isfinite(adv_cny) or adv_cny <= 0:
            missing_adv += 1
            continue
        if adv_cny < args.min_adv_cny:
            adv_blocked += 1
            continue
        if raw_shares > 0 and buy_block[i]:
            blocked_buy += 1
            continue
        if raw_shares < 0 and sell_block[i]:
            blocked_sell += 1
            continue

        max_trade_shares = (
            np.floor(
                float(args.adv_participation_cap)
                * adv_cny
                / price
                / lot_size
            )
            * lot_size
        )
        if max_trade_shares < lot_size:
            lot_blocked += 1
            continue
        candidate_shares = raw_shares
        if max_trade_shares < abs(raw_shares):
            capped += 1
            candidate_shares = np.sign(raw_shares) * max_trade_shares
        candidate_shares = np.sign(candidate_shares) * (
            np.floor(abs(candidate_shares) / lot_size) * lot_size
        )
        if abs(candidate_shares) < lot_size:
            lot_blocked += 1
            continue

        trade_value = abs(candidate_shares) * price
        commission = max(
            trade_value * float(args.commission_rate),
            float(args.min_commission_cny),
        )
        stamp_tax = (
            trade_value * float(args.stamp_tax_rate)
            if candidate_shares < 0
            else 0.0
        )
        slippage = trade_value * float(args.slippage_rate)
        fees = commission + stamp_tax + slippage

        if candidate_shares > 0:
            affordable = (
                np.floor(
                    max(cash - float(args.min_commission_cny), 0.0)
                    / (
                        price
                        * (
                            1.0
                            + float(args.commission_rate)
                            + float(args.slippage_rate)
                        )
                    )
                    / lot_size
                )
                * lot_size
            )
            if affordable < candidate_shares:
                candidate_shares = affordable
                if candidate_shares < lot_size:
                    lot_blocked += 1
                    continue
                trade_value = candidate_shares * price
                commission = max(
                    trade_value * float(args.commission_rate),
                    float(args.min_commission_cny),
                )
                stamp_tax = 0.0
                slippage = trade_value * float(args.slippage_rate)
                fees = commission + slippage
            cash -= trade_value + fees
        else:
            candidate_shares = -min(abs(candidate_shares), current_shares[i])
            if abs(candidate_shares) < lot_size:
                lot_blocked += 1
                continue
            trade_value = abs(candidate_shares) * price
            commission = max(
                trade_value * float(args.commission_rate),
                float(args.min_commission_cny),
            )
            stamp_tax = trade_value * float(args.stamp_tax_rate)
            slippage = trade_value * float(args.slippage_rate)
            fees = commission + stamp_tax + slippage
            cash += trade_value - fees
        executed_shares[i] = candidate_shares

    new_shares = current_shares + executed_shares
    executed_values = executed_shares * safe_prices
    desired_values = share_delta * safe_prices
    total_commission = total_stamp_tax = total_slippage = 0.0
    for shares, price in zip(executed_shares, prices):
        if abs(shares) < 1 or not np.isfinite(price):
            continue
        value = abs(shares) * price
        total_commission += max(
            value * float(args.commission_rate),
            float(args.min_commission_cny),
        )
        if shares < 0:
            total_stamp_tax += value * float(args.stamp_tax_rate)
        total_slippage += value * float(args.slippage_rate)
    total_cost = total_commission + total_stamp_tax + total_slippage
    info = {
        "blocked_buy": blocked_buy,
        "blocked_sell": blocked_sell,
        "adv_blocked": adv_blocked,
        "missing_adv": missing_adv,
        "no_open": no_open,
        "capped": capped,
        "lot_blocked": lot_blocked,
        "band_skipped": int(np.count_nonzero(within_band)),
        "turnover": float(np.sum(np.abs(executed_values)) / max(equity, 1.0)),
        "desired_turnover": float(
            np.sum(np.abs(desired_values)) / max(equity, 1.0)
        ),
        "executed_turnover": float(
            np.sum(np.abs(executed_values)) / max(equity, 1.0)
        ),
        "unfilled_turnover": float(
            np.sum(np.abs((share_delta - executed_shares) * safe_prices))
            / max(equity, 1.0)
        ),
        "cost": float(total_cost / max(equity, 1.0)),
        "commission": float(total_commission / max(equity, 1.0)),
        "stamp_tax": float(total_stamp_tax / max(equity, 1.0)),
        "slippage": float(total_slippage / max(equity, 1.0)),
    }
    return new_shares, cash, executed_shares, info
