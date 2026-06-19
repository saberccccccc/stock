"""Reusable label-free market breadth and target-risk Alpha overlays."""

from pathlib import Path

import pandas as pd


def compute_breadth(data_dir, start_date, end_date):
    frames = []
    for path in Path(data_dir).glob("*.csv"):
        if not path.name.endswith((".SZ.csv", ".SH.csv", ".BJ.csv")):
            continue
        try:
            frame = pd.read_csv(path, usecols=["trade_date", "close"])
        except Exception:
            continue
        if frame.empty:
            continue
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        frame = frame[
            (frame["trade_date"] >= start_date)
            & (frame["trade_date"] <= end_date)
        ]
        if len(frame) < 2:
            continue
        frame = frame.sort_values("trade_date")
        returns = pd.to_numeric(frame["close"], errors="coerce").pct_change()
        rows = pd.DataFrame(
            {
                "date": frame["trade_date"].dt.normalize(),
                "up": returns > 0,
                "valid": returns.notna(),
            }
        )
        frames.append(rows[rows["valid"]])
    if not frames:
        raise ValueError(f"No stock data found in {data_dir}")
    return (
        pd.concat(frames, ignore_index=True)
        .groupby("date", sort=True)
        .agg(up_ratio=("up", "mean"), breadth_n=("up", "size"))
        .reset_index()
    )


def rolling_breadth_map(breadth, window):
    window = int(window)
    if window <= 0:
        raise ValueError("breadth window must be positive")
    column = f"up_ma{window}"
    breadth[column] = breadth["up_ratio"].rolling(window).mean()
    values = breadth[column]
    return {
        pd.Timestamp(date).normalize(): value
        for date, value in zip(breadth["date"], values)
    }


def shrink_target_row(row, triggered, base_target_frac, risk_target_frac):
    codes = list(row.get("codes", []))
    alpha = list(row.get("alpha", []))
    n = len(codes)
    if n == 0:
        return row, None

    base_target_frac = float(base_target_frac)
    risk_target_frac = float(risk_target_frac)
    if not 0 < risk_target_frac <= base_target_frac:
        raise ValueError("risk target must be in (0, base target]")
    base_n = max(int(n * base_target_frac), 1)
    risk_n = max(int(n * risk_target_frac), 1)
    keep_n = risk_n if triggered else base_n

    if triggered and risk_n < base_n:
        order = (
            list(range(risk_n))
            + list(range(base_n, n))
            + list(range(risk_n, base_n))
        )
        codes = [codes[i] for i in order]
        alpha = (
            [alpha[i] for i in order]
            if len(alpha) == n
            else [float(n - i) for i in range(n)]
        )

    output = dict(row)
    output["codes"] = codes
    output["alpha"] = alpha
    details = {
        "triggered": bool(triggered),
        "base_target_frac": base_target_frac,
        "risk_target_frac": risk_target_frac,
        "effective_target_frac": risk_target_frac if triggered else base_target_frac,
        "base_n": int(base_n),
        "risk_n": int(risk_n),
        "keep_n": int(keep_n),
    }
    return output, details


def attach_breadth_market_multiplier(
    row,
    triggered,
    breadth_window,
    breadth_below,
    breadth_value,
    risk_market_mult,
):
    output = dict(row)
    output["breadth_market_transform"] = {
        "triggered": bool(triggered),
        "breadth_window": int(breadth_window),
        "breadth_below": float(breadth_below),
        "breadth_value": None if pd.isna(breadth_value) else float(breadth_value),
        "effective_market_mult": float(risk_market_mult) if triggered else 1.0,
    }
    return output
